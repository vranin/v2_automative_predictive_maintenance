import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from textblob import TextBlob
from openai import RateLimitError  # for safe fallback
import streamlit as st
from agents.customer_engagement_agent import CustomerEngagementAgent
from agents.diagnosis_agent import DiagnosisAgent

ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class FeedbackAgent:
    def __init__(self, use_llm: bool = True):
        # Toggle LLM usage with use_llm; when False, everything is rule‑based
        self.use_llm = use_llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3) if use_llm else None
        self.feedback_df = self._load_feedback()
        self.diagnosis_agent = DiagnosisAgent()
        self.customer_agent = CustomerEngagementAgent()

    def _load_feedback(self):
        feedback_path = "data/feedback.csv"
        if Path(feedback_path).exists():
            df = pd.read_csv(feedback_path)
            df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
            df["user_rating"] = pd.to_numeric(df["user_rating"], errors="coerce")
            df["sentiment"] = df["comments"].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            return df
        return pd.DataFrame()

    def request_feedback(
        self, vehicle_name: str, customer_name: str, diagnosis: dict | None = None
    ):
        diagnosis = diagnosis or {}
        risk = diagnosis.get("risk_level", "unknown")
        failure = diagnosis.get("predicted_failure", "N/A")

        base_text = (
            f"Hi {customer_name}, how did we do with your recent service for {vehicle_name}? "
            "Please rate us 1–5 stars and share any comments."
        )

        prompt = f"""
Recent service feedback request for {customer_name} - {vehicle_name}
Current diagnosis: {risk} risk ({failure})

Generate personalized request (2 sentences) based on the above:
"{base_text}"
"""

        if self.use_llm and self.llm is not None:
            try:
                response = self.llm.invoke(prompt)
                text = response.content.strip()
            except RateLimitError:
                text = base_text
        else:
            text = base_text

        return {
            "prompt": text,
            "vehicle_name": vehicle_name,
            "customer_name": customer_name,
            "feedback_id": f"FB{2025*10000 + pd.Timestamp.now().dayofyear * 100 + len(self.feedback_df) + 1}",
            "diagnosis_context": diagnosis,
        }

    def process_feedback(
        self,
        vehicle_name: str,
        customer_name: str,
        rating: float,
        issue_resolved: str,
        comments: str,
        service_center: str = "VESIT Service",
    ):
        diagnosis = self.diagnosis_agent.continuous_monitor(vehicle_name)
        risk = diagnosis.get("risk_level", "unknown")
        failure = diagnosis.get("predicted_failure", "N/A")

        notes_prompt = f"""
Vehicle: {vehicle_name}, Rating: {rating}, Issue Resolved: {issue_resolved}
Customer: "{comments}"
Diagnosis: {failure} (risk: {risk})

Generate:
- mechanic_notes (1 sentence)
- rca_capa_reference (format RCAYYYYMMDD)
"""

        if self.use_llm and self.llm is not None:
            try:
                notes_response = self.llm.invoke(notes_prompt)
                lines = notes_response.content.split("\n")
                mechanic_notes = lines[0].strip() if lines else "Customer feedback noted."
                rca_ref = (
                    lines[1].strip()
                    if len(lines) > 1
                    else f"RCA{pd.Timestamp.now().strftime('%Y%m%d')}"
                )
            except RateLimitError:
                mechanic_notes = "Customer feedback noted; review vehicle history and recent repairs."
                rca_ref = f"RCA{pd.Timestamp.now().strftime('%Y%m%d')}"
        else:
            mechanic_notes = "Customer feedback noted; review vehicle history and recent repairs."
            rca_ref = f"RCA{pd.Timestamp.now().strftime('%Y%m%d')}"

        new_feedback = {
            "feedback_id": f"FB{2025*10000 + pd.Timestamp.now().dayofyear * 100 + len(self.feedback_df) + 1}",
            "vehicle_name": vehicle_name,
            "service_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "user_rating": rating,
            "issue_resolved": issue_resolved,
            "comments": comments,
            "center_feedback": f"Customer satisfied: {rating}/5 - {risk} risk monitored",
            "mechanic_notes": mechanic_notes,
            "rca_capa_reference": rca_ref,
        }

        self.feedback_df = pd.concat(
            [self.feedback_df, pd.DataFrame([new_feedback])], ignore_index=True
        )
        self.feedback_df.to_csv("data/feedback.csv", index=False)

        return {
            "saved": True,
            "feedback_id": new_feedback["feedback_id"],
            "sentiment": TextBlob(comments).sentiment.polarity,
            "needs_followup": rating < 3 or risk in ["high", "critical"],
        }

    def aggregate_feedback(self, vehicle_name: str = None):
        df = self.feedback_df.copy()
        if vehicle_name:
            df = df[df["vehicle_name"] == vehicle_name]

        if df.empty:
            return {"message": "No feedback data"}

        metrics = {
            "total_feedback": len(df),
            "average_rating": df["user_rating"].mean(),
            "issues_not_resolved": len(df[df["issue_resolved"] == "No"]),
            "low_ratings": len(df[df["user_rating"] < 3]),
            "recent_service_avg": df.tail(5)["user_rating"].mean()
            if len(df) >= 5
            else 0,
            "top_issues": df["comments"]
            .str.lower()
            .str.extract("(brake|battery|oil)")
            .fillna("")
            .value_counts()
            .to_dict(),
        }

        insight_prompt = f"""
FEEDBACK ANALYSIS ({vehicle_name or 'ALL'}):
Avg Rating: {metrics['average_rating']:.1f}
Unresolved: {metrics['issues_not_resolved']}
Low ratings: {metrics['low_ratings']}

Actionable insight (1 sentence):
"""

        if self.use_llm and self.llm is not None:
            try:
                metrics["ai_insight"] = self.llm.invoke(insight_prompt).content.strip()
            except RateLimitError:
                metrics["ai_insight"] = (
                    "Focus on vehicles with unresolved issues and low ratings (<3) "
                    "for proactive follow-up calls."
                )
        else:
            metrics["ai_insight"] = (
                "Focus on vehicles with unresolved issues and low ratings (<3) "
                "for proactive follow-up calls."
            )

        return metrics

    def get_followup_alerts(self):
        priority_df = self.feedback_df[
            (self.feedback_df["user_rating"] < 3)
            | (self.feedback_df["issue_resolved"] == "No")
        ].tail(10)

        alerts = []
        for _, row in priority_df.iterrows():
            diagnosis = self.diagnosis_agent.continuous_monitor(row["vehicle_name"])
            risk = diagnosis.get("risk_level", "unknown")
            customer_msg = self.customer_agent.recommend_action(
                row["vehicle_name"], "Customer"
            )

            alerts.append(
                {
                    "feedback_id": row["feedback_id"],
                    "vehicle_name": row["vehicle_name"],
                    "rating": row["user_rating"],
                    "issue": str(row["comments"])[:50] + "...",
                    "status": row["issue_resolved"],
                    "rca_ref": row["rca_capa_reference"],
                    "diagnosis_risk": risk,
                    "followup_message": customer_msg,
                }
            )

        return alerts


@tool
def process_service_feedback(
    vehicle_name: str, rating: int, resolved: str, comments: str
) -> str:
    """Process customer feedback after service"""
    agent = FeedbackAgent()
    result = agent.process_feedback(vehicle_name, "Customer", rating, resolved, comments)
    return f"✅ Feedback {result['feedback_id']} saved for {vehicle_name}"
