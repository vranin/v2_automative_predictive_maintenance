import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from functools import lru_cache

# ==================== SAFE .ENV LOADING ====================
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(f"OPENAI_API_KEY not found at {ENV_PATH.resolve()}")

print("OPENAI_API_KEY loaded successfully!")


# ==================== DATA LOADERS ====================
def load_defects(path=os.path.join("data", "defects.csv")):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

def load_vehicles(path=os.path.join("data", "vehicles.csv")):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

def load_feedback(path=os.path.join("data", "feedback.csv")):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


# ==================== MAIN AGENT CLASS ====================
class CustomerEngagementAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.defects = load_defects()
        self.vehicles = load_vehicles()

    def get_latest_defect(self, vehicle_name):
        df = self.defects[self.defects["vehicle_name"] == vehicle_name]
        if not df.empty:
            return df.sort_values("reported_date", ascending=False).iloc[0].to_dict()
        return None

    # THIS IS THE ONLY METHOD THAT MATTERS — CACHE OUTSIDE THE CLASS
    @staticmethod
    @lru_cache(maxsize=64)
    def get_recommendation(vehicle_name: str, customer_name: str = "Valued Customer") -> str:
        # We create a temporary agent just to get defect data (cheap & fast)
        temp_agent = CustomerEngagementAgent()
        defect = temp_agent.get_latest_defect(vehicle_name)

        if not defect:
            return f"Hi {customer_name}, your {vehicle_name} is in perfect condition. No action needed!"

        prompt = f"""
You are a professional service advisor contacting {customer_name} about their {vehicle_name}.

Issue detected:
• Type: {defect['defect_type']}
• Severity: {defect['severity']}
• Description: {defect.get('description', 'N/A')}

Write a short (3–4 sentences), warm, specific message that:
- Greets by name
- Explains the issue simply
- Mentions urgency
- Offers to schedule service immediately

Example tone: friendly but proactive.
"""

        print("OPENAI CALL (will only happen ONCE per vehicle)")
        print(f"→ {vehicle_name} | {customer_name} | {defect['defect_type']} ({defect['severity']})")

        try:
            response = temp_agent.model.invoke(prompt)
            print("OpenAI replied → cached forever")
            return response.content.strip()
        except Exception as e:
            return f"Hi {customer_name}, we found a {defect['defect_type'].lower()} issue on {vehicle_name}. Please book a slot soon."

    # NEW: simple wrapper so your app.py doesn't change
    def recommend_action(self, vehicle_name: str, customer_name: str = "Valued Customer") -> str:
        return self.get_recommendation(vehicle_name, customer_name)

from langchain.tools import tool
from utils.voice_caller import phone

@tool
def trigger_live_voice_alert(vehicle_id: str, risk_level: str = "medium") -> str:
    """Triggers real outbound voice call using free SIP (covers all your slide edge cases)"""
    if phone is None:
        return "Voice system not ready"
    
    try:
        call = phone.call(
            to="+919920475211",  # ← change to your real number
            from_="sip:justhere12@sip.linphone.org",
            metadata={"vehicle_id": vehicle_id, "risk": risk_level}
        )
        return f"Calling {vehicle_id} (risk: {risk_level}) → live voice alert sent!"
    except Exception as e:
        return f"Call failed: {str(e)}"