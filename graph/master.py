import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from datetime import datetime
from agents.diagnosis_agent import DiagnosisAgent
from agents.customer_engagement_agent import CustomerEngagementAgent
from agents.scheduling_agent import SchedulingAgent
from agents.feedback_agent import FeedbackAgent
from utils.security_tools import agent_ueba

load_dotenv()


class AgentState(TypedDict):
    vehicle_name: str
    customer_name: str
    customer_location: tuple
    diagnosis: dict
    engagement_message: str
    available_slots: list
    booking_result: dict
    feedback_request: dict
    feedback_response: dict
    actions_taken: List[str]
    final_status: str
    priority: str
    customer_wants_booking: bool


class MasterOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.diagnosis_agent = DiagnosisAgent()
        self.customer_agent = CustomerEngagementAgent()
        self.scheduling_agent = SchedulingAgent()
        self.feedback_agent = FeedbackAgent()
        self.checkpointer = MemorySaver()
        self.graph = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("diagnose", self.diagnose_node)
        workflow.add_node("engage", self.engage_node)
        workflow.add_node("schedule", self.schedule_node)
        workflow.add_node("feedback", self.feedback_loop_node)
        workflow.add_node("voice_alert", self.voice_alert_node)
        workflow.add_node("route_decision", self.route_decision_node)

        workflow.set_entry_point("diagnose")
        workflow.add_edge("diagnose", "route_decision")
        workflow.add_conditional_edges(
            "route_decision",
            lambda s: s["next_step"],
            {
                "engage": "engage",
                "schedule": "schedule",
                "voice_alert": "voice_alert",
                "feedback": "feedback",
                "complete": END,
            },
        )
        workflow.add_edge("engage", "route_decision")
        workflow.add_edge("schedule", "feedback")
        workflow.add_edge("voice_alert", END)
        workflow.add_edge("feedback", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def diagnose_node(self, state, config):
        vehicle_name = state["vehicle_name"]

        agent_ueba.monitor_agent_call(
            source_agent="MasterOrchestrator",
            target_agent="DiagnosisAgent",
            vehicle_name=vehicle_name,
            action_type="diagnosis",
            data_size=0,
            response_time=0,
        )

        vehicles_df = self.diagnosis_agent.vehicles
        row = vehicles_df[vehicles_df["vehicle_name"] == vehicle_name]

        if row.empty:
            priority = "low"
            return {
                **state,
                "diagnosis": {
                    "vehicle_id": vehicle_name,
                    "anomaly_score": 0.0,
                    "risk_level": "low",
                    "predicted_failure": "none",
                    "urgency": "14d",
                    "battery_internal": 12.6,
                    "alarms_triggered": 0,
                },
                "priority": priority,
            }

        vehicle_id = str(row.iloc[0]["vehicle_id"])
        diagnosis = self.diagnosis_agent.continuous_monitor(vehicle_id)

        risk_level = str(diagnosis.get("risk_level", "low")).lower()
        anomaly = float(diagnosis.get("anomaly_score", 0.0))

        if risk_level in ["critical", "high"]:
            priority = risk_level
        elif risk_level == "medium" or anomaly > 0.5:
            priority = "medium"
        else:
            priority = "low"

        return {
            **state,
            "diagnosis": diagnosis,
            "priority": priority,
        }


    def engage_node(self, state: AgentState):
        agent_ueba.monitor_agent_call(
            source_agent="MasterOrchestrator",
            target_agent="CustomerEngagementAgent",
            vehicle_name=state["vehicle_name"],
            action_type="engagement",
            data_size=0,
            response_time=0,
        )

        print(f"📞 Engaging {state['customer_name']}...")
        message = self.customer_agent.recommend_action(
            state["vehicle_name"], state["customer_name"]
        )
        customer_response = self._simulate_customer_response(state["priority"])

        return {
            **state,
            "engagement_message": message,
            "customer_wants_booking": customer_response["wants_booking"],
            "actions_taken": state["actions_taken"] + ["Engagement message sent"],
        }

    def schedule_node(self, state: AgentState):
        agent_ueba.monitor_agent_call(
            source_agent="MasterOrchestrator",
            target_agent="SchedulingAgent",
            vehicle_name=state["vehicle_name"],
            action_type="scheduling",
            data_size=0,
            response_time=0,
        )

        print(f"📅 Scheduling {state['vehicle_name']}...")
        slots = None

        if state["priority"] in ["high", "critical"]:
            result = self.scheduling_agent.auto_reserve_high_risk(
                state["vehicle_name"], state["customer_location"]
            )
        else:
            slots = self.scheduling_agent.get_available_slots(
                state["vehicle_name"], state["customer_location"]
            )
            if slots:
                result = self.scheduling_agent.book_appointment(
                    state["vehicle_name"],
                    slots[0]["slot_id"],
                    state["customer_name"],
                    state["priority"],
                )
            else:
                result = {"status": "error", "message": "No slots available"}

        return {
            **state,
            "available_slots": slots or [],
            "booking_result": result,
            "actions_taken": state["actions_taken"]
            + [f"Booking: {result.get('status', 'failed')}"],
        }

    def feedback_loop_node(self, state: AgentState):
        agent_ueba.monitor_agent_call(
            source_agent="MasterOrchestrator",
            target_agent="FeedbackAgent",
            vehicle_name=state["vehicle_name"],
            action_type="feedback",
            data_size=0,
            response_time=0,
        )

        print("📝 Collecting feedback...")
        feedback_req = self.feedback_agent.request_feedback(
            state["vehicle_name"],
            state["customer_name"],
            state.get("diagnosis", {}),
        )
        feedback = self._simulate_feedback_response()

        self.feedback_agent.process_feedback(
            state["vehicle_name"],
            state["customer_name"],
            feedback["rating"],
            feedback["resolved"],
            feedback["comments"],
        )

        return {
            **state,
            "feedback_request": feedback_req,
            "feedback_response": feedback,
            "final_status": "COMPLETE",
            "actions_taken": state["actions_taken"] + ["Feedback processed"],
        }

    def voice_alert_node(self, state: AgentState):
        agent_ueba.monitor_agent_call(
            source_agent="MasterOrchestrator",
            target_agent="SchedulingAgent",
            vehicle_name=state["vehicle_name"],
            action_type="voice_alert",
            data_size=0,
            response_time=0,
        )

        print(f"🚨 EMERGENCY for {state['vehicle_name']}!")
        alert_result = "Voice call triggered"
        booking = self.scheduling_agent.auto_reserve_high_risk(
            state["vehicle_name"], state["customer_location"]
        )

        return {
            **state,
            "voice_alert_result": alert_result,
            "emergency_booking": booking,
            "final_status": "CRITICAL_EMERGENCY",
            "actions_taken": state["actions_taken"]
            + ["Emergency voice alert + booking"],
        }

    def route_decision_node(self, state: AgentState):
        priority = state.get("priority", "low")
        wants_booking = state.get("customer_wants_booking", False)

        if priority == "critical":
            next_step = "schedule"
        elif priority == "high":
            next_step = "schedule"
        elif priority == "medium":
            next_step = "engage" if not wants_booking else "schedule"
        else:
            next_step = "feedback"

        return {
            **state,
            "next_step": next_step,
        }

    def _simulate_customer_response(self, priority):
        responses = {
            "low": {"wants_booking": False},
            "medium": {"wants_booking": True},
            "high": {"wants_booking": True},
            "critical": {"wants_booking": True},
        }
        return responses.get(priority, {"wants_booking": False})

    def _simulate_feedback_response(self):
        return {"rating": 4.5, "resolved": "Yes", "comments": "Great service!"}

    def run_autonomous_workflow(
        self,
        vehicle_name: str,
        customer_name: str = "Customer",
        location: tuple = (19.0760, 72.8777),
    ):
        initial_state = {
            "vehicle_name": vehicle_name,
            "customer_name": customer_name,
            "customer_location": location,
            "actions_taken": [],
            "final_status": "running",
            "customer_wants_booking": False,
        }

        config = {
            "configurable": {
                "thread_id": f"{vehicle_name}_{datetime.now().timestamp()}"
            }
        }
        result = self.graph.invoke(initial_state, config)

        print(f"\n🎉 WORKFLOW COMPLETE: {result['final_status']}")
        for action in result["actions_taken"]:
            print(f"✓ {action}")

        return result


if __name__ == "__main__":
    master = MasterOrchestrator()
    result = master.run_autonomous_workflow("Car B", "Jane Smith")
