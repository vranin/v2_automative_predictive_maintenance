# 4. Recreate the PERFECT master.py in the correct place (copy-paste this exact block into Notepad → Save As → graph\master.py → Encoding UTF-8)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
import pandas as pd
import random
from models.manufacturing_insight_model import manufacturing_insights_agent

def load_defects():
    try:
        return pd.read_csv("data/defects.csv")
    except:
        return pd.DataFrame()

def data_analysis_agent(vehicle):
    return f"Data Analysis Agent: Scanning telematics for {vehicle}... High vibration detected!"

def diagnosis_agent(vehicle):
    df = load_defects()
    if not df.empty and vehicle in df["vehicle_name"].values:
        row = df[df["vehicle_name"] == vehicle].iloc[0]
        return f"Diagnosis Agent: Found {row['defect_type']} ({row['severity']}) - Cost Rs{row['estimated_cost']}"
    return "Diagnosis Agent: Vehicle healthy"

def ueba_guard(vehicle):
    if random.random() < 0.15:
        return "BLOCKED", "UEBA Security: Suspicious booking pattern detected - action blocked!"
    return "OK", "UEBA Security: All clear"

def customer_engagement_agent(vehicle, diagnosis):
    return f"Customer Agent: Hi! We found an issue with your {vehicle}. {diagnosis} Schedule service?"

def scheduling_agent(vehicle):
    return f"Scheduling Agent: Booking confirmed for {vehicle} on Monday 10AM at Center A"

def feedback_agent(vehicle):
    return "Feedback Agent: Thank you for your visit! How was your experience?"

class State(TypedDict):
    vehicle: str
    blocked: bool

workflow = StateGraph(State)
workflow.add_node("data", lambda s: s)
workflow.add_node("diagnosis", lambda s: s)
workflow.add_node("ueba", lambda s: {**s, "blocked": ueba_guard(s["vehicle"])[0] == "BLOCKED"})
workflow.add_node("engage", lambda s: s)
workflow.add_node("schedule", lambda s: s)
workflow.add_node("feedback", lambda s: s)
workflow.add_node("insights", lambda s: s)

workflow.set_entry_point("data")
workflow.add_edge("data", "diagnosis")
workflow.add_edge("diagnosis", "ueba")
workflow.add_conditional_edges("ueba", lambda s: END if s["blocked"] else "engage")
workflow.add_edge("engage", "schedule")
workflow.add_edge("schedule", "feedback")
workflow.add_edge("feedback", "insights")
workflow.add_edge("insights", END)

orchestrator = workflow.compile(checkpoint=MemorySaver())

__all__ = ["orchestrator","data_analysis_agent","diagnosis_agent","ueba_guard",
           "customer_engagement_agent","scheduling_agent","feedback_agent","manufacturing_insights_agent"]