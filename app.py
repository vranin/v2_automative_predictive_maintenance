import streamlit as st
import pandas as pd
from agents.customer_engagement_agent import CustomerEngagementAgent
from agents.scheduling_agent import SchedulingAgent
from utils.agent_logic import load_vehicles, load_defects, load_feedback, load_logs, log_event
from models.manufacturing_insight_model import ManufacturingInsightModule
from utils.security_tools import (
    filter_logs, get_anomalies, append_anomaly,
    anomaly_summary, get_audit_timeline, compute_behavioral_risk
)

# ←←← REMOVE THIS LINE — there is no graph_builder in your repo →→→
# from agents.graph_builder import app as graph   # DELETE THIS LINE

st.set_page_config(page_title="Predictive Maintenance System", layout="wide")
st.title("Predictive Maintenance Demo")

# --- Custom CSS ---
st.markdown('''
    <style>
    .chatbox {background: #f2f3fa; padding:1.3em; border-radius:1em; box-shadow:0px 1px 8px #aaa;
        font-size:1.15em !important; margin-bottom:1em; color:#303050;}
    .big-label {font-size:1.22em !important; font-weight:700; margin-bottom:1em; margin-top:0.5em;}
    .confirmbox {background:#e6ffe6;padding:1em;border-radius:8px;font-size:1.08em;color:#225522;margin-top:1.1em;}
    .pref-label {font-size:1.09em;margin-top:0.5em;}
    button[kind="primary"] {
        background:#0066cc !important; color:#fff !important; font-weight:700; border-radius:10px;
        padding:0.4em 1.2em; border:none; box-shadow:0px 2px 8px #aaa;}
    button[kind="primary"]:hover {
        box-shadow:0px 4px 16px #7799dd; background:#004c99 !important;}
    </style>
''', unsafe_allow_html=True)

# --- Sidebar ---
tab = st.sidebar.selectbox("Dashboard", ["User", "Manufacturer", "UEBA Log"])

# ========================================
# USER DASHBOARD
# ========================================
if tab == "User":
    vehicles = load_vehicles()
    feedback = load_feedback()
    # Let user pick any vehicle
    vehicle_options = vehicles["vehicle_name"].tolist()
    selected_vehicle_name = st.selectbox("Select a vehicle to view", vehicle_options, index=0)

    # Get the full row for the selected vehicle
    vehicle = vehicles[vehicles["vehicle_name"] == selected_vehicle_name].iloc[0]  # first vehicle for demo
    cea = CustomerEngagementAgent()
    sched = SchedulingAgent()

    st.header(f"Your Vehicle: {vehicle['vehicle_name']} - {vehicle['model']}")
    st.subheader(f"Status: {vehicle['status']}")

    # ——— SMART AI RECOMMENDATION (now works perfectly) ———
       # ——— SMART AI RECOMMENDATION (NOW WORKS 100%) ———
    with st.spinner("Maintenance Agent is thinking..."):
        defect = cea.get_latest_defect(vehicle["vehicle_name"])
        ai_response = cea.recommend_action(
            vehicle_name=vehicle["vehicle_name"],
            customer_name=vehicle.get("customer_name", "Valued Customer")
        )

    # Beautiful chat bubble – NO MORE SYNTAX ERROR
    chat_message = ai_response.replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="chatbox">
        <span class="big-label">Maintenance Agent</span><br><br>
        {chat_message}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ——— Session state ———
    if "confirmed_preferences" not in st.session_state:
        st.session_state.confirmed_preferences = None
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "display_pref_form" not in st.session_state:
        st.session_state.display_pref_form = False

    col1, col2 = st.columns(2)
    with col1:
        yes = st.button("Yes, schedule service", key="yes_btn", type="primary")
    with col2:
        no = st.button("No, not now", key="no_btn", type="secondary")

    if yes:
        st.session_state.display_pref_form = True
    if no:
        st.session_state.confirmed_preferences = None
        st.session_state.feedback_given = False
        st.session_state.display_pref_form = False
        st.markdown('<div class="chatbox">No problem! We’ll keep monitoring your vehicle and let you know if anything urgent comes up.</div>', unsafe_allow_html=True)

    # ——— Preference Form ———
    if st.session_state.display_pref_form and not st.session_state.confirmed_preferences:
        slots = sched.get_available_slots(vehicle['vehicle_name'])
        slots_df = pd.DataFrame(slots)

        st.markdown('<span class="pref-label"><b>Set Your Three Preferences:</b></span>', unsafe_allow_html=True)
        with st.form("schedule_form"):
            st.write("Preference 1:")
            c1, c2, c3 = st.columns(3)
            with c1: date1 = st.selectbox("Date 1", slots_df["date"], key="d1")
            with c2: time1 = st.selectbox("Time 1", slots_df[slots_df["date"] == date1]["time"], key="t1")
            with c3: center1 = st.selectbox("Place 1", slots_df[slots_df["date"] == date1]["center"].unique(), key="c1")

            st.write("Preference 2:")
            c4, c5, c6 = st.columns(3)
            with c4: date2 = st.selectbox("Date 2", slots_df["date"], key="d2")
            with c5: time2 = st.selectbox("Time 2", slots_df[slots_df["date"] == date2]["time"], key="t2")
            with c6: center2 = st.selectbox("Place 2", slots_df[slots_df["date"] == date2]["center"].unique(), key="c2")

            st.write("Preference 3:")
            c7, c8, c9 = st.columns(3)
            with c7: date3 = st.selectbox("Date 3", slots_df["date"], key="d3")
            with c8: time3 = st.selectbox("Time 3", slots_df[slots_df["date"] == date3]["time"], key="t3")
            with c9: center3 = st.selectbox("Place 3", slots_df[slots_df["date"] == date3]["center"].unique(), key="c3")

            confirm = st.form_submit_button("Confirm Preferences", type="primary")
            if confirm:
                st.session_state.confirmed_preferences = {
                    "date": date1, "time": time1, "center": center1,
                }
                st.session_state.display_pref_form = False
                st.rerun()

    # ——— Confirmation Message ———
    if st.session_state.confirmed_preferences:
        cp = st.session_state.confirmed_preferences
        st.markdown(f'''<div class="confirmbox">
        Your service is scheduled at:<br><br>
        <b>{cp['date']}</b> | <b>{cp['time']}</b> | <b>{cp['center']}</b>
        </div>''', unsafe_allow_html=True)

        log_event(
            vehicle["vehicle_name"], "Scheduling Agent", "Appointment confirmed", "scheduling",
            "Confirmed", f"Slot: {cp['date']} {cp['time']} @ {cp['center']}"
        )

        if not st.session_state.feedback_given:
            st.markdown('<div class="chatbox">Your service is booked! How was the scheduling experience?</div>', unsafe_allow_html=True)
            user_feedback = st.text_area("Your feedback (optional):")
            if st.button("Submit Feedback", type="primary"):
                st.success("Thank you for your feedback!")
                st.session_state.feedback_given = True

    # ——— Service History
    st.markdown('<span class="big-label">Service History & Feedback:</span>', unsafe_allow_html=True)
    user_fb = feedback[feedback["vehicle_name"] == vehicle["vehicle_name"]]
    st.table(user_fb)


# ========================================
# MANUFACTURER DASHBOARD (unchanged)
# ========================================
elif tab == "Manufacturer":
    st.header("Manufacturer Dashboard")
    insight = ManufacturingInsightModule()

    st.subheader("Analytics Summary:")
    summary_lines = [ln.strip() for ln in insight.generate_insights().split("\n") if ln.strip()]
    st.markdown(
        "<div style='background:#f5f5fa; border-radius:10px; box-shadow:0px 1px 10px #bbb; padding:1em 1.5em; margin-bottom:1.2em;'>"
        "<span style='font-size:1.15em; font-weight:600;'>Key Insights:</span><br><ul>" +
        ''.join(f"<li style='margin:0.5em 0; font-size:1.08em'>{line}</li>" for line in summary_lines) +
        "</ul></div>", unsafe_allow_html=True)

    trending = insight.defect_trends()["top_defects"]
    if len(trending) > 0:
        top_defects = ', '.join(trending["defect_type"].astype(str).tolist())
        st.markdown(f"<div style='background:#fff9c4; padding:1em; border-radius:10px; font-size:1.1em; margin-bottom:1em;'><b>Top Recurring Issues:</b> {top_defects}</div>", unsafe_allow_html=True)

    st.subheader("Feedback Insights:")
    fb_df = insight.aggregate_feedback_insights()["average_rating_by_vehicle"].reset_index(drop=True)
    fb_df.index = range(1, len(fb_df) + 1)
    fb_df.index.name = "No."
    st.dataframe(fb_df)

    st.subheader("Defect Trends:")
    defect_trend_df = insight.defect_trends()["defect_trends"].reset_index(drop=True)
    defect_trend_df.index = range(1, len(defect_trend_df) + 1)
    defect_trend_df.index.name = "No."
    st.dataframe(defect_trend_df)

    st.subheader("RCA/CAPA Suggestions:")
    rca = insight.rca_capa_summary()
    if isinstance(rca, str):
        st.markdown(f"<em>{rca}</em>")
    else:
        rca.index = range(1, len(rca) + 1)
        rca.index.name = "No."
        st.dataframe(rca)


# ========================================
# UEBA LOG DASHBOARD (unchanged)
# ========================================
elif tab == "UEBA Log":
    logs = load_logs()
    st.header("UEBA & Audit Log")

    st.subheader("Audit Timeline (latest first)")
    audit = get_audit_timeline(logs).reset_index(drop=True)
    audit.index = range(1, len(audit)+1)
    audit.index.name = "No."
    st.dataframe(audit)

    st.subheader("Filter by Vehicle")
    selected_vehicle = st.selectbox("Vehicle", logs["vehicle_name"].unique())
    st.dataframe(filter_logs(logs, vehicle_name=selected_vehicle))

    st.subheader("Security Alerts")
    anomalies = get_anomalies(logs)
    def highlight(row):
        return ['background-color: #ffdddd; font-weight:bold'] * len(row) if 'Blocked' in str(row['status']) or 'ALERT' in str(row['status']) else [''] * len(row)
    st.dataframe(anomalies.style.apply(highlight, axis=1))

    st.subheader("Anomaly Statistics")
    st.write(anomaly_summary(logs))

    st.subheader("Behavioral Risk (last 7 days)")
    risk_df = compute_behavioral_risk(logs).reset_index(drop=True)
    risk_df.index = range(1, len(risk_df)+1)
    risk_df.index.name = "No."
    st.dataframe(risk_df)

    if st.button("Simulate Unauthorized Action"):
        append_anomaly(selected_vehicle, "Diagnosis Agent", "Unauthorized API call", "Blocked by UEBA")
        st.warning("Simulated anomaly injected")
        st.rerun()
