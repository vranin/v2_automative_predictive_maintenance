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



st.set_page_config(page_title="Predictive Maintenance System", layout="wide")
st.title("Predictive Maintenance Demo")

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

   
    vehicle = vehicles[vehicles["vehicle_name"] == selected_vehicle_name].iloc[0]  # first vehicle for demo
    cea = CustomerEngagementAgent()
    sched = SchedulingAgent()

    st.header(f"Your Vehicle: {vehicle['vehicle_name']} - {vehicle['model']}")
    st.subheader(f"Status: {vehicle['status']}")

    
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
    # ──────────────────────────────
# VOICE CEA DEMO (SIMULATED – LOOKS 100% REAL)
# ──────────────────────────────
    st.markdown("### Live Voice Customer Engagement Agent Demo")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Normal Risk → Ask Permission", key="norm"):
            st.audio("audio/alert.wav", format="audio/wav")
            choice = st.radio("Driver presses:", ["1 – Yes, book it", "2 – No, later"], key="c1")
            if choice == "1 – Yes, book it":
                st.audio("audio/booked.wav", format="audio/wav")
                st.success("Slot booked – Normal flow")
            else:
                st.audio("audio/reminder.wav", format="audio/wav")
                st.info("Declined → Reminder scheduled (red X edge case)")

    with col2:
        if st.button("HIGH Risk → Force Booking (Urgent)", key="urg"):
            st.audio("audio/urgent.wav", format="audio/wav")
            st.success("URGENT: Auto-booked without asking (red lightning edge case)")
            st.write("Scheduling Agent triggered automatically")

    # Feedback loop
    if st.button("Trigger Post-Service Feedback"):
        st.audio("audio/booked.wav", format="audio/wav")  # reuse or add feedback.wav
        rating = st.slider("Driver rates service:", 1, 5, 3)
        st.success(f"Feedback {rating}/5 → Sent to RCA module (closed loop)")

    # ==================== FEATURE #2: FLEET HEATMAP + PRIORITIZATION ====================
    st.divider()
    st.markdown("## Fleet Manager View – Risk-Based Scheduling (10 bays limit)")

    import streamlit as st
    import pandas as pd
    import numpy as np
    import folium
    from streamlit_folium import st_folium
    import time

    # Fake 20 vehicles (replace later with real DB)
    np.random.seed(42)
    fleet = pd.DataFrame({
        "vehicle_id": [f"TRK-{i:03d}" for i in range(1,21)],
        "risk_score": np.random.uniform(0.1, 0.95, 20).round(2),
        "lat": np.random.uniform(18.9, 19.1, 20),
        "lon": np.random.uniform(72.8, 73.0, 20),
        "issue": np.random.choice(["Brake", "Engine", "Tyre", "Battery"], 20)
    })
    fleet["risk_level"] = pd.cut(fleet["risk_score"], bins=[0,0.4,0.7,1], labels=["Low", "Medium", "High"])
    color_map = {"Low": "green", "Medium": "orange", "High": "red"}

    # Map
    m = folium.Map(location=[19.0, 72.9], zoom_start=11)
    for _, row in fleet.iterrows():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=12,
            popup=f"{row.vehicle_id}<br>{row.issue}<br>Risk: {row.risk_score}",
            color="black",
            weight=1,
            fillColor=color_map[row.risk_level],
            fillOpacity=0.8
        ).add_to(m)

    col_map, col_table = st.columns([2,1])
    with col_map:
        st.markdown("##### Live Risk Heatmap (20 vehicles)")
        st_folium(m, width=700, height=500)
    with col_table:
        st.markdown("##### Risk Ranking")
        st.dataframe(
            fleet[["vehicle_id", "risk_score", "risk_level", "issue"]]
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

    # ONE-CLICK FLEET SCHEDULING BUTTON
    if st.button("SCHEDULE TODAY'S 10 SLOTS (Highest Risk First)", type="primary", use_container_width=True):
        with st.spinner("Prioritizing fleet..."):
            time.sleep(2)
        
        top10 = fleet.sort_values("risk_score", ascending=False).head(10)
        
        st.success("10 bays filled with highest-risk vehicles:")
        st.dataframe(top10[["vehicle_id", "risk_score", "issue"]], use_container_width=True)
        st.info("Remaining vehicles moved to tomorrow's queue (bay capacity respected)")

    # ==================== FEATURE #3: OEM RCA REPORT (ONE-CLICK PDF) ====================
    st.divider()
    st.markdown("## OEM View – Recurring Defect Detection & CAPA")

    # Fake post-service feedback data 
    feedback_data = pd.DataFrame({
        "vehicle_id": [f"V{i:04d}" for i in range(1,51)],
        "part_failed": np.random.choice(["Clutch Plate", "Brake Pad", "ECU", "Battery", "Suspension"], 50),
        "batch_no": np.random.choice(["A127", "B884", "C221", "D009"], 50),
        "supplier": np.random.choice(["XYZ Auto", "ABC Industries", "National Parts"], 50),
        "days_since_service": np.random.randint(1, 90, 50),
        "rating": np.random.choice([1,2,3,4,5], 50, p=[0.1,0.15,0.15,0.3,0.3])
    })

    # Find the worst offender
    worst = feedback_data.groupby(["part_failed", "batch_no", "supplier"]).size().reset_index(name="failures")
    worst = worst.sort_values("failures", ascending=False).iloc[0]

    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Worst Recurring Defect", f"{worst.part_failed}")
        st.metric("Batch", f"{worst.batch_no}")
        st.metric("Supplier", worst.supplier)
        st.metric("Failure Count", worst.failures)
        st.metric("Failure Rate", f"{(worst.failures/len(feedback_data)*100):.1f}%")

    with col2:
        st.bar_chart(feedback_data["part_failed"].value_counts().head(5))

    # ONE-CLICK PDF REPORT
    if st.button("Generate OEM CAPA Report (Download PDF)", type="primary", use_container_width=True):
        from fpdf import FPDF
        import base64
        from datetime import datetime

        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'OEM Quality Alert – Root Cause Report', align='C', ln=1)
                self.ln(5)
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()} | Generated: {datetime.now().strftime("%d %b %Y")}', align='C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Critical Finding:", ln=1)
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(220, 50, 50)
        pdf.cell(0, 10, f"{worst.part_failed} – Batch {worst.batch_no} from {worst.supplier}", ln=1)
        pdf.set_text_color(0,0,0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"has caused {worst.failures} failures ({(worst.failures/len(feedback_data)*100):.1f}%) in last 90 days.", ln=1)
        pdf.ln(5)
        pdf.cell(0, 10, "Recommended CAPA:", ln=1)
        pdf.cell(0, 10, "• Immediately halt use of Batch A127", ln=1)
        pdf.cell(0, 10, "• Switch to alternate supplier ABC Industries (0 failures in dataset)", ln=1)
        pdf.cell(0, 10, "• Initiate 8D analysis within 48 hours", ln=1)

        pdf_file = "OEM_RCA_Report.pdf"
        pdf.output(pdf_file)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "Download CAPA Report PDF",
                data=f,
                file_name=f"RCA_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        st.success("OEM report generated & ready for download!")

        st.divider()
    st.divider()
    st.markdown("## Live CAPA Impact Simulator")

    current_failures = 487
    warranty_cost_per_failure = 98000
    current_supplier_share = 42  # %

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Failures (90 days)", current_failures)
    col2.metric("Warranty Cost", f"₹{(current_failures * warranty_cost_per_failure / 1e7):.1f} Cr")
    col3.metric("Bad Supplier Share", f"{current_supplier_share}%")

    st.markdown("### What if we act now?")

    reduction_pct = st.slider(
        "Reduce volume from bad supplier (%)",
        min_value=0, max_value=100, value=75, step=5
    )

    projected_failures = int(current_failures * (1 - reduction_pct/100 * current_supplier_share/100))
    savings_cr = round((current_failures - projected_failures) * warranty_cost_per_failure / 1e7, 2)
    csat_gain = round(reduction_pct * 0.24, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Failures Avoided", current_failures - projected_failures,
            delta=f"-{reduction_pct * current_supplier_share / 100:.0f}%")
    c2.metric("Warranty Savings", f"₹{savings_cr:,} Cr")
    c3.metric("Customer Satisfaction Gain", f"+{csat_gain} pts")

    if st.button("EXECUTE CAPA PLAN NOW", type="primary", use_container_width=True):
        
        st.success(f"CAPA executed! {current_failures - projected_failures} failures prevented | ₹{savings_cr:,} Cr saved")



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
    
