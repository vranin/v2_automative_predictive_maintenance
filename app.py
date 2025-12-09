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

st.set_page_config(
    page_title="Predictive Vehicle Guardian",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Epic hero header
st.markdown("""
<style>
    .big-font {font-size: 52px !important; font-weight: bold; color: #00ff9d; text-align: center;}
    .sub-font {font-size: 24px !important; color: #ffffff; text-align: center; margin-bottom: 40px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Predictive Vehicle Guardian</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Prevent breakdowns • Save lives • Cut costs • Protect the planet</p>', unsafe_allow_html=True)

# Live impact counters
c1, c2, c3, c4 = st.columns(4)
c1.metric("Breakdowns Prevented", "2,847", "↑ 47 today")
c2.metric("Money Saved", "₹18.4 Cr", "↑ ₹42L today")
c3.metric("CO₂ Avoided", "1,247 tons", "6,200 trees")
c4.metric("Lives Protected", "9,421", "Zero fatalities")
st.markdown("---")



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

    # Vehicle selector
    vehicle_options = vehicles["vehicle_name"].tolist()
    selected_vehicle_name = st.selectbox("Select a vehicle to view", vehicle_options, index=0)
    vehicle = vehicles[vehicles["vehicle_name"] == selected_vehicle_name].iloc[0].to_dict()

    cea = CustomerEngagementAgent()
    sched = SchedulingAgent()

    st.header(f"Your Vehicle: {vehicle['vehicle_name']} - {vehicle['model']}")
    st.subheader(f"Status: {vehicle['status']}")

    with st.spinner("Analyzing vehicle health..."):
        ai_response = cea.recommend_action(
            vehicle_name=vehicle["vehicle_name"],
            customer_name=vehicle.get("customer_name", "Valued Customer")
        )

    # Chat bubble – FIXED (no backslash in f-string)
    chat_html = ai_response.replace("\n", "<br>")
    st.markdown(
        f"<div class='chatbox'>"
        f"<span class='big-label'>Maintenance Agent</span><br><br>"
        f"{chat_html}"
        f"</div>",
        unsafe_allow_html=True
    )

    # Session state
    defaults = {"confirmed_preferences": None, "display_pref_form": False, "feedback_given": False}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Yes/No buttons
    c1, c2 = st.columns(2)
    with c1:
        yes = st.button("Yes, schedule service", type="primary")
    with c2:
        no = st.button("No, not now")

    if yes:
        st.session_state.display_pref_form = True
    if no:
        st.session_state.confirmed_preferences = None
        st.session_state.display_pref_form = False
        st.markdown('<div class="chatbox">Okay! We will keep monitoring your vehicle.</div>', unsafe_allow_html=True)

    # Scheduling form
 # ——— Scheduling Preference Form (3 Slots) ———
    if st.session_state.display_pref_form and not st.session_state.confirmed_preferences:
        slots = sched.get_available_slots(vehicle['vehicle_name'])
        slots_df = pd.DataFrame(slots)

        st.markdown("### Set Your 3 Preferred Slots")
        st.info("We'll book the first available slot from your preferences, or suggest alternatives if none match.")

        with st.form("schedule_form"):
            # Preference 1
            st.write("**Preference 1 (Top Choice)**")
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                date1 = st.selectbox("Date", sorted(slots_df["date"].unique()), key="d1")
            with col1b:
                times1 = slots_df[slots_df["date"] == date1]["time"].tolist()
                time1 = st.selectbox("Time", times1, key="t1")
            with col1c:
                centers1 = slots_df[(slots_df["date"] == date1) & (slots_df["time"] == time1)]["center"].unique()
                center1 = st.selectbox("Service Center", centers1, key="c1")

            # Preference 2
            st.write("**Preference 2**")
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                date2 = st.selectbox("Date", sorted(slots_df["date"].unique()), key="d2")
            with col2b:
                times2 = slots_df[slots_df["date"] == date2]["time"].tolist()
                time2 = st.selectbox("Time", times2, key="t2")
            with col2c:
                centers2 = slots_df[(slots_df["date"] == date2) & (slots_df["time"] == time2)]["center"].unique()
                center2 = st.selectbox("Service Center", centers2, key="c2")

            # Preference 3
            st.write("**Preference 3**")
            col3a, col3b, col3c = st.columns(3)
            with col3a:
                date3 = st.selectbox("Date", sorted(slots_df["date"].unique()), key="d3")
            with col3b:
                times3 = slots_df[slots_df["date"] == date3]["time"].tolist()
                time3 = st.selectbox("Time", times3, key="t3")
            with col3c:
                centers3 = slots_df[(slots_df["date"] == date3) & (slots_df["time"] == time3)]["center"].unique()
                center3 = st.selectbox("Service Center", centers3, key="c3")

            submitted = st.form_submit_button("Submit Preferences & Book Service", type="primary")
            if submitted:
                # Store all 3 preferences
                preferences = [
                    {"date": date1, "time": time1, "center": center1},
                    {"date": date2, "time": time2, "center": center2},
                    {"date": date3, "time": time3, "center": center3}
                ]
                
                # For now, book the first one (Preference 1) — you can add logic here to check availability across all 3
                confirmed_slot = preferences[0]  # Or implement matching logic
                
                st.session_state.confirmed_preferences = confirmed_slot
                st.session_state.user_preferences = preferences  # Optional: store all for display
                st.rerun()
  # ——— AFTER BOOKING IS CONFIRMED ———
    if st.session_state.confirmed_preferences:
        cp = st.session_state.confirmed_preferences
        all_prefs = st.session_state.get("user_preferences", [cp])  # Fallback to just confirmed if not stored
        
        st.markdown("### Your Booking Confirmation")
        st.success("We've booked your top available preference!")
        
        # Show all 3 with the confirmed one highlighted
        for i, pref in enumerate(all_prefs, 1):
            is_confirmed = (pref == cp)
            status = "✅ BOOKED" if is_confirmed else "Alternative"
            color = "background-color: #d4edda; color: #155724;" if is_confirmed else ""
            
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border: 1px solid #ddd; {color}">
            Preference {i}: <strong>{pref['date']} at {pref['time']} - {pref['center']}</strong> | {status}
            </div>
            """, unsafe_allow_html=True)
    
   

        st.markdown("### Post-Service Feedback")
        vid = vehicle["vehicle_name"].replace(" ", "_").replace("/", "")

        if st.button("I have completed the service – Give Feedback", type="primary", key=f"fb_{vid}"):
            st.session_state[f"showfb_{vid}"] = True
            st.rerun()

        if st.session_state.get(f"showfb_{vid}", False):
            st.info("Your feedback goes straight to the manufacturer")

            rating = st.slider("Service rating", 1, 5, 4, key=f"r_{vid}")
            resolved = st.radio("Issue resolved?", ["Yes", "No", "Partially"], key=f"res_{vid}")
            part = st.selectbox("Part replaced", ["Brake Pad", "Clutch Plate", "ECU", "Battery", "Other"], key=f"p_{vid}")

            if st.button("Send to Manufacturer", type="primary", key=f"send_{vid}"):
                st.balloons()
                st.success("Feedback sent!")

                entry = {
                    "time": pd.Timestamp.now().strftime("%b %d, %H:%M"),
                    "vehicle": vehicle["vehicle_name"],
                    "rating": f"{rating} stars",
                    "resolved": resolved,
                    "part": part,
                }

                if "oem_feedback_buffer" not in st.session_state:
                    st.session_state.oem_feedback_buffer = []
                st.session_state.oem_feedback_buffer.insert(0, entry)

                st.session_state[f"showfb_{vid}"] = False
                st.rerun()

    # SERVICE HISTORY – ONLY ONCE
    st.markdown("### Service History")
    hist = feedback[feedback["vehicle_name"] == vehicle["vehicle_name"]]
    if len(hist) > 0:
        st.dataframe(hist.drop(columns=["feedback_id"], errors="ignore"), use_container_width=True, hide_index=True)
    else:
        st.info("No service records yet")
    
        # ————————— LIVE MAP OF NEAREST SERVICE CENTERS —————————
    st.markdown("### Nearest Service Centers")

    # Simulated real locations (replace with your actual centers if you want)
    service_centers = [
        {"name": "Downtown Service Hub",    "lat": 28.6139, "lon": 77.2090, "wait": "15 min"},
        {"name": "North Delhi Center",   "lat": 28.7041, "lon": 77.1025, "wait": "22 min"},
        {"name": "Gurgaon Premium",         "lat": 28.4595, "lon": 77.0266, "wait": "18 min"},
        {"name": "Noida Sector-62",         "lat": 28.6129, "lon": 77.3619, "wait": "30 min"},
        {"name": "Faridabad Workshop",      "lat": 28.4089, "lon": 77.3178, "wait": "35 min"},
    ]

    # Simulated driver location (changes slightly per vehicle for realism)
    import numpy as np
    driver_lat = 28.5355 + np.random.normal(0, 0.03)   # Delhi NCR area
    driver_lon = 77.3910 + np.random.normal(0, 0.03)

    # Create dataframe for map
    import pandas as pd
    df_centers = pd.DataFrame(service_centers)
    df_driver = pd.DataFrame([{
        "name": f"{vehicle['vehicle_name']} (You are here)",
        "lat": driver_lat,
        "lon": driver_lon,
        "wait": "—"
    }])

    # Combine and color differently
    df_map = pd.concat([df_driver, df_centers], ignore_index=True)
    df_map["color"] = ["#00ff9d"] + ["#ff465a"] * len(df_centers)  # Green = driver, Red = centers
    df_map["size"]  = [80] + [50] * len(df_centers)

    # Display the beautiful map
    st.map(df_map, latitude="lat", longitude="lon", color="color", size="size", zoom=10)

    # Show list below map with travel time
    st.markdown("Available slots today:")
    cols = st.columns(len(service_centers))
    for col, center in zip(cols, service_centers):
        with col:
            st.markdown(f"""
            **{center['name']}**  
            Wait time: **{center['wait']}**  
            """)
            if st.button("Navigate", key=f"nav_{center['name']}"):
                st.success(f"Opening Google Maps to {center['name']}...")
                st.markdown(f"[Click here if not redirected](https://maps.google.com/?q={center['lat']},{center['lon']})")
    


elif tab == "Manufacturer":
    st.header("Manufacturer Dashboard")
    
    st.header("Live Feedback from Drivers (Real-Time Feed)")
    if "oem_feedback_buffer" in st.session_state and st.session_state.oem_feedback_buffer:
        df = pd.DataFrame(st.session_state.oem_feedback_buffer)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(df)} feedback(s) received • Updates instantly")
    else:
        st.info("Waiting for driver feedback… appears here instantly")
    insight = ManufacturingInsightModule()

    st.subheader("Analytics Summary:")
    summary_lines = [ln.strip() for ln in insight.generate_insights().split("\n") if ln.strip()]
    st.markdown(
        "<div style='background:##FF0000; border-radius:10px; box-shadow:0px 1px 10px #bbb; padding:1em 1.5em; margin-bottom:1.2em;'>"
        "<span style='font-size:1.15em; font-weight:600;'>Key Insights:</span><br><ul>" +
        ''.join(f"<li style='margin:0.5em 0; font-size:1.08em'>{line}</li>" for line in summary_lines) +
        "</ul></div>", unsafe_allow_html=True)

    trending = insight.defect_trends()["top_defects"]
    if len(trending) > 0:
        top_defects = ', '.join(trending["defect_type"].astype(str).tolist())
        st.markdown(f"<div style='background:##FF0000; padding:1em; border-radius:10px; font-size:1.1em; margin-bottom:1em;'><b>Top Recurring Issues:</b> {top_defects}</div>", unsafe_allow_html=True)

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
    
        st.divider()
    st.markdown("## OEM View – Recurring Defect Detection & CAPA")

    # Fake post-service feedback data
    feedback_data = pd.DataFrame({
        "vehicle_id": [f"V{i:04d}" for i in range(1, 51)],
        "part_failed": np.random.choice(["Clutch Plate", "Brake Pad", "ECU", "Battery", "Suspension"], 50),
        "batch_no": np.random.choice(["A127", "B884", "C221", "D009"], 50),
        "supplier": np.random.choice(["XYZ Auto", "ABC Industries", "National Parts"], 50),
        "days_since_service": np.random.randint(1, 90, 50),
        "rating": np.random.choice([1, 2, 3, 4, 5], 50, p=[0.1, 0.15, 0.15, 0.3, 0.3])
    })

    # Find worst recurring defect
    worst = feedback_data.groupby(["part_failed", "batch_no", "supplier"]).size().reset_index(name="failures")
    worst = worst.sort_values("failures", ascending=False).iloc[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Worst Recurring Defect", worst["part_failed"])
        st.metric("Batch", worst["batch_no"])
        st.metric("Supplier", worst["supplier"])
        st.metric("Failure Count", worst["failures"])
        st.metric("Failure Rate", f"{(worst.failures/len(feedback_data)*100):.1f}%")

    with col2:
        st.bar_chart(feedback_data["part_failed"].value_counts().head(5))
    from fpdf import FPDF
    from datetime import datetime

    class PDFReport(FPDF):
        def header(self):
            self.set_font('helvetica', 'B', 18)          # safe built-in font
            self.set_text_color(200, 0, 0)
            self.cell(0, 15, 'OEM Quality Alert - Root Cause Report', align='C', ln=1)
            self.ln(8)

        def footer(self):
            self.set_y(-15)
            self.set_font('helvetica', 'I', 9)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()} - {datetime.now().strftime("%d %b %Y")}', align='C')

    # Generate PDF — using only safe characters
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font('helvetica', '', 13)
    pdf.cell(0, 10, "Critical Finding:", ln=1)

    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(0, 12, f"{worst.part_failed} - Batch {worst.batch_no} from {worst.supplier}", ln=1)

    pdf.set_text_color(0, 0, 0)

    pdf.set_font('helvetica', '', 13)
    pdf.multi_cell(0, 10,
        f"has caused {worst.failures} failures "
        f"({worst.failures/len(feedback_data)*100:.1f}%) in the last 90 days.\n\n"
        "Recommended Corrective & Preventive Actions (CAPA):\n"
        f"- Immediately halt use of Batch {worst.batch_no}\n"
        "- Switch to alternate supplier ABC Industries (0 failures recorded)\n"
        "- Initiate 8D root cause analysis within 48 hours\n"
        "- Notify all affected vehicles for recall inspection"
    )

    pdf_file = "OEM_RCA_Report.pdf"
    pdf.output(pdf_file)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download CAPA Report PDF",
            data=f,
            file_name=f"RCA_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.success("CAPA report generated successfully!")


elif tab == "UEBA Log":
    logs = load_logs()
    st.header("UEBA & Security Audit Log")

    st.subheader("Audit Timeline (Latest First)")
    audit = get_audit_timeline(logs).reset_index(drop=True)
    audit.index = range(1, len(audit) + 1)
    audit.index.name = "No."
    st.dataframe(audit, use_container_width=True)

    st.subheader("Filter by Vehicle")
    selected_vehicle = st.selectbox("Select Vehicle", options=sorted(logs["vehicle_name"].unique()), key="ueba_vehicle")
    st.dataframe(filter_logs(logs, vehicle_name=selected_vehicle), use_container_width=True)

    st.subheader("Security Alerts & Anomalies")
    anomalies = get_anomalies(logs)
    def highlight_anomaly(row):
        return ['background-color: #ffcccc; font-weight: bold'] * len(row) if any(x in str(row['status']) for x in ['Blocked', 'ALERT', 'UNAUTHORIZED']) else [''] * len(row)
    st.dataframe(anomalies.style.apply(highlight_anomaly, axis=1), use_container_width=True)

    st.subheader("Anomaly Statistics")
    st.write(anomaly_summary(logs))

    st.subheader("Behavioral Risk Score (Last 7 Days)")
    risk_df = compute_behavioral_risk(logs)
    risk_df = risk_df.reset_index(drop=True)
    risk_df.index = range(1, len(risk_df) + 1)
    risk_df.index.name = "Rank"
    st.dataframe(risk_df, use_container_width=True)

    if st.button("Simulate Unauthorized Access Attempt", type="secondary"):
        append_anomaly(selected_vehicle, "Diagnosis Agent", "Unauthorized API call detected", "BLOCKED by UEBA")
        st.warning("Simulated security incident injected!")
        st.rerun()



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
    
