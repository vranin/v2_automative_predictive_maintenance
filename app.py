import os
import numpy as np
import pandas as pd
import streamlit as st

from agents.customer_engagement_agent import CustomerEngagementAgent
from agents.scheduling_agent import SchedulingAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.feedback_agent import FeedbackAgent
from graph.master import MasterOrchestrator
from models.manufacturing_insight_model import ManufacturingInsightModule
from utils.agent_logic import load_vehicles, load_defects, load_feedback, load_logs, log_event
from utils.security_tools import (
    filter_logs,
    get_anomalies,
    append_anomaly,
    anomaly_summary,
    get_audit_timeline,
    compute_behavioral_risk,
    agent_ueba,
)

st.set_page_config(
    page_title="Predictive Vehicle Guardian",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .big-font {font-size: 52px !important; font-weight: bold; color: #00ff9d; text-align: center;}
    .sub-font {font-size: 24px !important; color: #ffffff; text-align: center; margin-bottom: 40px;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="big-font">Predictive Vehicle Guardian</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-font">Prevent breakdowns • Save lives • Cut costs • Protect the planet</p>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Breakdowns Prevented", "2,847", "↑ 47 today")
c2.metric("Money Saved", "₹18.4 Cr", "↑ ₹42L today")
c3.metric("CO₂ Avoided", "1,247 tons", "6,200 trees")
c4.metric("Lives Protected", "9,421", "Zero fatalities")
st.markdown("---")

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

tab = st.sidebar.selectbox("Dashboard", ["User", "Manufacturer", "UEBA Log"])

if tab == "User":
    vehicles = load_vehicles()
    feedback_df = load_feedback()

    cea = CustomerEngagementAgent()
    sched = SchedulingAgent()
    diag = DiagnosisAgent()
    master = MasterOrchestrator()

    vehicle_options = vehicles["vehicle_name"].tolist()
    selected_vehicle_name = st.selectbox(
        "Select a vehicle to view", vehicle_options, index=0
    )
    vehicle = (
        vehicles[vehicles["vehicle_name"] == selected_vehicle_name]
        .iloc[0]
        .to_dict()
    )

    st.header(f"Your Vehicle: {vehicle['vehicle_name']} - {vehicle.get('model', '')}")
    st.subheader(f"Status: {vehicle.get('status', 'Unknown')}")

    with st.spinner("Analyzing vehicle health..."):
        base_diag = diag.continuous_monitor(vehicle["vehicle_id"])
        ai_response = cea.recommend_action(
            vehicle_name=vehicle["vehicle_name"],
            customer_name=vehicle.get("customer_name", "Valued Customer"),
        )

    if "master_result" in st.session_state and st.session_state.master_result:
        diagnosis = st.session_state.master_result.get("diagnosis", base_diag)
    else:
        diagnosis = base_diag

    risk_col, issue_col, urg_col = st.columns(3)
    risk_col.metric("Risk Level", diagnosis.get("risk_level", "unknown"))
    issue_col.metric("Predicted Failure", diagnosis.get("predicted_failure", "N/A"))
    urg_col.metric("Urgency", diagnosis.get("urgency", "N/A"))

    chat_html = ai_response.replace("\n", "<br>")
    st.markdown(
        f"<div class='chatbox'>"
        f"<span class='big-label'>Maintenance Agent</span><br><br>"
        f"{chat_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if "master_result" not in st.session_state:
        st.session_state.master_result = None
    if "user_flow_stage" not in st.session_state:
        st.session_state.user_flow_stage = "diagnosis"
    if "user_booking" not in st.session_state:
        st.session_state.user_booking = None

    if st.session_state.user_flow_stage == "diagnosis":
        c1, c2 = st.columns(2)
        with c1:
            proceed = st.button(
                "Yes, continue to scheduling", type="primary", key="go_sched"
            )
        with c2:
            no = st.button("No, not now", key="no_user")

        if proceed:
            st.session_state.user_flow_stage = "scheduling"
            st.rerun()
        if no:
            st.markdown(
                '<div class="chatbox">Okay! We will keep monitoring your vehicle.</div>',
                unsafe_allow_html=True,
            )

    if st.session_state.user_flow_stage == "scheduling":
        st.markdown("### Schedule your service")

        centers_dict = sched.centers
        center_names = list(centers_dict.keys())
        selected_center = st.selectbox(
            "Preferred Service Center", center_names, key="user_pref_center"
        )

        # Dropdowns for up to 3 preferred slots
        # Expecting sched.get_available_slots(vehicle_name, center_name or coords)
        slots = sched.get_available_slots(
            vehicle["vehicle_name"], centers_dict[selected_center]
        )
        slots_df = pd.DataFrame(slots)

        if slots_df.empty:
            st.warning("No slots available for this center right now.")
        else:
            st.markdown("Pick up to 3 preferred time slots.")

            def format_date(d):
                # if already a string like "2025-12-15", just return it
                if isinstance(d, str):
                    return d.split(" ")[0]
                # if it's a Timestamp/datetime, format it
                return d.strftime("%Y-%m-%d")

            slots_df["label"] = slots_df.apply(
                lambda r: f"{format_date(r['date'])} {r['time']}", axis=1
            )

            pref1 = st.selectbox(
                "1st preference (Top choice)",
                options=slots_df["label"].tolist(),
                key="pref1",
            )
            pref2 = st.selectbox(
                "2nd preference (optional)",
                options=["None"] + slots_df["label"].tolist(),
                key="pref2",
            )
            pref3 = st.selectbox(
                "3rd preference (optional)",
                options=["None"] + slots_df["label"].tolist(),
                key="pref3",
            )

            if st.button("Confirm booking", type="primary", key="confirm_booking"):
                prefs = [p for p in [pref1, pref2, pref3] if p != "None"]
                if not prefs:
                    st.warning("Please select at least your 1st preferred time slot.")
                else:
                    with st.spinner("Booking your service..."):
                        booking_result = sched.book_with_preferences(
                            vehicle_name=vehicle["vehicle_name"],
                            center_name=selected_center,
                            preferences=prefs,
                            customer_name=vehicle.get(
                                "customer_name", "Valued Customer"
                            ),
                        )
                    st.session_state.user_booking = booking_result
                    st.session_state.user_flow_stage = "booked"

                    if (
                        isinstance(booking_result, dict)
                        and booking_result.get("status")
                        in ["reserved", "confirmed"]
                    ):
                        st.success(
                            f"Booking confirmed: {booking_result['date']} {booking_result['time']} "
                            f"at {booking_result['center']} (status: {booking_result['status']})"
                        )
                    else:
                        st.warning(f"Could not confirm booking: {booking_result}")

    if st.session_state.user_flow_stage == "booked" and st.session_state.user_booking:
        br = st.session_state.user_booking
        st.markdown("### Your booking")
        st.success(
            f"{br['date']} {br['time']} at {br['center']} for {vehicle['vehicle_name']} "
            f"(status: {br['status']})"
        )

    latest = (
        feedback_df[feedback_df["vehicle_name"] == vehicle["vehicle_name"]]
        .sort_values("service_date", ascending=False)
        .head(1)
    )
    if not latest.empty:
        row = latest.iloc[0]
        st.markdown("### Latest service feedback")
        st.info(
            f"Your last service for {vehicle['vehicle_name']} on {row['service_date']} "
            f"was recorded with rating {row['user_rating']} and "
            f"resolved: {row['issue_resolved']}."
        )

    st.markdown("### Service History")
    hist = feedback_df[feedback_df["vehicle_name"] == vehicle["vehicle_name"]]

    if len(hist) > 0:
        st.dataframe(
            hist.drop(columns=["feedback_id"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No service records yet")

    st.markdown("### Nearest Service Centers")
    centers_dict = sched.centers
    driver_lat, driver_lon = 19.0467, 72.9064
    centers_df = pd.DataFrame(
        [
            {"name": name, "lat": coords[0], "lon": coords[1], "wait": "15–40 min"}
            for name, coords in centers_dict.items()
        ]
    )
    driver_df = pd.DataFrame(
        [
            {
                "name": f"{vehicle['vehicle_name']} (You are here)",
                "lat": driver_lat,
                "lon": driver_lon,
                "wait": "—",
            }
        ]
    )
    df_map = pd.concat([driver_df, centers_df], ignore_index=True)
    df_map["color"] = ["#00ff9d"] + ["#ff465a"] * len(centers_df)
    df_map["size"] = [80] + [50] * len(centers_df)

    st.map(df_map, latitude="lat", longitude="lon", color="color", size="size", zoom=11)

elif tab == "Manufacturer":
    vehicles = load_vehicles()
    feedback_df = load_feedback()
    st.header("Manufacturer Dashboard")

    if "oem_feedback_buffer" in st.session_state and st.session_state.oem_feedback_buffer:
        df = pd.DataFrame(st.session_state.oem_feedback_buffer)
        st.subheader("Live Feedback from Drivers (Real-Time Feed)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(df)} feedback(s) received • Updates instantly")
    else:
        st.subheader("Live Feedback from Drivers (Real-Time Feed)")
        st.info("Waiting for driver feedback… appears here instantly")

    st.markdown("### Upload Post-Service Report and Trigger Feedback")

    selected_vehicle_for_report = st.selectbox(
        "Vehicle for which service has been completed",
        vehicles["vehicle_name"].tolist(),
        key="oem_report_vehicle",
    )

    uploaded_report = st.file_uploader(
        "Upload service report (PDF, TXT, or CSV)",
        type=["pdf", "txt", "csv"],
        key="service_report_uploader",
    )

    customer_name_for_report = st.text_input(
        "Customer name (for feedback record)",
        value="Customer",
        key="oem_customer_name",
    )

    if st.button("Save report and activate Feedback Agent", key="oem_save_report"):
        if uploaded_report is None:
            st.warning("Please upload a service report file first.")
        else:
            report_bytes = uploaded_report.getvalue()
            report_name = uploaded_report.name

            year_folder = pd.Timestamp.now().strftime("%Y")
            month_folder = pd.Timestamp.now().strftime("%m")
            out_dir = os.path.join("data", "service_reports", year_folder, month_folder)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, report_name)
            with open(out_path, "wb") as f:
                f.write(report_bytes)

            feedback_agent = FeedbackAgent()
            rating_default = 4.0
            resolved_default = "Yes"
            comments_default = f"Service completed; report stored at {out_path}"

            result = feedback_agent.process_feedback(
                vehicle_name=selected_vehicle_for_report,
                customer_name=customer_name_for_report,
                rating=rating_default,
                issue_resolved=resolved_default,
                comments=comments_default,
                service_center="OEM Workshop",
            )

            st.session_state.oem_feedback_buffer = st.session_state.get(
                "oem_feedback_buffer", []
            )
            st.session_state.oem_feedback_buffer.insert(
                0,
                {
                    "vehicle_name": selected_vehicle_for_report,
                    "rating": rating_default,
                    "resolved": resolved_default,
                    "comments": comments_default,
                    "saved": result.get("saved", False),
                    "feedback_id": result.get("feedback_id", ""),
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "report_path": out_path,
                },
            )

            st.success(
                f"Report saved and FeedbackAgent activated for {selected_vehicle_for_report} "
                f"(feedback id: {result.get('feedback_id', 'N/A')})."
            )

    insight = ManufacturingInsightModule()

    st.subheader("Analytics Summary:")
    summary_lines = [
        ln.strip() for ln in insight.generate_insights().split("\n") if ln.strip()
    ]
    st.markdown(
        "<div style='background:#f8f9ff; border-radius:10px; box-shadow:0px 1px 10px #bbb; padding:1em 1.5em; margin-bottom:1.2em;'>"
        "<span style='font-size:1.15em; font-weight:600;'>Key Insights:</span><br><ul>"
        + "".join(
            f"<li style='margin:0.5em 0; font-size:1.08em'>{line}</li>"
            for line in summary_lines
        )
        + "</ul></div>",
        unsafe_allow_html=True,
    )

    trending = insight.defect_trends()["top_defects"]
    if len(trending) > 0:
        top_defects = ", ".join(trending["defect_type"].astype(str).tolist())
        st.markdown(
            f"<div style='background:#fff3cd; padding:1em; border-radius:10px; font-size:1.1em; margin-bottom:1em;'><b>Top Recurring Issues:</b> {top_defects}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Feedback Insights:")
    fb_df = insight.aggregate_feedback_insights()[
        "average_rating_by_vehicle"
    ].reset_index(drop=True)
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
        st.markdown(f"<em>{rca}</em>", unsafe_allow_html=True)
    else:
        rca.index = range(1, len(rca) + 1)
        rca.index.name = "No."
        st.dataframe(rca)

    st.markdown("### Autonomous Agentic Flow Demo (Per Vehicle)")
    demo_vehicle = st.selectbox(
        "Vehicle for autonomous run",
        vehicles["vehicle_name"].tolist(),
        key="oem_master_vehicle",
    )
    if st.button("Run Master Orchestrator", key="oem_run_master"):
        master = MasterOrchestrator()
        with st.spinner("Running master agent for OEM demo..."):
            result = master.run_autonomous_workflow(
                vehicle_name=demo_vehicle,
                customer_name="OEM Fleet",
                location=(19.0760, 72.8777),
            )
        st.success(
            f"Master workflow completed with status: {result.get('final_status','N/A')}"
        )
        st.json(result)

elif tab == "UEBA Log":
    logs = load_logs()
    st.header("UEBA & Security Audit Log")

    st.subheader("Audit Timeline (Latest First)")
    audit = get_audit_timeline(logs).reset_index(drop=True)
    audit.index = range(1, len(audit) + 1)
    audit.index.name = "No."
    st.dataframe(audit, use_container_width=True)

    st.subheader("Filter by Vehicle")
    if not logs.empty:
        selected_vehicle = st.selectbox(
            "Select Vehicle",
            options=sorted(logs["vehicle_name"].unique()),
            key="ueba_vehicle",
        )
        st.dataframe(
            filter_logs(logs, vehicle_name=selected_vehicle),
            use_container_width=True,
        )

    st.subheader("Security Alerts & Anomalies")
    anomalies = get_anomalies(logs)

    def highlight_anomaly(row):
        return (
            ["background-color: #ffcccc; font-weight: bold"] * len(row)
            if any(x in str(row["status"]) for x in ["Blocked", "ALERT", "UNAUTHORIZED"])
            else [""]
            * len(row)
        )

    if not anomalies.empty:
        st.dataframe(
            anomalies.style.apply(highlight_anomaly, axis=1),
            use_container_width=True,
        )
    else:
        st.info("No anomalies detected.")

    st.subheader("Anomaly Statistics")
    st.write(anomaly_summary(logs))

    st.subheader("Behavioral Risk Score (Last 7 Days)")
    risk_df = compute_behavioral_risk(logs)
    risk_df = risk_df.reset_index(drop=True)
    risk_df.index = range(1, len(risk_df) + 1)
    risk_df.index.name = "Rank"
    st.dataframe(risk_df, use_container_width=True)

    st.subheader("Agent-to-Agent UEBA Dashboard")

    metrics, recent_interactions = agent_ueba.get_agent_dashboard()
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total interactions", int(metrics["total_interactions"]))
    mcol2.metric("Blocked interactions", int(metrics["blocked_interactions"]))
    mcol3.metric("Avg anomaly score", f"{metrics['avg_anomaly_score']:.2f}")
    mcol4.metric("Suspicious patterns", int(metrics["suspicious_patterns"]))

    st.markdown("#### Recent Agent Interactions")
    if not recent_interactions.empty:
        recent_interactions_view = recent_interactions.copy()
        recent_interactions_view.index = range(1, len(recent_interactions_view) + 1)
        recent_interactions_view.index.name = "No."
        st.dataframe(recent_interactions_view, use_container_width=True)
    else:
        st.info("No agent interaction logs yet.")

    st.markdown("#### Quarantine Agent")
    all_agents = (
        sorted(set(agent_ueba.agent_log["source_agent"].dropna().tolist()))
        if not agent_ueba.agent_log.empty
        else []
    )
    if all_agents:
        agent_to_quarantine = st.selectbox(
            "Select agent to quarantine", all_agents, key="quarantine_agent"
        )
        if st.button("Quarantine selected agent", type="secondary"):
            agent_ueba.quarantine_agent(agent_to_quarantine)
            st.warning(f"Agent {agent_to_quarantine} has been quarantined.")
    else:
        st.info("No agents available to quarantine.")

    if not logs.empty:
        if st.button("Simulate Unauthorized Access Attempt", type="secondary"):
            append_anomaly(
                selected_vehicle,
                "Diagnosis Agent",
                "Unauthorized API call detected",
                "BLOCKED by UEBA",
            )
            st.warning("Simulated security incident injected!")
            st.rerun()
