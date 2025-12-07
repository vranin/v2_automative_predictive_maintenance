import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import os

DATA_DIR = "data"
LOGS_PATH = os.path.join(DATA_DIR, "logs.csv")

# Load system/UEBA logs
def load_logs(path=LOGS_PATH):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

# Filter logs for specific vehicle or agent
def filter_logs(logs_df, vehicle_name=None, agent=None):
    df = logs_df
    if vehicle_name:
        df = df[df["vehicle_name"] == vehicle_name]
    if agent:
        df = df[df["agent"] == agent]
    return df

# Identify and return ALL anomalous (alert/blocked) log events 
def get_anomalies(logs_df):
    return logs_df[(logs_df["status"].str.contains("Blocked", na=False)) | 
                   (logs_df["status"].str.contains("ALERT", na=False))]

# Add a UEBA anomaly event manually (for demo/testing)
def append_anomaly(vehicle, agent, action, details, path=LOGS_PATH):
    try:
        logs = pd.read_csv(path)
    except FileNotFoundError:
        logs = pd.DataFrame()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    anomaly_entry = {
        "timestamp": timestamp,
        "vehicle_name": vehicle,
        "agent": agent,
        "action": action,
        "event_type": "ueba",
        "status": "Blocked",
        "details": details
    }
    logs = pd.concat([logs, pd.DataFrame([anomaly_entry])], ignore_index=True)
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logs.to_csv(path, index=False)
    return logs

# Get summary statistics of anomalies (total count, last event)
def anomaly_summary(logs_df):
    anomalies = get_anomalies(logs_df)
    total = len(anomalies)
    last_event = anomalies.tail(1).to_dict('records') if total > 0 else "None"
    return {"total_anomalies": total, "last_event": last_event}

# Return full chronological audit log (desc/asc) for dashboard display
def get_audit_timeline(logs_df, descending=True):
    return logs_df.sort_values("timestamp", ascending=not descending)

# --- Behavioral Baseline with Simple Risk Score ---
def compute_behavioral_risk(logs_df, entity_col="vehicle_name", risk_window_days=7):
    logs_df["date"] = pd.to_datetime(logs_df["timestamp"]).dt.date
    end_date = logs_df["date"].max()
    start_date = end_date - pd.Timedelta(days=risk_window_days-1)
    window_logs = logs_df[(logs_df["date"] >= start_date) & (logs_df["date"] <= end_date)]
    # Calculate baseline (mean actions/entity per day)
    baseline = window_logs.groupby([entity_col, "date"]).size().groupby(entity_col).mean()
    today = end_date
    today_counts = window_logs[window_logs["date"] == today].groupby(entity_col).size()
    risk_df = pd.DataFrame({
        "baseline": baseline,
        "today": today_counts
    }).fillna(0)
    risk_df["deviation"] = risk_df["today"] / risk_df["baseline"].replace(0, 1)
    if len(risk_df) > 1:
        model = IsolationForest(contamination=0.2, random_state=42)
        risk_df["is_anomaly"] = model.fit_predict(risk_df[["deviation"]]) == -1
    else:
        risk_df["is_anomaly"] = False
    return risk_df.reset_index()
