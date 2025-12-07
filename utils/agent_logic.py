import pandas as pd
from datetime import datetime
import os

DATA_DIR = "data"  # Make sure the 'data' folder is in your project root

# Utility function to load vehicle data
def load_vehicles(path=os.path.join(DATA_DIR, "vehicles.csv")):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

# Load defects for analytics, status, or agent workflow
def load_defects(path=os.path.join(DATA_DIR, "defects.csv")):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

# Load historical logs for workflow/UEBA
def load_logs(path=os.path.join(DATA_DIR, "logs.csv")):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

# Load user feedback
def load_feedback(path=os.path.join(DATA_DIR, "feedback.csv")):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

# Append an event to logs.csv
def log_event(
    vehicle_name, agent, action, event_type, status, details,
    output='', user_confirm='', date='', center='', path=os.path.join(DATA_DIR, "logs.csv")
):
    # Prepare row dict with all expected columns
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'vehicle_name': vehicle_name,
        'agent': agent,
        'action': action,
        'event_type': event_type,
        'status': status,
        'details': details,
        'output': output,
        'user_confirm': user_confirm,
        'date': date,
        'center': center
    }
    try:
        logs = pd.read_csv(path)
    except FileNotFoundError:
        logs = pd.DataFrame()
    
    logs = pd.concat([logs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
    logs.to_csv(path, index=False)
    return logs

