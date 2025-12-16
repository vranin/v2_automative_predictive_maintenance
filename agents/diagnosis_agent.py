import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class DiagnosisAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isoforest = IsolationForest(contamination=0.1, random_state=42)

        self.vehicles = self._load_vehicles()
        self.defects = (
            pd.read_csv("data/defects.csv")
            if Path("data/defects.csv").exists()
            else pd.DataFrame()
        )
        self.telematics = self._load_kaggle_telematics()
        self._fit_models()

    def _load_vehicles(self):
        vehicles_path = "data/vehicles.csv"
        if Path(vehicles_path).exists():
            return pd.read_csv(vehicles_path)
        raise FileNotFoundError("vehicles.csv not found in data/ folder")

    def _load_kaggle_telematics(self):
        kaggle_path = "data/Telematicsdata.csv"
        if not Path(kaggle_path).exists():
            raise FileNotFoundError("Put Telematicsdata.csv in data/ folder")

        df = pd.read_csv(kaggle_path)
        df["parsed_value"] = df["value"].astype(str).apply(self._parse_telematics_value)

        df["battery_internal"] = df.apply(
            lambda row: row["parsed_value"]
            if row["variable"] == "INTERNAL BATTERY"
            else np.nan,
            axis=1,
        )
        df["battery_external"] = df.apply(
            lambda row: row["parsed_value"]
            if row["variable"] == "EXTERNAL BATTERY"
            else np.nan,
            axis=1,
        )
        df["towing"] = df.apply(
            lambda row: row["parsed_value"] if row["variable"] == "TOWING" else 0,
            axis=1,
        )
        df["ignition"] = df.apply(
            lambda row: row["parsed_value"]
            if row["variable"] == "IGNITION_STATUS"
            else 0,
            axis=1,
        )
        df["alarm_class"] = df["alarmClass"]

        df["battery_internal"] = df["battery_internal"].ffill().fillna(50)
        df["battery_external"] = df["battery_external"].ffill().fillna(36)

        df = df.reset_index(drop=True)

        vehicle_ids = self.vehicles["vehicle_id"].tolist()  # [101,102,103,104]
        df["vehicle_id"] = df.index.map(lambda i: vehicle_ids[i % len(vehicle_ids)])

        telematics = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    df["timestamp"].astype(str) + ":00",
                    format="%H:%M:%S",
                    errors="coerce",
                ),
                "vehicle_id": df["vehicle_id"],
                "battery_voltage": (df["battery_internal"] + df["battery_external"]) / 2,
                "alarm_level": df["alarm_class"],
                "towing_status": df["towing"],
                "ignition_status": df["ignition"],
                "vibration": df["alarm_class"] * 0.5,
            }
        )
        telematics.to_csv("data/telematics.csv", index=False)
        print(f"✅ Converted {len(df)} Kaggle rows → {len(telematics)} standardized rows")
        return telematics.dropna()

    def _parse_telematics_value(self, value):
        val = str(value).strip()
        if "," in val:
            return 0
        try:
            return float(val)
        except Exception:
            return 0

    def _fit_models(self):
        features = [
            "battery_voltage",
            "alarm_level",
            "towing_status",
            "ignition_status",
            "vibration",
        ]
        X = self.telematics[features].fillna(self.telematics[features].mean())
        if len(X) > 10:
            X_scaled = self.scaler.fit_transform(X)
            self.isoforest.fit(X_scaled)

    def continuous_monitor(self, vehicle_id: str):
        """Real-time diagnosis using telematics.
        Accepts either numeric vehicle_id ('102') or vehicle_name ('Car B').
        """
        # If it's not numeric, treat it as vehicle_name and map to id
        if not str(vehicle_id).isdigit():
            row = self.vehicles[self.vehicles["vehicle_name"] == vehicle_id]
            if row.empty:
                # Fallback if name not found
                return {
                    "vehicle_id": vehicle_id,
                    "anomaly_score": 0.0,
                    "risk_level": "low",
                    "predicted_failure": "none",
                    "urgency": "14d",
                    "battery_internal": 12.6,
                    "alarms_triggered": 0,
                }
            vid = int(row.iloc[0]["vehicle_id"])
        else:
            vid = int(vehicle_id)

        vehicle_data = self.telematics[self.telematics["vehicle_id"] == vid].tail(50)

        # Get static status from vehicles.csv
        row = self.vehicles[self.vehicles["vehicle_id"] == vid]
        status = row.iloc[0]["status"] if not row.empty else ""

        if vehicle_data.empty:
            return {
                "vehicle_id": vid,
                "anomaly_score": 0.0,
                "risk_level": "low",
                "predicted_failure": "none",
                "urgency": "14d",
                "battery_internal": 12.6,
                "alarms_triggered": 0,
            }

        features = [
            "battery_voltage",
            "alarm_level",
            "towing_status",
            "ignition_status",
            "vibration",
        ]
        X = vehicle_data[features].fillna(method="ffill").tail(10)
        X_scaled = self.scaler.transform(X)

        anomaly_score = self.isoforest.decision_function(X_scaled)[-1]
        latest = vehicle_data.iloc[-1]

        diagnosis = self._rule_based_diagnosis(latest, anomaly_score, status)

        return {
            "vehicle_id": vid,
            "anomaly_score": anomaly_score,
            "risk_level": diagnosis["risk"],
            "predicted_failure": diagnosis["failure_type"],
            "urgency": diagnosis["urgency"],
            "battery_internal": latest["battery_voltage"],
            "alarms_triggered": int(latest["alarm_level"]),
        }

    def _rule_based_diagnosis(self, latest, anomaly_score: float, status: str):
        """Use telematics + status + anomaly_score to label risk/failure/urgency."""
        bv = latest["battery_voltage"]
        alarms = latest["alarm_level"]
        towing = latest["towing_status"]
        ignition = latest["ignition_status"]

        # explicit high/critical cases from vehicles.csv
        if "Fault: Brake Issue" in status:
            return {"risk": "critical", "failure_type": "brake", "urgency": "2d"}
        if "Fault: Oil Leak" in status:
            return {"risk": "high", "failure_type": "oil_leak", "urgency": "3d"}

        # sensor-based rules for others
        if bv <= 11.8 or alarms >= 3:
            risk = "high"
            failure_type = "battery"
            urgency = "3d"
        elif towing == 1 and ignition == 0:
            risk = "critical"
            failure_type = "towing"
            urgency = "1d"
        elif anomaly_score < -0.1:
            risk = "medium"
            failure_type = "telematics"
            urgency = "7d"
        else:
            risk = "low"
            failure_type = "none"
            urgency = "14d"

        return {"risk": risk, "failure_type": failure_type, "urgency": urgency}


if __name__ == "__main__":
    agent = DiagnosisAgent()
    print("Car A:", agent.continuous_monitor("101"))
    print("Car B:", agent.continuous_monitor("102"))
    print("Car C:", agent.continuous_monitor("103"))
    print("Car D:", agent.continuous_monitor("104"))
