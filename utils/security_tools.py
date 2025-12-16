import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from langchain_openai import ChatOpenAI

class AgentUEBA:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent_log = self._init_agent_log()
        self.isolation_forest = self._train_behavior_model()
    
    def _init_agent_log(self):
        """Agent interaction audit trail"""
        log_path = "data/agent_interactions.csv"
        if Path(log_path).exists():
            return pd.read_csv(log_path)
        
        cols = [
            'timestamp', 'source_agent', 'target_agent', 'vehicle_name', 
            'action_type', 'data_size', 'response_time_ms', 'anomaly_score',
            'cross_agent_calls', 'data_consistency', 'blocked'
        ]
        empty_df = pd.DataFrame(columns=cols)
        empty_df.to_csv(log_path, index=False)
        return empty_df
    
    def _train_behavior_model(self):
        """ML model for inter-agent anomaly detection"""
        features = ['data_size', 'response_time_ms', 'cross_agent_calls']
        if len(self.agent_log) > 50:
            X = self.agent_log[features].fillna(0)
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X)
            return model
        return None
    
    def monitor_agent_call(self, source_agent: str, target_agent: str, 
                          vehicle_name: str, action_type: str, 
                          data_size: int = 0, response_time: float = 0) -> Dict:
        """Real-time agent interaction monitoring"""
        
        # 1. Log interaction
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source_agent': source_agent,
            'target_agent': target_agent,
            'vehicle_name': vehicle_name,
            'action_type': action_type,
            'data_size': data_size,
            'response_time_ms': response_time,
            'cross_agent_calls': self._get_cross_calls(source_agent),
            'data_consistency': self._check_data_consistency(vehicle_name),
        }
        
        # 2. Anomaly scoring
        anomaly_score = self._calculate_anomaly(log_entry)
        log_entry['anomaly_score'] = anomaly_score
        log_entry['blocked'] = anomaly_score > 0.5
        
        # 3. Save log
        self.agent_log = pd.concat([self.agent_log, pd.DataFrame([log_entry])], ignore_index=True)
        self.agent_log.to_csv("data/agent_interactions.csv", index=False)
        
        return {
            "allowed": not log_entry['blocked'],
            "anomaly_score": anomaly_score,
            "risk_factors": self._get_risk_factors(log_entry),
            "log_entry": log_entry
        }
    
    def _calculate_anomaly(self, log_entry: Dict) -> float:
        """ML + rule-based anomaly detection"""
        score = 0
        
        # ML anomaly (if trained)
        if self.isolation_forest:
            features = np.array([[log_entry['data_size'], log_entry['response_time_ms'], log_entry['cross_agent_calls']]])
            ml_anomaly = self.isolation_forest.decision_function(features)[0]
            score += (1 - ml_anomaly) * 0.4  # Normalize
        
        # Rule-based risks
        if log_entry['cross_agent_calls'] > 5:  # Agent hopping
            score += 0.2
        if log_entry['response_time_ms'] > 5000:  # Slow agent
            score += 0.15
        if log_entry['data_consistency'] < 0.8:  # Data mismatch
            score += 0.25
        
        return min(score, 1.0)
    
    def _get_cross_calls(self, source_agent: str) -> int:
        """How many different agents this agent called recently"""
        recent = self.agent_log[
            (self.agent_log['source_agent'] == source_agent) & 
            (self.agent_log['timestamp'] > (datetime.now() - timedelta(hours=1)).isoformat())
        ]
        return recent['target_agent'].nunique()
    
    def _check_data_consistency(self, vehicle_name: str) -> float:
        """Cross-check data across agents"""
        # Simulate data consistency check
        # Production: Compare diagnosis vs feedback vs scheduling data
        return np.random.uniform(0.7, 1.0)
    
    def _get_risk_factors(self, log_entry: Dict) -> List[str]:
        """Explainable risk factors"""
        risks = []
        if log_entry['cross_agent_calls'] > 5:
            risks.append("Excessive agent hopping")
        if log_entry['response_time_ms'] > 5000:
            risks.append("Agent response timeout")
        if log_entry['data_consistency'] < 0.8:
            risks.append("Data inconsistency detected")
        return risks
    
    def get_agent_dashboard(self):
        """Live agent behavior analytics"""
        recent = self.agent_log.tail(20)

        if self.agent_log.empty:
            metrics = {
                "total_interactions": 0,
                "blocked_interactions": 0,
                "avg_anomaly_score": 0.0,
                "busiest_agent": "N/A",
                "suspicious_patterns": 0,
            }
        else:
            metrics = {
                "total_interactions": len(self.agent_log),
                "blocked_interactions": len(self.agent_log[self.agent_log["blocked"]]),
                "avg_anomaly_score": float(self.agent_log["anomaly_score"].mean()),
                "busiest_agent": self.agent_log["source_agent"]
                .value_counts()
                .index[0],
                "suspicious_patterns": len(
                    self.agent_log[self.agent_log["anomaly_score"] > 0.5]
                ),
            }

        return metrics, recent

    
    def quarantine_agent(self, agent_name: str):
        """Emergency: Block rogue agent"""
        self.agent_log.loc[self.agent_log['source_agent'] == agent_name, 'blocked'] = True
        self.agent_log.to_csv("data/agent_interactions.csv", index=False)
        print(f"🚨 AGENT QUARANTINED: {agent_name}")



# Global instance
agent_ueba = AgentUEBA()

def filter_logs(logs: pd.DataFrame, vehicle_name: str) -> pd.DataFrame:
    return logs[logs["vehicle_name"] == vehicle_name].copy()

def get_anomalies(logs: pd.DataFrame) -> pd.DataFrame:
    if "status" not in logs.columns:
        return pd.DataFrame(columns=logs.columns)
    mask = logs["status"].astype(str).str.contains(
        "Blocked|ALERT|UNAUTHORIZED", case=False, na=False
    )
    return logs[mask].copy()

def append_anomaly(vehicle_name: str, source: str, message: str, status: str) -> None:
    log_path = "data/security_logs.csv"
    if Path(log_path).exists():
        df = pd.read_csv(log_path)
    else:
        df = pd.DataFrame(
            columns=["timestamp", "vehicle_name", "source", "message", "status"]
        )
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "vehicle_name": vehicle_name,
        "source": source,
        "message": message,
        "status": status,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(log_path, index=False)

def anomaly_summary(logs: pd.DataFrame) -> dict:
    if logs.empty or "status" not in logs.columns:
        return {"total_events": 0, "alerts": 0, "blocked": 0}
    total = len(logs)
    alerts = logs["status"].astype(str).str.contains("ALERT", case=False, na=False).sum()
    blocked = logs["status"].astype(str).str.contains("Blocked", case=False, na=False).sum()
    return {
        "total_events": int(total),
        "alerts": int(alerts),
        "blocked": int(blocked),
    }

def get_audit_timeline(logs: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in logs.columns:
        return logs.copy()
    df = logs.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp", ascending=False)

def compute_behavioral_risk(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty or "vehicle_name" not in logs.columns:
        return pd.DataFrame(columns=["vehicle_name", "risk_score"])
    grouped = logs.groupby("vehicle_name")
    counts = grouped.size().rename("event_count")
    blocked = grouped["status"].apply(
        lambda s: s.astype(str).str.contains("Blocked", case=False, na=False).sum()
    )
    risk = 0.3 * counts + 0.7 * blocked
    out = pd.DataFrame({"vehicle_name": risk.index, "risk_score": risk.values})
    return out.sort_values("risk_score", ascending=False)
