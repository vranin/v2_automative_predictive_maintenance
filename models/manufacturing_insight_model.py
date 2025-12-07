import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import IsolationForest
import os

DATA_DIR = "data"

def load_feedback(path=os.path.join(DATA_DIR, "feedback.csv")):
    print(f"Loading {os.path.abspath(path)}; exists: {os.path.isfile(path)}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

def load_defects(path=os.path.join(DATA_DIR, "defects.csv")):
    print(f"Loading {os.path.abspath(path)}; exists: {os.path.isfile(path)}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()

def load_rca_capa_records(path=os.path.join(DATA_DIR, "rca_capa_records.csv")):
    print(f"Loading {os.path.abspath(path)}; exists: {os.path.isfile(path)}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] {path} not found. Returning empty DataFrame.")
        return pd.DataFrame()


# --- INSIGHT MODULE CLASS ---

class ManufacturingInsightModule:
    def __init__(self):
        self.feedback = load_feedback()
        self.defects = load_defects()
        self.rca_capa = load_rca_capa_records()
        self._add_sentiment()

    def _add_sentiment(self):
        self.feedback["sentiment"] = self.feedback["comments"].apply(self.basic_sentiment)

    @staticmethod
    def basic_sentiment(comment):
        if not isinstance(comment, str):
            return 0
        blob = TextBlob(comment)
        return blob.sentiment.polarity  # -1 (negative) to +1 (positive)

    def aggregate_feedback_insights(self):
        summary = self.feedback.groupby("vehicle_name")["user_rating"].agg(['mean', 'count']).reset_index()
        avg_global = self.feedback["user_rating"].mean()
        avg_sentiment = self.feedback["sentiment"].mean()
        low_rated = self.feedback[self.feedback["user_rating"] <= 3]
        return {
            "average_rating_by_vehicle": summary,
            "global_average_rating": avg_global,
            "average_text_sentiment": avg_sentiment,
            "low_rated_feedback": low_rated
        }

    def defect_trends(self):
        defect_summary = self.defects.groupby(["defect_type", "severity"]).size().reset_index(name="count")
        top_defects = defect_summary.sort_values("count", ascending=False).head(3)
        return {
            "defect_trends": defect_summary,
            "top_defects": top_defects
        }

    def rca_capa_summary(self):
        if self.rca_capa.empty:
            return "No RCA/CAPA records provided."
        summary = self.rca_capa.groupby("issue")["corrective_action"].apply(lambda x: list(set(x))).reset_index()
        return summary

    def anomaly_vehicle_user_ratings(self):
        if "user_rating" not in self.feedback.columns:
            return []
        X = self.feedback[["user_rating"]].values
        if len(X) < 2:
            return []
        model = IsolationForest(contamination=0.1, random_state=0)
        preds = model.fit_predict(X)
        self.feedback["is_anomaly"] = preds == -1
        outliers = self.feedback[self.feedback["is_anomaly"]]
        return outliers["vehicle_name"].unique().tolist()

    def generate_insights(self):
        insights = self.aggregate_feedback_insights()
        trends = self.defect_trends()
        rca = self.rca_capa_summary()
        outlier_vehicles = self.anomaly_vehicle_user_ratings()

        messages = []
        if trends["top_defects"].shape[0]:
            messages.append(f"Top recurring issues: {', '.join(trends['top_defects']['defect_type'])}")
        if insights["global_average_rating"] < 4.0:
            messages.append("Global user satisfaction is below target; consider design review for low-rated vehicles.")
        if insights["average_text_sentiment"] < 0.3:
            messages.append("Negative trend in feedback comments detected. Review service and investigate root causes.")
        if outlier_vehicles:
            messages.append(f"Flagged vehicles for feedback anomalies: {', '.join(outlier_vehicles)}")
        if isinstance(rca, str):
            messages.append("No design suggestions from RCA/CAPA records.")
        else:
            rca_issues = ', '.join(rca["issue"])
            messages.append(f"RCA/CAPA design suggestions related to: {rca_issues}")

        return "\n".join(messages)

# --- Example usage ---

if __name__ == "__main__":
    module = ManufacturingInsightModule()
    print("=== Feedback Insights ===")
    print(module.aggregate_feedback_insights())
    print("\n=== Defect Trends ===")
    print(module.defect_trends())
    print("\n=== RCA/CAPA Summary ===")
    print(module.rca_capa_summary())
    print("\n=== Overall Insights ===")
    print(module.generate_insights())
