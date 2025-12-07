#mock feedback agent
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.agent_logic import load_feedback
class FeedbackAgent:
    def request_feedback(self, vehicle_name="Car B"):
        # Return a mock feedback prompt
        return {
            "prompt": f"Please rate your recent service for {vehicle_name} (1-5 stars) and leave comments.",
            "expected_fields": ["user_rating", "comments"]
        }

    def aggregate_feedback(self, feedback_list):
        # Aggregate feedback (mock summary)
        count = len(feedback_list)
        avg_rating = sum(f["user_rating"] for f in feedback_list) / count if count else 0
        return {
            "total_feedback": count,
            "average_rating": avg_rating
        }

if __name__ == "__main__":
    agent = FeedbackAgent()
    print(agent.request_feedback())
    print(agent.aggregate_feedback([{"user_rating": 4}, {"user_rating": 5}]))
