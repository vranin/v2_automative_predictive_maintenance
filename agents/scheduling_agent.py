#mock scheduling agent
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.agent_logic import log_event

class SchedulingAgent:
    def get_available_slots(self, vehicle_name="Car B"):
        # Return mock available slots and centers
        return [
            {"date": "2025-11-12", "time": "10:00", "center": "Center A"},
            {"date": "2025-11-13", "time": "14:00", "center": "Center B"}
        ]

    def confirm_slot(self, vehicle_name, date, time, center):
        # Simulates confirming the requested slot
        return {
            "status": "Confirmed",
            "vehicle_name": vehicle_name,
            "date": date,
            "time": time,
            "center": center
        }

    def book_appointment(self, vehicle_name="Car B", center="Center A"):
        # Preserve your original booking logic for default bookings
        return {
            "status": "Booked",
            "center": center,
            "slot": "2025-10-22 10:00",
            "confirmation_needed": True
        }

if __name__ == "__main__":
    agent = SchedulingAgent()
    slots = agent.get_available_slots()
    print("Available slots:", slots)
    confirmed = agent.confirm_slot("Car B", "2025-11-12", "10:00", "Center A")
    print("Confirmed appointment:", confirmed)
    booking = agent.book_appointment()
    print("Quick booking:", booking)
