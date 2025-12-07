#mock diagnosis agent
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.agent_logic import load_defects


class DiagnosisAgent:
    def process_vehicle(self, vehicle_name="Car B"):
        defects_df = load_defects()
        v_defects = defects_df[defects_df["vehicle_name"] == vehicle_name]
        if not v_defects.empty:
            defect = v_defects.iloc[-1].to_dict()
            return defect
        else:
            return {"message": "No defects detected, vehicle healthy."}

if __name__ == "__main__":
    agent = DiagnosisAgent()
    print(agent.process_vehicle())

