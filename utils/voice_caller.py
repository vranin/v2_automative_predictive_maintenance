# utils/voice_caller.py — FIXED IMPORTS + FULL CEA EDGE CASES
from pyVoIP.VoIP import VoIPPhone, InvalidStateError, CallState  # CORRECT IMPORT!
import threading
import time
import wave
import requests  # For calling your Scheduling Agent
import os

# Your free SIP credentials (from sip.linphone.org)
SIP_SERVER_IP = "sip.linphone.org"  # Or your SIP host
SIP_SERVER_PORT = 5060
SIP_USERNAME = "justhere12"  # ← CHANGE THIS
SIP_PASSWORD = ".Qs*YV!rUj8BuTW"  # ← CHANGE THIS
MY_IP = "192.168.1.101"  

phone = None

def handle_call(call):  # Callback for incoming calls (triggered by SIP URI call)
    # Simulate metadata from your dashboard (vehicle_id, risk) — in real: pass via SIP headers
    vehicle_id = "V0001"  # Get from call headers or DB query
    risk = "medium"  # Default; set via dashboard trigger

    try:
        call.answer()
        
        if risk == "high":
            # Edge case: Urgent auto-schedule (from your slide's red lightning)
            with wave.open('audio/urgent.wav', 'rb') as f:
                data = f.readframes(f.getnframes())
            call.writeAudio(data)  # Play urgent message
            print(f"URGENT: Auto-booking {vehicle_id}")
            # Call your Scheduling Agent (HTTP to your FastAPI/Streamlit)
            requests.post("http://localhost:8501/schedule-urgent", json={"vehicle_id": vehicle_id}, timeout=5)
            status = "auto_booked"
        else:
            # Normal flow: Explain + prompt (from your slide's normal path)
            with wave.open('audio/alert.wav', 'rb') as f:
                data = f.readframes(f.getnframes())
            call.writeAudio(data)  # Play alert
            digit = call.get_dtmf(timeout=15)  # Wait for keypad press (DTMF)
            if digit == "1":
                with wave.open('audio/booked.wav', 'rb') as f:
                    data = f.readframes(f.getnframes())
                call.writeAudio(data)
                requests.post("http://localhost:8501/book", json={"vehicle_id": vehicle_id}, timeout=5)
                status = "accepted"
            else:
                with wave.open('audio/reminder.wav', 'rb') as f:
                    data = f.readframes(f.getnframes())
                call.writeAudio(data)
                requests.post("http://localhost:8501/reminder", json={"vehicle_id": vehicle_id, "days": 2}, timeout=5)
                status = "declined"

        # Post-service feedback trigger (from your slide's loop)
        with wave.open('audio/booked.wav', 'rb') as f:  # Reuse or add feedback.wav
            data = f.readframes(f.getnframes())
        call.writeAudio(data)  # "How was service? Press 1-5"

        time.sleep(2)  # Brief pause
        call.hangup()
        
        # Log to your central DB (from architecture slide)
        print(f"Call ended for {vehicle_id}: {status}")
        
    except InvalidStateError:
        print("Call state invalid")
        call.hangup()
    except Exception as e:
        print(f"Error in call: {e}")
        call.hangup()

def start_sip_phone():
    global phone
    phone = VoIPPhone(
        SIP_SERVER_IP,
        SIP_SERVER_PORT,
        SIP_USERNAME,
        SIP_PASSWORD,
        callCallback=handle_call,  # Your CEA logic here
        myIP=MY_IP,
        sipPort=5060,
        rtpPortLow=10000,  # RTP port range (open in firewall if needed)
        rtpPortHigh=20000
    )
    phone.start()
    print(f"SIP phone running! Call: sip:{SIP_USERNAME}@{SIP_SERVER_IP} to test CEA.")
    print("Waiting for calls... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        phone.stop()

# Start in background when imported (for Streamlit)
if __name__ == "__main__":
    start_sip_phone()
else:
    # Background thread for Streamlit
    threading.Thread(target=start_sip_phone, daemon=True).start()
    time.sleep(5)  # Wait for SIP registration