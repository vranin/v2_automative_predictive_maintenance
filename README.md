# Autonomous Predictive Maintenance

A voice-enabled, AI-powered predictive maintenance platform for connected vehicles.  

---

## Overview

This project leverages a team of specialized intelligent agents—each with a unique role—to automate and enhance every step of the vehicle maintenance lifecycle.  
Together, these agents collaborate to detect early issues, interact with users via voice, diagnose problems, arrange bookings, analyze trends, and gather user feedback.  
The goal is to bring modern AI-driven predictive maintenance and conversational support to vehicles, making car care proactive, efficient, and user-friendly.  

---

## Features

- **Voice Assistant:** Talk to the app to report problems, get updates, or book appointments.  
- **AI Anomaly Detection:** Real-time defect/maintenance analysis.  
- **Multi-Agent Workflow:** Including Diagnosis, Scheduling, Feedback, and Voice CEA agents.  
- **Custom Dashboard:** Visualizes logs, metrics, issues, and actions live.  

---

## Folder Structure

```
.
├── agents/        # Autonomous agent logic
├── data/          # Vehicle, defect, and feedback data
├── models/        # ML models and analytics
├── utils/         # Helper functions, security
├── app.py         # Streamlit app main script
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Agents & Modules

- **Master Agent:** Coordinates all other agents and manages the help/workflow process.  
- **Diagnosis Agent:** Finds faults and diagnoses possible car issues from logs or data.  
- **Data Analysis Agent:** Analyzes operational/history data to spot trends and improve decisions.  
- **Customer Engagement Agent (CEA):** Voice/chat assistant that talks to users, gets info, answers questions.  
- **Scheduling Agent:** Arranges maintenance bookings and optimizes schedules.  
- **Feedback Agent:** Collects user ratings/comments after service for improvement.  
- **Manufacturing Quality Insight (MQI) Module:** Spots long-term patterns in all data for better vehicle design and planning.  

---

## Quick Start

1. **Clone the repo:**
   ```bash
   git clone https://github.com/<your-github-user>/<your-repo>.git
   cd <repo>
   ```

2. **Install all requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your secrets/config in `.env`:**
   ```
   (Do not commit this file!)
   ```

4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Tech Stack

- **Python**  
- **pandas**  
- **scikit-learn**  
- **textblob**  
- **dotenv**  
- **LangChain**  
- **Modular Agent Design**  
- **Voice/NLP**  
- **ML/Analytics**  
- **Data (CSV or Database)**  

---

**Future Enhancements**
- Integration with real-time IoT vehicle data streams.
- Building logic of all agents through logic
- Enhanced conversational understanding via fine-tuned LLMs.  
- Predictive failure forecasting dashboards.  
- API endpoints for third-party integration (OEMs, service centers).  

@"
# Autonomous Predictive Maintenance (v2 - Team Ready)

## Quick Start (works for anyone who clones)

```bash
git clone https://github.com/vranin/v2_automative_predictive_maintenance.git
cd v2_automative_predictive_maintenance

# Create virtual env
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install exact working packages
pip install -r requirements.txt

# Copy example env and fill your keys
copy .env.example .env

# Run
streamlit run app.py