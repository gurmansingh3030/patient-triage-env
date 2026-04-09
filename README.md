---
title: Patient Triage Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏥 Patient Complaint Triage Environment

> An OpenEnv-compatible AI agent training environment for hospital patient complaint triage — built for the Meta PyTorch Hackathon x Scaler SST, Round 1.

## 🎯 Overview

In real hospitals, triage nurses must quickly assess patient complaints and decide:
- How urgent is this? (critical / high / low)
- Which department should handle it?
- What immediate action is needed?

This environment trains AI agents to perform this exact task — making triage decisions that match expert medical judgment. With 40% of emergency department visits being non-urgent, intelligent triage can save lives and reduce healthcare costs.

## 🚀 Live Demo

- **HuggingFace Space:** https://huggingface.co/spaces/gurmansingh3030/patient-triage-env
- **API Docs:** https://gurmansingh3030-patient-triage-env.hf.space/docs
- **GitHub:** https://github.com/gurmansingh3030/patient-triage-env

## 📊 Tasks

| Task | Difficulty | Description | Max Score |
|------|-----------|-------------|-----------|
| `task_1_urgency` | 🟢 Easy | Classify complaint urgency (critical/high/low) | 1.0 |
| `task_2_routing` | 🟡 Medium | Classify urgency + route to correct department | 1.0 |
| `task_3_full_triage` | 🔴 Hard | Full triage: urgency + department + action plan | 1.0 |

## 📈 Baseline Scores

| Task | Score | Model |
|------|-------|-------|
| task_1_urgency | 0.85 | llama-3.1-8b-instant |
| task_2_routing | 0.875 | llama-3.1-8b-instant |
| task_3_full_triage | 0.85 | llama-3.1-8b-instant |

## 🔍 Observation Space

```json
{
  "task_id": "string",
  "complaint_id": "string",
  "complaint_text": "string — real patient complaint",
  "task_description": "string — what the agent must do"
}
```

## ⚡ Action Space

```json
{
  "urgency": "critical | high | low",
  "department": "cardiology | pediatrics | orthopedics | emergency | general | dermatology | endocrinology",
  "action_plan": "immediate_admission | urgent_consultation | schedule_appointment",
  "reasoning": "string — agent's explanation"
}
```

## 🏆 Reward Function

| Task | Scoring Logic |
|------|--------------|
| Task 1 | 1.0 correct, 0.4 partial (critical/high mix), 0.0 wrong |
| Task 2 | 0.5 urgency + 0.5 department |
| Task 3 | 0.3 urgency + 0.3 department + 0.4 action plan |

Partial rewards ensure meaningful signal throughout the trajectory — not just at episode end.

## 🛠️ Setup

```bash
git clone https://github.com/gurmansingh3030/patient-triage-env
cd patient-triage-env
pip install -r requirements.txt
```

## 🔑 Environment Variables

```bash
API_BASE_URL=https://api.groq.com/openai/v1   # LLM endpoint
MODEL_NAME=llama-3.1-8b-instant               # Model to use
HF_TOKEN=your_api_key                         # Your API key
```

## ▶️ Run Baseline

```bash
python3 inference.py
```

## 🐳 Docker

```bash
docker build -t patient-triage-env .
docker run \
  -e HF_TOKEN=your_key \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-8b-instant \
  patient-triage-env
```

## 🌍 Real-World Impact

- **40%** of ER visits are non-urgent — intelligent triage reduces overcrowding
- **India context** — 1.3 billion people, shortage of trained triage staff
- **Multilingual potential** — environment designed to support Hindi/English complaints
- Trained agents can assist ASHA workers and rural health centers

## 📁 Project Structure

```
patient-triage-env/
├── inference.py        # Baseline agent script
├── environment.py      # Core OpenEnv environment
├── tasks.py            # 3 tasks + graders
├── data.py             # Patient complaints dataset
├── app.py              # FastAPI server
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # Container config
└── requirements.txt    # Dependencies
```