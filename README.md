---
title: Patient Triage Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Patient Complaint Triage Environment

An OpenEnv-compatible environment where an agent reads a patient complaint and does what a triage nurse would: decide how urgent it is, pick the department, and pick the next action. Built for the Meta PyTorch Hackathon x Scaler SST, Round 1.

- Live Space: https://huggingface.co/spaces/gurmansingh3030/patient-triage-env
- API docs: https://gurmansingh3030-patient-triage-env.hf.space/docs
- GitHub: https://github.com/gurmansingh3030/patient-triage-env

## Why

A large share of ER visits are non-urgent, and in India there aren't enough trained triage staff to sort them. I wanted an environment where an LLM agent gets scored on triage decisions, so it can be tested (and later trained) on this.

## Tasks

| Task | Difficulty | What the agent does |
|---|---|---|
| `task_1_urgency` | Easy | Classify urgency: critical / high / low |
| `task_2_routing` | Medium | Urgency + department |
| `task_3_full_triage` | Hard | Urgency + department + action plan |

## Observation

```json
{
  "task_id": "string",
  "complaint_id": "string",
  "complaint_text": "the patient's complaint",
  "task_description": "what the agent has to do"
}
```

## Action

```json
{
  "urgency": "critical | high | low",
  "department": "cardiology | pediatrics | orthopedics | emergency | general | dermatology | endocrinology",
  "action_plan": "immediate_admission | urgent_consultation | schedule_appointment",
  "reasoning": "short explanation"
}
```

## Rewards

| Task | Scoring |
|---|---|
| Task 1 | 1.0 correct, 0.4 if it confuses critical and high, 0.0 otherwise |
| Task 2 | 0.5 urgency + 0.5 department |
| Task 3 | 0.3 urgency + 0.3 department + 0.4 action plan |

Partial credit is there so the agent gets a signal even when it's only half right.

## Baseline

Run with `llama-3.1-8b-instant` through Groq:

| Task | Score |
|---|---|
| task_1_urgency | 0.85 |
| task_2_routing | 0.875 |
| task_3_full_triage | 0.85 |

## Setup

```bash
git clone https://github.com/gurmansingh3030/patient-triage-env
cd patient-triage-env
pip install -r requirements.txt
```

Set these environment variables:

```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=your_api_key
```

Run the baseline:

```bash
python3 inference.py
```

## Docker

```bash
docker build -t patient-triage-env .
docker run \
  -e HF_TOKEN=your_key \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-8b-instant \
  patient-triage-env
```

## Files

```
patient-triage-env/
├── inference.py      # baseline agent
├── environment.py    # the environment
├── tasks.py          # tasks and graders
├── data.py           # patient complaints
├── app.py            # FastAPI server
├── openenv.yaml      # OpenEnv metadata
├── Dockerfile
└── requirements.txt
```

## Limitations / next steps

- The complaint set is small, so scores are noisy.
- Only English complaints right now. Hindi/English support is the next thing I want to add.
- Not a medical tool. It's a benchmark for agents, not for real triage decisions.
