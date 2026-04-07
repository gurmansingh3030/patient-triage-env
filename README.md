---
title: Patient Triage Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Patient Complaint Triage Environment

An OpenEnv-compatible environment where an AI agent learns to triage hospital patient complaints.

## What it does
The agent reads real-world style patient complaints and must:
- Classify urgency (critical / high / low)
- Route to correct department
- Generate appropriate action plans

## Tasks
| Task | Difficulty | Description |
|------|-----------|-------------|
| task_1_urgency | Easy | Classify complaint urgency only |
| task_2_routing | Medium | Classify urgency + route to department |
| task_3_full_triage | Hard | Full triage: urgency + department + action plan |

## Observation Space
```json
{
  "task_id": "string",
  "complaint_id": "string",
  "complaint_text": "string",
  "task_description": "string"
}
```

## Action Space
```json
{
  "urgency": "critical | high | low",
  "department": "cardiology | pediatrics | orthopedics | emergency | general | dermatology | endocrinology",
  "action_plan": "immediate_admission | urgent_consultation | schedule_appointment",
  "reasoning": "string"
}
```

## Reward Function
- Task 1: 1.0 correct, 0.4 partial (serious mix), 0.0 wrong
- Task 2: 0.5 urgency + 0.5 department
- Task 3: 0.3 urgency + 0.3 department + 0.4 action plan

## Baseline Scores
| Task | Score |
|------|-------|
| task_1_urgency | 0.85 |
| task_2_routing | 0.875 |
| task_3_full_triage | ~0.90 |

## Setup
```bash
git clone <your-repo>
cd patient-triage-env
pip install -r requirements.txt
```

## Environment Variables# patient-triage-env
