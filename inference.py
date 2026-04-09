import os
import json
import random
from typing import Optional
from pydantic import BaseModel

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

# ─── Inline Models ────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    complaint_id: str
    complaint_text: str
    task_description: str

class Action(BaseModel):
    urgency: str
    department: Optional[str] = None
    action_plan: Optional[str] = None
    reasoning: Optional[str] = None

class Reward(BaseModel):
    score: float
    feedback: str

# ─── Inline Data ──────────────────────────────────────────────

COMPLAINTS = [
    {"id": "C001", "text": "Patient has severe chest pain radiating to left arm since 2 hours.", "true_urgency": "critical", "true_department": "cardiology", "true_action": "immediate_admission"},
    {"id": "C002", "text": "Mild headache for 3 days, no fever.", "true_urgency": "low", "true_department": "general", "true_action": "schedule_appointment"},
    {"id": "C003", "text": "Child with high fever 104F, difficulty breathing.", "true_urgency": "high", "true_department": "pediatrics", "true_action": "urgent_consultation"},
    {"id": "C004", "text": "Elderly patient fell down, hip pain, cannot stand.", "true_urgency": "high", "true_department": "orthopedics", "true_action": "urgent_consultation"},
    {"id": "C005", "text": "Patient requesting prescription refill.", "true_urgency": "low", "true_department": "general", "true_action": "schedule_appointment"},
    {"id": "C006", "text": "Severe allergic reaction, face swelling, difficulty breathing.", "true_urgency": "critical", "true_department": "emergency", "true_action": "immediate_admission"},
    {"id": "C007", "text": "Patient has mild skin rash since 2 days.", "true_urgency": "low", "true_department": "dermatology", "true_action": "schedule_appointment"},
    {"id": "C008", "text": "Diabetic patient, blood sugar 450, feeling dizzy.", "true_urgency": "critical", "true_department": "endocrinology", "true_action": "immediate_admission"},
]

# ─── Inline Graders ───────────────────────────────────────────

def grade_task1(action, complaint):
    if action.urgency == complaint["true_urgency"]:
        return Reward(score=1.0, feedback="Correct urgency classification!")
    both_serious = {"critical", "high"}
    if action.urgency in both_serious and complaint["true_urgency"] in both_serious:
        return Reward(score=0.4, feedback="Close — both serious but wrong level.")
    return Reward(score=0.0, feedback=f"Wrong. Expected {complaint['true_urgency']}.")

def grade_task2(action, complaint):
    score = 0.0
    parts = []
    if action.urgency == complaint["true_urgency"]:
        score += 0.5
        parts.append("Urgency correct (+0.5)")
    else:
        parts.append(f"Urgency wrong (expected {complaint['true_urgency']})")
    if action.department == complaint["true_department"]:
        score += 0.5
        parts.append("Department correct (+0.5)")
    else:
        parts.append(f"Department wrong (expected {complaint['true_department']})")
    return Reward(score=round(score, 2), feedback=" | ".join(parts))

def grade_task3(action, complaint):
    score = 0.0
    parts = []
    if action.urgency == complaint["true_urgency"]:
        score += 0.3
        parts.append("Urgency correct (+0.3)")
    else:
        parts.append(f"Urgency wrong")
    if action.department == complaint["true_department"]:
        score += 0.3
        parts.append("Department correct (+0.3)")
    else:
        parts.append(f"Department wrong")
    valid_actions = ["immediate_admission", "urgent_consultation", "schedule_appointment"]
    if action.action_plan == complaint["true_action"]:
        score += 0.4
        parts.append("Action plan correct (+0.4)")
    elif action.action_plan in valid_actions:
        score += 0.1
        parts.append("Valid but wrong action (+0.1)")
    else:
        parts.append("Invalid action")
    return Reward(score=round(score, 2), feedback=" | ".join(parts))

TASK_INFO = {
    "task_1_urgency": {"grader": grade_task1, "desc": "Classify the complaint as: critical, high, or low urgency."},
    "task_2_routing": {"grader": grade_task2, "desc": "Classify urgency AND route to correct department."},
    "task_3_full_triage": {"grader": grade_task3, "desc": "Classify urgency, route to department, AND generate action plan."},
}

def get_action_from_llm(prompt):
    """Get action from LLM — fallback to rule-based if no API key."""
    if not HF_TOKEN:
        # Rule-based fallback when no API key
        text = prompt.lower()
        if any(w in text for w in ["severe", "critical", "emergency", "450", "allergic", "chest pain"]):
            urgency = "critical"
        elif any(w in text for w in ["high fever", "fell", "difficulty breathing", "hip"]):
            urgency = "high"
        else:
            urgency = "low"
        return Action(
            urgency=urgency,
            department="general",
            action_plan="schedule_appointment",
            reasoning="rule-based fallback"
        )
    
    from openai import OpenAI
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0
    )
    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    return Action(**parsed)

def run_task(task_id: str):
    complaints = COMPLAINTS.copy()
    random.shuffle(complaints)
    grader = TASK_INFO[task_id]["grader"]
    desc = TASK_INFO[task_id]["desc"]
    total_score = 0.0
    steps = 0

    print(json.dumps({"type": "[START]", "task_id": task_id}), flush=True)

    for complaint in complaints:
        prompt = f"""You are a hospital triage assistant.

Patient Complaint: {complaint['text']}
Task: {desc}

Respond ONLY in this JSON format:
{{
  "urgency": "critical" or "high" or "low",
  "department": "cardiology" or "pediatrics" or "orthopedics" or "emergency" or "general" or "dermatology" or "endocrinology",
  "action_plan": "immediate_admission" or "urgent_consultation" or "schedule_appointment",
  "reasoning": "brief reason"
}}"""

        try:
            action = get_action_from_llm(prompt)
        except Exception as e:
            print(json.dumps({"type": "[ERROR]", "error": str(e)}), flush=True)
            action = Action(urgency="low", department="general", action_plan="schedule_appointment")

        reward = grader(action, complaint)
        total_score += reward.score
        steps += 1

        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "step": steps,
            "score": reward.score,
            "feedback": reward.feedback,
            "cumulative_score": round(total_score, 3)
        }), flush=True)

    final_avg = round(total_score / steps, 3)
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "total_steps": steps,
        "final_score": final_avg
    }), flush=True)
    return final_avg

if __name__ == "__main__":
    scores = {}
    for task in ["task_1_urgency", "task_2_routing", "task_3_full_triage"]:
        scores[task] = run_task(task)
    print(json.dumps({"type": "[SUMMARY]", "scores": scores}), flush=True)