import os
import json
import random
from typing import Optional, List
from pydantic import BaseModel
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "patient_triage"
SUCCESS_SCORE_THRESHOLD = 0.5

class Action(BaseModel):
    urgency: str
    department: Optional[str] = None
    action_plan: Optional[str] = None
    reasoning: Optional[str] = None

class Reward(BaseModel):
    score: float
    feedback: str

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

def grade_task1(action, complaint):
    if action.urgency == complaint["true_urgency"]:
        return Reward(score=1.0, feedback="Correct")
    both_serious = {"critical", "high"}
    if action.urgency in both_serious and complaint["true_urgency"] in both_serious:
        return Reward(score=0.4, feedback="Close")
    return Reward(score=0.0, feedback="Wrong")

def grade_task2(action, complaint):
    score = 0.0
    if action.urgency == complaint["true_urgency"]:
        score += 0.5
    if action.department == complaint["true_department"]:
        score += 0.5
    return Reward(score=round(score, 2), feedback="graded")

def grade_task3(action, complaint):
    score = 0.0
    if action.urgency == complaint["true_urgency"]:
        score += 0.3
    if action.department == complaint["true_department"]:
        score += 0.3
    valid_actions = ["immediate_admission", "urgent_consultation", "schedule_appointment"]
    if action.action_plan == complaint["true_action"]:
        score += 0.4
    elif action.action_plan in valid_actions:
        score += 0.1
    return Reward(score=round(score, 2), feedback="graded")

TASK_INFO = {
    "task_1_urgency": {"grader": grade_task1, "desc": "Classify the complaint as: critical, high, or low urgency."},
    "task_2_routing": {"grader": grade_task2, "desc": "Classify urgency AND route to correct department."},
    "task_3_full_triage": {"grader": grade_task3, "desc": "Classify urgency, route to department, AND generate action plan."},
}

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action(complaint, desc):
    if not HF_TOKEN:
        text = complaint["text"].lower()
        if any(w in text for w in ["severe", "chest pain", "allergic", "blood sugar 450"]):
            urgency = "critical"
        elif any(w in text for w in ["fever", "fell", "difficulty breathing"]):
            urgency = "high"
        else:
            urgency = "low"
        return Action(urgency=urgency, department="general", action_plan="schedule_appointment")

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
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
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0
    )
    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    return Action(**parsed)

def run_task(task_id):
    complaints = COMPLAINTS.copy()
    random.shuffle(complaints)
    grader = TASK_INFO[task_id]["grader"]
    desc = TASK_INFO[task_id]["desc"]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step, complaint in enumerate(complaints, 1):
            done = step == len(complaints)
            try:
                action = get_action(complaint, desc)
                error = None
            except Exception as e:
                action = Action(urgency="low", department="general", action_plan="schedule_appointment")
                error = str(e)

            reward_obj = grader(action, complaint)
            reward = reward_obj.score
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.urgency, reward=reward, done=done, error=error)

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

if __name__ == "__main__":
    for task in ["task_1_urgency", "task_2_routing", "task_3_full_triage"]:
        run_task(task)