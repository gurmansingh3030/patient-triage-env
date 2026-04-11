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
    {"id": "C001", "text": "Patient has severe chest pain radiating to left arm since 2 hours. Sweating profusely.", "true_urgency": "critical", "true_department": "cardiology", "true_action": "immediate_admission"},
    {"id": "C002", "text": "Severe allergic reaction, face swelling, difficulty breathing after eating peanuts.", "true_urgency": "critical", "true_department": "emergency", "true_action": "immediate_admission"},
    {"id": "C003", "text": "Diabetic patient reports blood sugar 450, feeling dizzy and confused.", "true_urgency": "critical", "true_department": "endocrinology", "true_action": "immediate_admission"},
    {"id": "C004", "text": "Patient unconscious, brought by family, suspected drug overdose.", "true_urgency": "critical", "true_department": "emergency", "true_action": "immediate_admission"},
    {"id": "C005", "text": "Pregnant woman, 8 months, heavy bleeding and severe abdominal pain.", "true_urgency": "critical", "true_department": "emergency", "true_action": "immediate_admission"},
    {"id": "C006", "text": "Patient having seizures, uncontrolled for 5 minutes.", "true_urgency": "critical", "true_department": "emergency", "true_action": "immediate_admission"},
    {"id": "C007", "text": "Burn victim, 40% body surface area burned, in severe pain.", "true_urgency": "critical", "true_department": "emergency", "true_action": "immediate_admission"},
    {"id": "C008", "text": "Patient with knife stab wound to abdomen, actively bleeding.", "true_urgency": "critical", "true_department": "emergency", "true_action": "immediate_admission"},
    {"id": "C009", "text": "Child with high fever 104F, difficulty breathing, not eating since yesterday.", "true_urgency": "high", "true_department": "pediatrics", "true_action": "urgent_consultation"},
    {"id": "C010", "text": "Elderly patient fell down, complaining of hip pain, cannot stand.", "true_urgency": "high", "true_department": "orthopedics", "true_action": "urgent_consultation"},
    {"id": "C011", "text": "Patient with sudden vision loss in left eye since 1 hour.", "true_urgency": "high", "true_department": "emergency", "true_action": "urgent_consultation"},
    {"id": "C012", "text": "Asthma patient, inhaler not working, breathing becoming difficult.", "true_urgency": "high", "true_department": "general", "true_action": "urgent_consultation"},
    {"id": "C013", "text": "Patient with severe vomiting and diarrhea for 24 hours, signs of dehydration.", "true_urgency": "high", "true_department": "general", "true_action": "urgent_consultation"},
    {"id": "C014", "text": "Child swallowed unknown tablets, parents unsure how many.", "true_urgency": "high", "true_department": "pediatrics", "true_action": "urgent_consultation"},
    {"id": "C015", "text": "Patient with broken arm after road accident, bone visible.", "true_urgency": "high", "true_department": "orthopedics", "true_action": "urgent_consultation"},
    {"id": "C016", "text": "Sudden severe headache, worst of patient life, started 30 min ago.", "true_urgency": "high", "true_department": "emergency", "true_action": "urgent_consultation"},
    {"id": "C017", "text": "Mild headache for 3 days, no fever. Patient wants routine checkup.", "true_urgency": "low", "true_department": "general", "true_action": "schedule_appointment"},
    {"id": "C018", "text": "Patient requesting prescription refill for blood pressure medication.", "true_urgency": "low", "true_department": "general", "true_action": "schedule_appointment"},
    {"id": "C019", "text": "Patient has mild skin rash since 2 days, no other symptoms.", "true_urgency": "low", "true_department": "dermatology", "true_action": "schedule_appointment"},
    {"id": "C020", "text": "Patient wants annual health checkup and blood tests.", "true_urgency": "low", "true_department": "general", "true_action": "schedule_appointment"},
    {"id": "C021", "text": "Mild lower back pain for 1 week, no trauma, able to walk normally.", "true_urgency": "low", "true_department": "orthopedics", "true_action": "schedule_appointment"},
    {"id": "C022", "text": "Patient has dandruff and itchy scalp for 2 weeks.", "true_urgency": "low", "true_department": "dermatology", "true_action": "schedule_appointment"},
    {"id": "C023", "text": "Child due for routine vaccination appointment.", "true_urgency": "low", "true_department": "pediatrics", "true_action": "schedule_appointment"},
    {"id": "C024", "text": "Patient complains of mild acne on face, wants treatment advice.", "true_urgency": "low", "true_department": "dermatology", "true_action": "schedule_appointment"},
    {"id": "C025", "text": "Follow up appointment for diabetes management, stable condition.", "true_urgency": "low", "true_department": "endocrinology", "true_action": "schedule_appointment"},
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
    valid_actions = ["immediate_admission", "urgent_consultation", "schedule_appointment"]
    if action.urgency == complaint["true_urgency"]:
        score += 0.3
    if action.department == complaint["true_department"]:
        score += 0.3
    if action.action_plan == complaint["true_action"]:
        score += 0.4
    elif action.action_plan in valid_actions:
        score += 0.15
    return Reward(score=round(score, 2), feedback="graded")

def grade_task4(action, complaint):
    score = 0.0
    valid_actions = ["immediate_admission", "urgent_consultation", "schedule_appointment"]
    if action.urgency == complaint["true_urgency"]:
        score += 0.25
    if action.department == complaint["true_department"]:
        score += 0.25
    if action.action_plan == complaint["true_action"]:
        score += 0.25
    elif action.action_plan in valid_actions:
        score += 0.1
    if action.reasoning and len(action.reasoning) > 20:
        score += 0.25
    return Reward(score=round(score, 2), feedback="graded")

TASK_INFO = {
    "task_1_urgency": {"grader": grade_task1, "desc": "Classify the complaint as: critical, high, or low urgency."},
    "task_2_routing": {"grader": grade_task2, "desc": "Classify urgency AND route to correct department."},
    "task_3_full_triage": {"grader": grade_task3, "desc": "Classify urgency, route to department, AND generate action plan."},
    "task_4_reasoning": {"grader": grade_task4, "desc": "Full triage with urgency, department, action plan AND detailed reasoning why."},
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
        if any(w in text for w in ["severe", "chest pain", "allergic", "blood sugar 450", "unconscious", "seizure", "burn", "stab", "bleeding", "pregnant"]):
            urgency = "critical"
        elif any(w in text for w in ["fever", "fell", "difficulty breathing", "vision loss", "broken", "headache", "vomiting", "swallowed"]):
            urgency = "high"
        else:
            urgency = "low"
        return Action(urgency=urgency, department="general", action_plan="schedule_appointment", reasoning="rule-based fallback")

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    prompt = f"""You are a hospital triage assistant.
Patient Complaint: {complaint['text']}
Task: {desc}
Respond ONLY in this JSON format:
{{
  "urgency": "critical" or "high" or "low",
  "department": "cardiology" or "pediatrics" or "orthopedics" or "emergency" or "general" or "dermatology" or "endocrinology",
  "action_plan": "immediate_admission" or "urgent_consultation" or "schedule_appointment",
  "reasoning": "brief medical reason for this decision"
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
                action = Action(urgency="low", department="general", action_plan="schedule_appointment", reasoning="fallback")
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
    for task in ["task_1_urgency", "task_2_routing", "task_3_full_triage", "task_4_reasoning"]:
        run_task(task)