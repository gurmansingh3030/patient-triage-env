from dotenv import load_dotenv
load_dotenv()

import os
import json
from openai import OpenAI
from environment import PatientTriageEnv
from tasks import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def run_task(task_id: str):
    env = PatientTriageEnv(task_id=task_id)
    obs = env.reset()
    total_score = 0.0
    steps = 0

    print(json.dumps({"type": "[START]", "task_id": task_id}))

    while True:
        prompt = f"""You are a hospital triage assistant.

Patient Complaint: {obs.complaint_text}
Task: {obs.task_description}

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

        try:
            parsed = json.loads(raw)
            action = Action(**parsed)
        except:
            action = Action(urgency="low")

        obs_next, reward, done, info = env.step(action)
        total_score += reward.score
        steps += 1

        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "step": steps,
            "score": reward.score,
            "feedback": reward.feedback,
            "cumulative_score": info["cumulative_score"]
        }))

        if done:
            break
        obs = obs_next

    final_avg = round(total_score / steps, 3)
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "total_steps": steps,
        "final_score": final_avg
    }))
    return final_avg

if __name__ == "__main__":
    scores = {}
    for task in ["task_1_urgency", "task_2_routing", "task_3_full_triage"]:
        scores[task] = run_task(task)
    print(json.dumps({"type": "[SUMMARY]", "scores": scores}))