from pydantic import BaseModel
from typing import Optional
from data import COMPLAINTS

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

class Task1Grader:
    task_id = "task_1_urgency"
    description = "Classify the complaint as: critical, high, or low urgency."

    def grade(self, action: Action, complaint: dict) -> Reward:
        if action.urgency == complaint["true_urgency"]:
            return Reward(score=1.0, feedback="Correct urgency classification!")
        both_serious = {"critical", "high"}
        if action.urgency in both_serious and complaint["true_urgency"] in both_serious:
            return Reward(score=0.4, feedback="Close — both serious but wrong level.")
        return Reward(score=0.0, feedback=f"Wrong. Expected {complaint['true_urgency']}.")

class Task2Grader:
    task_id = "task_2_routing"
    description = "Classify urgency AND route to correct department."

    def grade(self, action: Action, complaint: dict) -> Reward:
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

class Task3Grader:
    task_id = "task_3_full_triage"
    description = "Classify urgency, route to department, AND generate action plan."

    VALID_ACTIONS = ["immediate_admission", "urgent_consultation", "schedule_appointment"]

    def grade(self, action: Action, complaint: dict) -> Reward:
        score = 0.0
        parts = []
        if action.urgency == complaint["true_urgency"]:
            score += 0.3
            parts.append("Urgency correct (+0.3)")
        else:
            parts.append(f"Urgency wrong (expected {complaint['true_urgency']})")
        if action.department == complaint["true_department"]:
            score += 0.3
            parts.append("Department correct (+0.3)")
        else:
            parts.append(f"Department wrong (expected {complaint['true_department']})")
        if action.action_plan == complaint["true_action"]:
            score += 0.4
            parts.append("Action plan correct (+0.4)")
        elif action.action_plan in self.VALID_ACTIONS:
            score += 0.15
            parts.append(f"Valid but wrong action (+0.15)")
        else:
            parts.append(f"Invalid action (expected {complaint['true_action']})")
        return Reward(score=round(score, 2), feedback=" | ".join(parts))

class Task4Grader:
    task_id = "task_4_reasoning"
    description = "Full triage with reasoning — classify urgency, department, action plan, AND explain why."

    VALID_ACTIONS = ["immediate_admission", "urgent_consultation", "schedule_appointment"]

    def grade(self, action: Action, complaint: dict) -> Reward:
        score = 0.0
        parts = []
        if action.urgency == complaint["true_urgency"]:
            score += 0.25
            parts.append("Urgency correct (+0.25)")
        else:
            parts.append(f"Urgency wrong (expected {complaint['true_urgency']})")
        if action.department == complaint["true_department"]:
            score += 0.25
            parts.append("Department correct (+0.25)")
        else:
            parts.append(f"Department wrong (expected {complaint['true_department']})")
        if action.action_plan == complaint["true_action"]:
            score += 0.25
            parts.append("Action plan correct (+0.25)")
        elif action.action_plan in self.VALID_ACTIONS:
            score += 0.1
            parts.append(f"Valid but wrong action (+0.1)")
        else:
            parts.append(f"Invalid action")
        if action.reasoning and len(action.reasoning) > 20:
            score += 0.25
            parts.append("Reasoning provided (+0.25)")
        else:
            parts.append("Missing/short reasoning (0.0)")
        return Reward(score=round(score, 2), feedback=" | ".join(parts))

TASKS = {
    "task_1_urgency": Task1Grader(),
    "task_2_routing": Task2Grader(),
    "task_3_full_triage": Task3Grader(),
    "task_4_reasoning": Task4Grader(),
}