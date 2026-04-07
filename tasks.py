from pydantic import BaseModel
from typing import Optional
from data import COMPLAINTS

# ─── Pydantic Models ───────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    complaint_id: str
    complaint_text: str
    task_description: str

class Action(BaseModel):
    urgency: str                          # critical / high / low
    department: Optional[str] = None      # needed for medium + hard
    action_plan: Optional[str] = None     # needed for hard only
    reasoning: Optional[str] = None

class Reward(BaseModel):
    score: float                          # 0.0 to 1.0
    feedback: str

# ─── Task 1: Easy — Urgency Classification ─────────────────────

class Task1Grader:
    task_id = "task_1_urgency"
    description = "Classify the complaint as: critical, high, or low urgency."

    def grade(self, action: Action, complaint: dict) -> Reward:
        if action.urgency == complaint["true_urgency"]:
            return Reward(score=1.0, feedback="Correct urgency classification!")
        
        # Partial credit — critical/high mix gets 0.4
        both_serious = {"critical", "high"}
        if action.urgency in both_serious and complaint["true_urgency"] in both_serious:
            return Reward(score=0.4, feedback="Close — both are serious but urgency level wrong.")
        
        return Reward(score=0.0, feedback=f"Wrong. Expected {complaint['true_urgency']}.")

# ─── Task 2: Medium — Urgency + Department Routing ─────────────

class Task2Grader:
    task_id = "task_2_routing"
    description = "Classify urgency AND route to correct department."

    def grade(self, action: Action, complaint: dict) -> Reward:
        score = 0.0
        feedback_parts = []

        # Urgency = 50% weight
        if action.urgency == complaint["true_urgency"]:
            score += 0.5
            feedback_parts.append("Urgency correct (+0.5)")
        else:
            feedback_parts.append(f"Urgency wrong (expected {complaint['true_urgency']})")

        # Department = 50% weight
        if action.department == complaint["true_department"]:
            score += 0.5
            feedback_parts.append("Department correct (+0.5)")
        else:
            feedback_parts.append(f"Department wrong (expected {complaint['true_department']})")

        return Reward(score=round(score, 2), feedback=" | ".join(feedback_parts))

# ─── Task 3: Hard — Full Triage with Action Plan ───────────────

class Task3Grader:
    task_id = "task_3_full_triage"
    description = "Classify urgency, route to department, AND generate action plan."

    VALID_ACTIONS = ["immediate_admission", "urgent_consultation", "schedule_appointment"]

    def grade(self, action: Action, complaint: dict) -> Reward:
        score = 0.0
        feedback_parts = []

        # Urgency = 30%
        if action.urgency == complaint["true_urgency"]:
            score += 0.3
            feedback_parts.append("Urgency correct (+0.3)")
        else:
            feedback_parts.append(f"Urgency wrong (expected {complaint['true_urgency']})")

        # Department = 30%
        if action.department == complaint["true_department"]:
            score += 0.3
            feedback_parts.append("Department correct (+0.3)")
        else:
            feedback_parts.append(f"Department wrong (expected {complaint['true_department']})")

        # Action plan = 40%
        if action.action_plan == complaint["true_action"]:
            score += 0.4
            feedback_parts.append("Action plan correct (+0.4)")
        elif action.action_plan in self.VALID_ACTIONS:
            score += 0.1  # partial — valid action but wrong one
            feedback_parts.append(f"Valid but wrong action (expected {complaint['true_action']}) (+0.1)")
        else:
            feedback_parts.append(f"Invalid/missing action plan (expected {complaint['true_action']})")

        return Reward(score=round(score, 2), feedback=" | ".join(feedback_parts))


TASKS = {
    "task_1_urgency": Task1Grader(),
    "task_2_routing": Task2Grader(),
    "task_3_full_triage": Task3Grader(),
}