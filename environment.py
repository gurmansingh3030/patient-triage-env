from pydantic import BaseModel
from typing import Optional
import random
from data import COMPLAINTS
from tasks import Observation, Action, Reward, TASKS

class EnvironmentState(BaseModel):
    current_task_id: str
    current_complaint_index: int
    total_complaints: int
    cumulative_score: float
    steps_taken: int
    done: bool

class PatientTriageEnv:
    def __init__(self, task_id: str = "task_1_urgency"):
        self.task_id = task_id
        self.grader = TASKS[task_id]
        self.complaints = COMPLAINTS.copy()
        random.shuffle(self.complaints)
        self.current_index = 0
        self.cumulative_score = 0.0
        self.steps_taken = 0
        self.done = False

    def reset(self) -> Observation:
        self.complaints = COMPLAINTS.copy()
        random.shuffle(self.complaints)
        self.current_index = 0
        self.cumulative_score = 0.0
        self.steps_taken = 0
        self.done = False
        return self._get_observation()

    def step(self, action: Action):
        if self.done:
            raise ValueError("Episode done. Call reset() first.")
        complaint = self.complaints[self.current_index]
        reward = self.grader.grade(action, complaint)
        self.cumulative_score += reward.score
        self.steps_taken += 1
        self.current_index += 1
        if self.current_index >= len(self.complaints):
            self.done = True
        info = {
            "complaint_id": complaint["id"],
            "steps_taken": self.steps_taken,
            "cumulative_score": round(self.cumulative_score, 3),
            "average_score": round(self.cumulative_score / self.steps_taken, 3)
        }
        next_obs = self._get_observation() if not self.done else None
        return next_obs, reward, self.done, info

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            current_task_id=self.task_id,
            current_complaint_index=self.current_index,
            total_complaints=len(self.complaints),
            cumulative_score=round(self.cumulative_score, 3),
            steps_taken=self.steps_taken,
            done=self.done
        )

    def _get_observation(self) -> Observation:
        complaint = self.complaints[self.current_index]
        return Observation(
            task_id=self.task_id,
            complaint_id=complaint["id"],
            complaint_text=complaint["text"],
            task_description=self.grader.description
        )