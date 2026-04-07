from fastapi import FastAPI
from environment import PatientTriageEnv
from tasks import Action

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "name": "Patient Triage Environment"}

@app.post("/reset")
def reset(task_id: str = "task_1_urgency"):
    env = PatientTriageEnv(task_id=task_id)
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: Action, task_id: str = "task_1_urgency"):
    env = PatientTriageEnv(task_id=task_id)
    env.reset()
    obs, reward, done, info = env.step(action)
    return {"reward": reward.dict(), "done": done, "info": info}

@app.get("/state")
def state(task_id: str = "task_1_urgency"):
    env = PatientTriageEnv(task_id=task_id)
    return env.state().dict()