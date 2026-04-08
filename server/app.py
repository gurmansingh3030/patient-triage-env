from fastapi import FastAPI
from environment import PatientTriageEnv
from tasks import Action
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "name": "Patient Triage Environment"}

@app.post("/reset")
def reset(task_id: str = "task_1_urgency"):
    env = PatientTriageEnv(task_id=task_id)
    obs = env.reset()
    return obs.model_dump()

@app.get("/state")
def state(task_id: str = "task_1_urgency"):
    env = PatientTriageEnv(task_id=task_id)
    return env.state().model_dump()

@app.post("/step")
def step(action: Action, task_id: str = "task_1_urgency"):
    env = PatientTriageEnv(task_id=task_id)
    env.reset()
    obs_next, reward, done, info = env.step(action)
    return {"reward": reward.model_dump(), "done": done, "info": info}

def main(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()