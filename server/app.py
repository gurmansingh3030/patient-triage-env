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
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)