from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.environment import EmailTriageEnv
from env.models import Action, Observation, StepResult


class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"] = Field(
        ..., description="Task difficulty to load."
    )


class ResetResponse(BaseModel):
    observation: Observation
    task: str


class StateResponse(BaseModel):
    observation: Observation


app = FastAPI(title="Email Triage AI Environment", version="1.0.0")
env = EmailTriageEnv()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Email triage environment is running."}


@app.post("/reset", response_model=ResetResponse)
def reset_environment(request: ResetRequest) -> ResetResponse:
    try:
        observation = env.reset(request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(observation=observation, task=request.task)


@app.post("/step", response_model=StepResult)
def step_environment(action: Action) -> StepResult:
    try:
        return env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def get_state() -> StateResponse:
    return StateResponse(observation=env.state())
