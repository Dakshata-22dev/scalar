from typing import Literal

from pydantic import BaseModel, Field

from env.graders import grade_classification, grade_followup, grade_reply


class Email(BaseModel):
    id: int
    subject: str
    body: str
    sender: str
    priority: Literal["low", "medium", "high"]
    category: str
    keywords: list[str]
    follow_up: bool
    read: bool = False


class InboxItem(BaseModel):
    id: int
    subject: str
    sender: str
    priority: str
    read: bool


class Observation(BaseModel):
    current_index: int
    total_emails: int
    current_email: Email | None
    inbox_summary: list[InboxItem]
    done: bool
    task: str


class Action(BaseModel):
    classification: str = Field(..., description="Predicted category label for the email.")
    reply: str = Field(..., description="Proposed reply to the sender.")
    follow_up: bool = Field(..., description="Whether the email needs follow-up.")


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


def compute_reward(action: Action, gt: Email, task: str) -> float:
    classification_score = grade_classification(action.classification, gt.category)
    reply_score = grade_reply(action.reply, gt.model_dump(), task)
    follow_up_score = grade_followup(action.follow_up, gt.follow_up)

    total = 0.4 * classification_score + 0.3 * reply_score + 0.3 * follow_up_score
    return round((total * 2.0) - 1.0, 3)
