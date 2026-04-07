import json
from pathlib import Path

from env.models import Action, Email, InboxItem, Observation, StepResult
from env.reward import score_action
from env.tasks import TASKS


class EmailTriageEnv:
    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_path = self.base_dir / "data" / "emails.json"
        self.task: str | None = None
        self.emails: list[Email] = []
        self.current_index = 0
        self.done = True

    def reset(self, task: str) -> Observation:
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASKS)}.")

        all_emails = self._load_emails()
        selected_indices = TASKS[task]

        try:
            self.emails = [all_emails[index] for index in selected_indices]
        except IndexError as exc:
            raise ValueError("Task configuration references an invalid email index.") from exc

        self.task = task
        self.current_index = 0
        self.done = len(self.emails) == 0
        return self._get_obs()

    def step(self, action: Action) -> StepResult:
        if self.task is None or not self.emails:
            raise ValueError("Environment is not initialized. Call /reset first.")
        if self.done:
            raise ValueError("Episode is already complete. Call /reset to start again.")

        current_email = self.emails[self.current_index]
        reward = score_action(action, current_email, self.task)
        current_email.read = True

        self.current_index += 1
        self.done = self.current_index >= len(self.emails)
        observation = self._get_obs()

        info = {
            "email_id": current_email.id,
            "expected_category": current_email.category,
            "expected_follow_up": current_email.follow_up,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            done=self.done,
            info=info,
        )

    def state(self) -> Observation:
        return self._get_obs()

    def _get_obs(self) -> Observation:
        current_email = None if self.done or not self.emails else self.emails[self.current_index]
        inbox_summary = [
            InboxItem(
                id=email.id,
                subject=email.subject,
                sender=email.sender,
                priority=email.priority,
                read=email.read,
            )
            for email in self.emails
        ]
        return Observation(
            current_index=self.current_index,
            total_emails=len(self.emails),
            current_email=current_email,
            inbox_summary=inbox_summary,
            done=self.done,
            task=self.task or "uninitialized",
        )

    def _load_emails(self) -> list[Email]:
        with self.data_path.open("r", encoding="utf-8") as handle:
            raw_emails = json.load(handle)
        return [Email(**email) for email in raw_emails]
