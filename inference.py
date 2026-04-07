import asyncio
import json
import os
from typing import Any

from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "email-triage-env"
TASK_NAME = "easy"
TEMPERATURE = 0.2
MAX_TOKENS = 250
MAX_STEPS = 12
MAX_TOTAL_REWARD = 5.0
SUCCESS_SCORE_THRESHOLD = 0.6
IMAGE_NAME = LOCAL_IMAGE_NAME or "email-triage-env:latest"

SYSTEM_PROMPT = """You are an email triage agent.
For each email, choose the best category, draft a concise professional reply, and decide whether follow-up is required.
Return only valid JSON with keys: classification, reply, follow_up.
The follow_up value must be a boolean.
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"START task={json.dumps(task)} env={json.dumps(env)} model={json.dumps(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = "null" if error is None else json.dumps(error)
    print(
        f"STEP step={step} action={json.dumps(action)} reward={reward:.3f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        f"END success={str(success).lower()} steps={steps} score={score:.3f} rewards={json.dumps(rewards)}",
        flush=True,
    )


def _safe_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_safe_dump(item) for item in value]
    return value


def build_user_prompt(step: int, observation: Any, last_reward: float, history: list[str]) -> str:
    payload = {
        "task": getattr(observation, "task", TASK_NAME),
        "step": step,
        "last_reward": last_reward,
        "current_email": _safe_dump(getattr(observation, "current_email", None)),
        "inbox_summary": _safe_dump(getattr(observation, "inbox_summary", [])),
        "history": history,
    }
    return json.dumps(payload, ensure_ascii=True)


def parse_action(text: str) -> dict[str, Any]:
    fallback = {
        "classification": "work",
        "reply": "Thanks, I received your email and will review it shortly.",
        "follow_up": False,
    }

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return fallback

    classification = str(payload.get("classification", fallback["classification"])).strip() or fallback["classification"]
    reply = str(payload.get("reply", fallback["reply"])).strip() or fallback["reply"]
    follow_up = bool(payload.get("follow_up", fallback["follow_up"]))

    return {
        "classification": classification,
        "reply": reply,
        "follow_up": follow_up,
    }


def get_model_action(client: OpenAI, step: int, observation: Any, last_reward: float, history: list[str]) -> dict[str, Any]:
    user_prompt = build_user_prompt(step, observation, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text if text else "{}")
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return parse_action("{}")


async def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set for inference.py.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    from openenv import OpenEnv

    env = await OpenEnv.from_docker_image(IMAGE_NAME)

    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=TASK_NAME)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_payload = get_model_action(client, step, result.observation, last_reward, history)

            result = await env.step(action_payload)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=json.dumps(action_payload, ensure_ascii=True),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"Step {step}: {json.dumps(action_payload, ensure_ascii=True)} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())