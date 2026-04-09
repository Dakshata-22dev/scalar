import asyncio
import json
import os
from typing import Any

from openai import OpenAI


API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "email-triage-env"
TASK_NAMES = ["easy", "medium", "hard"]
TEMPERATURE = 0.2
MAX_TOKENS = 250
MAX_STEPS = 12
MAX_TOTAL_REWARD = 5.0
SUCCESS_SCORE_THRESHOLD = 0.6
IMAGE_NAME = LOCAL_IMAGE_NAME or "email-triage-env:latest"
MIN_SCORE_EPSILON = 0.001
MAX_SCORE_EPSILON = 0.999

SYSTEM_PROMPT = """You are an email triage agent.
For each email, choose the best category, draft a concise professional reply, and decide whether follow-up is required.
Return only valid JSON with keys: classification, reply, follow_up.
The follow_up value must be a boolean.
"""

FALLBACK_ACTION = {
    "classification": "work",
    "reply": "Thanks, I received your email and will review it shortly.",
    "follow_up": False,
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={json.dumps(task)} env={json.dumps(env)} model={json.dumps(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = "null" if error is None else json.dumps(error)
    print(
        f"[STEP] step={step} action={json.dumps(action)} reward={reward:.3f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={json.dumps(rewards)}",
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


def build_user_prompt(task_name: str, step: int, observation: Any, last_reward: float, history: list[str]) -> str:
    payload = {
        "task": getattr(observation, "task", task_name),
        "step": step,
        "last_reward": last_reward,
        "current_email": _safe_dump(getattr(observation, "current_email", None)),
        "inbox_summary": _safe_dump(getattr(observation, "inbox_summary", [])),
        "history": history,
    }
    return json.dumps(payload, ensure_ascii=True)


def parse_action(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return FALLBACK_ACTION.copy()

    classification = str(payload.get("classification", FALLBACK_ACTION["classification"])).strip()
    reply = str(payload.get("reply", FALLBACK_ACTION["reply"])).strip()
    follow_up = bool(payload.get("follow_up", FALLBACK_ACTION["follow_up"]))

    return {
        "classification": classification or FALLBACK_ACTION["classification"],
        "reply": reply or FALLBACK_ACTION["reply"],
        "follow_up": follow_up,
    }


def warmup_model(client: OpenAI) -> None:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Respond with empty JSON."},
            {"role": "user", "content": "{}"},
        ],
        temperature=0.0,
        max_tokens=8,
        stream=False,
    )
    _ = completion.choices[0].message.content


def get_model_action(client: OpenAI, task_name: str, step: int, observation: Any, last_reward: float, history: list[str]) -> dict[str, Any]:
    user_prompt = build_user_prompt(task_name, step, observation, last_reward, history)

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
        return FALLBACK_ACTION.copy()


def clamp_score(raw_score: float) -> float:
    if raw_score <= 0.0:
        return MIN_SCORE_EPSILON
    if raw_score >= 1.0:
        return MAX_SCORE_EPSILON
    return raw_score


async def run_task(env: Any, client: OpenAI, task_name: str) -> tuple[bool, int, float, list[float]]:
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = MIN_SCORE_EPSILON
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            action_payload = get_model_action(client, task_name, step, result.observation, last_reward, history)
            error = None

            try:
                result = await env.step(action_payload)
                reward = float(getattr(result, "reward", 0.0) or 0.0)
                done = bool(getattr(result, "done", False))
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)
                print(f"[DEBUG] env.step() failed: {exc}", flush=True)

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

        raw_score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else MIN_SCORE_EPSILON
        score = clamp_score(raw_score)
        success = score >= SUCCESS_SCORE_THRESHOLD
        return success, steps_taken, score, rewards

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    env = None

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        warmup_model(client)

        from openenv import OpenEnv

        env = await OpenEnv.from_docker_image(IMAGE_NAME)

        for task_name in TASK_NAMES:
            await run_task(env, client, task_name)

    except Exception as exc:
        print(f"[DEBUG] inference main() failed: {exc}", flush=True)
        for task_name in TASK_NAMES:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=MIN_SCORE_EPSILON, rewards=[])
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())