from env.models import Action, Email, compute_reward


def score_action(action: Action, ground_truth: Email, task: str) -> float:
    return compute_reward(action, ground_truth, task)
