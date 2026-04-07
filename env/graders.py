from typing import Any


def grade_classification(predicted: str, expected: str) -> float:
    if not predicted:
        return 0.0
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0


def grade_reply(reply: str, email: dict[str, Any], task: str) -> float:
    if not reply or not reply.strip():
        return 0.0

    reply_lower = reply.lower()
    sender_name = email["sender"].split("@")[0].replace(".", " ")
    keyword_hits = sum(1 for keyword in email["keywords"] if keyword.lower() in reply_lower)

    score = 0.2
    if any(token in reply_lower for token in ["thanks", "thank you", "received", "noted"]):
        score += 0.2
    if email["category"].lower() in reply_lower or keyword_hits > 0:
        score += 0.3
    if sender_name.split()[0] in reply_lower:
        score += 0.1
    if task == "hard" and any(token in reply_lower for token in ["timeline", "next steps", "priority"]):
        score += 0.2
    elif task in {"easy", "medium"} and len(reply.split()) >= 6:
        score += 0.2

    return min(score, 1.0)


def grade_followup(predicted: bool, expected: bool) -> float:
    return 1.0 if bool(predicted) is bool(expected) else 0.0
