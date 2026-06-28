"""Contextual query resolution before retrieval planning."""

import json
from typing import Any, TypedDict, cast

from springgraph.rag.llm import invoke_query_resolver_model
from springgraph.rag.prompts.loader import load_prompt


class ResolvedTarget(TypedDict, total=False):
    """Target resolved from current question or recent conversation."""

    type: str
    name: str
    source: str
    confidence: float


class QueryResolution(TypedDict, total=False):
    """Structured contextual query resolution."""

    is_follow_up: bool
    needs_context: bool
    needs_clarification: bool
    resolved_target: ResolvedTarget
    rewritten_question: str
    retrieval_intent: str
    preferred_tools: list[str]
    reason: str


def resolve_contextual_query(
    question: str,
    conversation_history: list[dict[str, str]],
) -> QueryResolution:
    """Resolve follow-up references into one standalone retrieval question."""
    normalized_question = question.strip()
    if not conversation_history:
        return _default_resolution(normalized_question)
    prompt = "\n\n".join(
        [
            load_prompt("resolve_contextual_query.md"),
            f"Current question:\n{normalized_question}",
            "Recent conversation:\n"
            f"{_conversation_history_summary(conversation_history)}",
        ]
    )
    payload = _invoke_json(prompt)
    return _normalize_resolution(payload, normalized_question)


def _default_resolution(question: str) -> QueryResolution:
    return {
        "is_follow_up": False,
        "needs_context": False,
        "needs_clarification": False,
        "resolved_target": {
            "type": "unknown",
            "name": "",
            "source": "none",
            "confidence": 0.0,
        },
        "rewritten_question": question,
        "retrieval_intent": "unknown",
        "preferred_tools": [],
        "reason": "No recent conversation was available.",
    }


def _normalize_resolution(
    payload: dict[str, Any],
    fallback_question: str,
) -> QueryResolution:
    raw_target = payload.get("resolved_target")
    target_payload = raw_target if isinstance(raw_target, dict) else {}
    rewritten_question = _string(payload.get("rewritten_question"), fallback_question)
    return {
        "is_follow_up": bool(payload.get("is_follow_up", False)),
        "needs_context": bool(payload.get("needs_context", False)),
        "needs_clarification": bool(payload.get("needs_clarification", False)),
        "resolved_target": {
            "type": _string(target_payload.get("type"), "unknown"),
            "name": _string(target_payload.get("name"), ""),
            "source": _string(target_payload.get("source"), "none"),
            "confidence": _float_value(target_payload.get("confidence"), 0.0),
        },
        "rewritten_question": rewritten_question,
        "retrieval_intent": _string(payload.get("retrieval_intent"), "unknown"),
        "preferred_tools": _string_list(payload.get("preferred_tools")),
        "reason": _string(payload.get("reason"), ""),
    }


def _conversation_history_summary(messages: list[dict[str, str]]) -> str:
    relevant = messages[-6:]
    if not relevant:
        return "No prior conversation."
    lines: list[str] = []
    for message in relevant:
        role = message.get("role", "").strip()
        content = message.get("content", "").strip()
        if not role or not content:
            continue
        if len(content) > 1000:
            content = f"{content[:997]}..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _invoke_json(prompt: str) -> dict[str, Any]:
    raw = invoke_query_resolver_model(prompt)
    text = _strip_json_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Query resolver did not return valid JSON: {raw[:500]}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Query resolver JSON response must be an object.")
    return cast(dict[str, Any], payload)


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _string(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _float_value(value: object, fallback: float) -> float:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
