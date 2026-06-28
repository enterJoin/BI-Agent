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


class ContextConstraints(TypedDict, total=False):
    """Structured constraints derived from current question and context."""

    targets: list[ResolvedTarget]
    scope: str
    evidence_must_be_reachable_from_targets: bool


class QueryResolution(TypedDict, total=False):
    """Structured contextual query resolution."""

    is_follow_up: bool
    needs_context: bool
    needs_clarification: bool
    context_mode: str
    resolved_target: ResolvedTarget
    resolved_targets: list[ResolvedTarget]
    hard_constraints: ContextConstraints
    soft_context: dict[str, object]
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
        "context_mode": "new_topic",
        "resolved_target": {
            "type": "unknown",
            "name": "",
            "source": "none",
            "confidence": 0.0,
        },
        "resolved_targets": [],
        "hard_constraints": {
            "targets": [],
            "scope": "none",
            "evidence_must_be_reachable_from_targets": False,
        },
        "soft_context": {},
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
    resolved_target = _normalize_target(target_payload)
    resolved_targets = _normalize_targets(payload.get("resolved_targets"))
    if not resolved_targets and resolved_target.get("name"):
        resolved_targets = [resolved_target]
    context_mode = _context_mode(payload.get("context_mode"))
    hard_constraints = _normalize_constraints(
        payload.get("hard_constraints"),
        fallback_targets=resolved_targets,
        context_mode=context_mode,
    )
    rewritten_question = _string(payload.get("rewritten_question"), fallback_question)
    return {
        "is_follow_up": bool(payload.get("is_follow_up", False)),
        "needs_context": bool(payload.get("needs_context", False)),
        "needs_clarification": bool(payload.get("needs_clarification", False)),
        "context_mode": context_mode,
        "resolved_target": resolved_target,
        "resolved_targets": resolved_targets,
        "hard_constraints": hard_constraints,
        "soft_context": _mapping(payload.get("soft_context")),
        "rewritten_question": rewritten_question,
        "retrieval_intent": _string(payload.get("retrieval_intent"), "unknown"),
        "preferred_tools": _string_list(payload.get("preferred_tools")),
        "reason": _string(payload.get("reason"), ""),
    }


def _normalize_target(payload: dict[str, object]) -> ResolvedTarget:
    return {
        "type": _string(payload.get("type"), "unknown"),
        "name": _string(payload.get("name"), ""),
        "source": _string(payload.get("source"), "none"),
        "confidence": _float_value(payload.get("confidence"), 0.0),
    }


def _normalize_targets(value: object) -> list[ResolvedTarget]:
    if not isinstance(value, list):
        return []
    targets = []
    for item in value:
        if not isinstance(item, dict):
            continue
        target = _normalize_target(item)
        if target.get("name"):
            targets.append(target)
    return targets


def _normalize_constraints(
    value: object,
    *,
    fallback_targets: list[ResolvedTarget],
    context_mode: str,
) -> ContextConstraints:
    payload = value if isinstance(value, dict) else {}
    targets = _normalize_targets(payload.get("targets"))
    if not targets and context_mode == "object_followup":
        targets = fallback_targets
    scope = _string(payload.get("scope"), "target_call_chain")
    if context_mode != "object_followup":
        scope = "none"
        targets = []
    return {
        "targets": targets,
        "scope": scope,
        "evidence_must_be_reachable_from_targets": bool(
            targets
            and payload.get("evidence_must_be_reachable_from_targets", True)
        ),
    }


def _context_mode(value: object) -> str:
    if not isinstance(value, str):
        return "new_topic"
    mode = value.strip()
    if mode in {"object_followup", "topic_expansion", "new_topic"}:
        return mode
    return "new_topic"


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


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
