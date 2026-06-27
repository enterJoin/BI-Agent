"""LLM planning helpers for controlled Agentic RAG."""

import json
from typing import Any, cast

from springgraph.rag.agent.state import (
    PlanStep,
    QuestionUnderstanding,
    RetrievalPlan,
)
from springgraph.rag.config.loader import load_intent_configs
from springgraph.rag.config.models import IntentConfig, ToolConfig
from springgraph.rag.intent import infer_query_intent, intent_default_filters
from springgraph.rag.llm import invoke_agent_model
from springgraph.rag.prompts.loader import load_prompt
from springgraph.rag.schemas import RagEvidence, SourceSnippet
from springgraph.rag.target_trace import (
    explicit_trace_target,
    persistence_edge_kinds,
)


def plan_question_retrieval(
    question: str,
    available_tools: list[ToolConfig],
    source_available: bool,
    memory_observations: list[str],
) -> tuple[QuestionUnderstanding, RetrievalPlan]:
    """Ask the LLM to understand the question and create one retrieval plan."""
    prompt = "\n\n".join(
        [
            load_prompt("plan_retrieval.md"),
            f"Question:\n{question}",
            f"Available tools:\n{_tools_json(available_tools)}",
            f"Source reading allowed by request: {source_available}",
            "Thread memory observations:\n"
            f"{json.dumps(memory_observations, ensure_ascii=False)}",
        ]
    )
    payload = _invoke_json(prompt)
    understanding_payload = payload.get("question_understanding")
    if not isinstance(understanding_payload, dict):
        understanding_payload = {}
    plan_payload = payload.get("retrieval_plan")
    if not isinstance(plan_payload, dict):
        plan_payload = {}
    understanding = _normalize_understanding(understanding_payload, question)
    plan = _normalize_plan(plan_payload, understanding, question)
    plan = apply_intent_defaults(plan, understanding, load_intent_configs())
    return understanding, plan


def _normalize_understanding(
    payload: dict[str, Any],
    fallback_question: str,
) -> QuestionUnderstanding:
    return {
        "task_goal": _string(payload.get("task_goal"), fallback_question),
        "intent": _string(payload.get("intent"), "unknown"),
        "sub_questions": _string_list(payload.get("sub_questions")),
        "business_terms": _string_list(payload.get("business_terms")),
        "technical_terms": _string_list(payload.get("technical_terms")),
        "entities": _string_list(payload.get("entities")),
        "expected_evidence": _string_list(payload.get("expected_evidence")),
    }


def _normalize_plan(
    payload: dict[str, Any],
    understanding: QuestionUnderstanding,
    fallback_question: str,
) -> RetrievalPlan:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = []
    task_goal = _string(
        payload.get("task_goal"),
        understanding.get("task_goal", fallback_question),
    )
    return {
        "task_goal": task_goal,
        "steps": [
            _normalize_step(item, fallback_question)
            for item in raw_steps
            if isinstance(item, dict)
        ],
    }


def apply_intent_defaults(
    plan: RetrievalPlan,
    understanding: QuestionUnderstanding,
    intent_configs: dict[str, IntentConfig],
) -> RetrievalPlan:
    """Apply configured intent defaults to planned tool filters."""
    steps = plan.get("steps", [])
    updated_steps = [
        _apply_step_intent_defaults(step, understanding, intent_configs)
        for step in steps
    ]
    return {**plan, "steps": updated_steps}


def _apply_step_intent_defaults(
    step: PlanStep,
    understanding: QuestionUnderstanding,
    intent_configs: dict[str, IntentConfig],
) -> PlanStep:
    filters = dict(step.get("filters", {}))
    tool_name = step.get("tool_name", "")
    explicit_intent = _string_or_none(filters.get("intent")) or _known_intent(
        understanding.get("intent"),
        intent_configs,
    )
    group_by = _string_or_none(filters.get("group_by"))
    intent = infer_query_intent(
        query=step.get("query", ""),
        explicit_intent=explicit_intent,
        group_by=group_by,
        intent_configs=intent_configs,
    )
    if intent is None:
        return {**step, "filters": filters}
    config = intent_configs.get(intent)
    if config is None or config.default_tool != tool_name:
        return {**step, "filters": filters}
    merged_filters = {
        **intent_default_filters(intent, intent_configs),
        **filters,
        "intent": intent,
    }
    trace_target = explicit_trace_target(step.get("query", ""))
    if intent == "persistence_location" and trace_target is not None:
        return {
            **step,
            "tool_name": "target_trace",
            "filters": {
                **merged_filters,
                "target": trace_target,
                "direction": "incoming",
                "edge_kinds": persistence_edge_kinds(),
            },
        }
    return {**step, "filters": merged_filters}


def build_final_prompt(
    question: str,
    understanding: QuestionUnderstanding,
    evidence: list[RagEvidence],
    source_snippets: list[SourceSnippet],
    observations: list[str],
    conversation_history: list[dict[str, str]],
    source_reading_skipped_reason: str | None,
) -> str:
    """Build the final answer prompt."""
    return "\n\n".join(
        [
            load_prompt("final_answer.md"),
            f"Current question:\n{question}",
            f"Question understanding:\n{json.dumps(understanding, ensure_ascii=False)}",
            f"Tool observations:\n{json.dumps(observations, ensure_ascii=False)}",
            f"Evidence:\n{_evidence_summary(evidence)}",
            f"Source snippets:\n{_snippet_summary(source_snippets)}",
            f"Source reading skipped reason: {source_reading_skipped_reason}",
            "Conversation history for reference only:\n"
            f"{_conversation_history_summary(conversation_history)}",
        ]
    )


def _invoke_json(prompt: str) -> dict[str, Any]:
    raw = invoke_agent_model(prompt)
    text = _strip_json_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON response must be an object.")
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


def _tools_json(tools: list[ToolConfig]) -> str:
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "capabilities": tool.capabilities,
            "requires": tool.requires,
        }
        for tool in tools
    ]
    return json.dumps(payload, ensure_ascii=False)


def _normalize_step(item: dict[str, object], fallback_query: str) -> PlanStep:
    filters = item.get("filters")
    return {
        "tool_name": _string(item.get("tool_name"), ""),
        "query": _string(item.get("query"), fallback_query),
        "filters": filters if isinstance(filters, dict) else {},
        "reason": _string(item.get("reason"), ""),
    }


def _evidence_summary(evidence: list[RagEvidence]) -> str:
    lines: list[str] = []
    for index, item in enumerate(evidence, start=1):
        location = item.file_path or "unknown"
        if item.start_line is not None:
            end_line = item.end_line or item.start_line
            location = f"{location}:{item.start_line}-{end_line}"
        details = item.content_excerpt or ""
        route_details = _route_details(item)
        if route_details:
            details = f"{details}; {route_details}" if details else route_details
        if len(details) > 280:
            details = f"{details[:277]}..."
        lines.append(
            f"{index}. [{item.source}/{item.evidence_type}] "
            f"{item.symbol or ''} at {location}; {details}"
        )
    return "\n".join(lines) if lines else "No evidence yet."


def _snippet_summary(snippets: list[SourceSnippet]) -> str:
    lines: list[str] = []
    for index, snippet in enumerate(snippets, start=1):
        lines.append(
            f"{index}. {snippet.file_path}:{snippet.start_line}-{snippet.end_line}\n"
            f"{snippet.content[:500]}"
        )
    return "\n\n".join(lines) if lines else "No source snippets."


def _conversation_history_summary(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "").strip()
        content = message.get("content", "").strip()
        if not role or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _route_details(item: RagEvidence) -> str:
    raw_metadata = item.metadata.get("metadata")
    if not isinstance(raw_metadata, dict):
        return ""
    http_method = raw_metadata.get("http_method")
    route_path = raw_metadata.get("path")
    handler = raw_metadata.get("handler")
    if not any(isinstance(value, str) and value for value in (http_method, route_path)):
        return ""
    parts = []
    if isinstance(http_method, str) and http_method:
        parts.append(f"http_method={http_method}")
    if isinstance(route_path, str) and route_path:
        parts.append(f"path={route_path}")
    if isinstance(handler, str) and handler:
        parts.append(f"handler={handler}")
    return "; ".join(parts)


def _string(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _known_intent(
    value: object,
    intent_configs: dict[str, IntentConfig],
) -> str | None:
    if not isinstance(value, str):
        return None
    intent = value.strip().lower()
    if intent not in intent_configs:
        return None
    return intent


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
