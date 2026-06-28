"""LLM planning helpers for controlled Agentic RAG."""

import json
import re
from typing import Any, cast

from springgraph.rag.agent.state import (
    PlanStep,
    QuestionUnderstanding,
    RetrievalPlan,
)
from springgraph.rag.config.loader import (
    load_aggregation_specs,
    load_execution_trace_config,
    load_intent_configs,
)
from springgraph.rag.config.models import IntentConfig, ToolConfig
from springgraph.rag.intent import infer_query_intent, intent_default_filters
from springgraph.rag.llm import invoke_agent_model
from springgraph.rag.prompts.loader import load_prompt
from springgraph.rag.schemas import RagEvidence, SourceSnippet
from springgraph.rag.target_trace import (
    explicit_trace_target,
    persistence_edge_kinds,
)
from springgraph.rag.task_planning import (
    looks_like_task_planning,
    task_planning_default_steps,
    task_planning_intent_name,
    task_planning_prompt_guidance,
    task_planning_tool_order,
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
    if looks_like_task_planning(question):
        understanding["intent"] = task_planning_intent_name()
    plan = _normalize_plan(plan_payload, understanding, question)
    plan = apply_intent_defaults(plan, understanding, load_intent_configs())
    plan = apply_execution_trace_defaults(
        plan=plan,
        understanding=understanding,
        question=question,
        source_available=source_available,
    )
    plan = apply_entrypoint_lookup_defaults(
        plan=plan,
        understanding=understanding,
        question=question,
    )
    plan = apply_task_planning_defaults(
        plan=plan,
        understanding=understanding,
        question=question,
        source_available=source_available,
    )
    return understanding, plan


def apply_execution_trace_defaults(
    plan: RetrievalPlan,
    understanding: QuestionUnderstanding,
    question: str,
    source_available: bool,
) -> RetrievalPlan:
    """Prefer deep execution tracing for detailed flow questions."""
    if not source_available or not _looks_like_execution_trace_question(question):
        return plan
    steps = [
        step
        for step in plan.get("steps", [])
        if step.get("tool_name") != "source_read"
    ]
    if not _has_tool_step(steps, "execution_trace"):
        steps.insert(
            0,
            {
                "tool_name": "execution_trace",
                "query": question,
                "filters": {"intent": "execution_flow"},
                "reason": (
                    "Trace detailed execution steps, branches, calls, and "
                    "persistence points."
                ),
            },
        )
    if understanding.get("intent") in {"unknown", ""}:
        understanding["intent"] = "execution_flow"
    return {**plan, "steps": steps}


def apply_entrypoint_lookup_defaults(
    plan: RetrievalPlan,
    understanding: QuestionUnderstanding,
    question: str,
) -> RetrievalPlan:
    """Prefer entrypoint aggregation for questions asking which job persists data."""
    if not _looks_like_job_persistence_lookup(question):
        return plan
    steps = [
        step
        for step in plan.get("steps", [])
        if step.get("tool_name")
        not in {"target_trace", "source_read", "execution_trace"}
    ]
    vector_step: PlanStep = {
        "tool_name": "vector_search",
        "query": question,
        "filters": {"intent": "business_rule_lookup"},
        "reason": "Retrieve project library business rules before ranking jobs.",
    }
    job_step: PlanStep = {
        "tool_name": "aggregate_query",
        "query": question,
        "filters": {
            "intent": "persistence_location",
            "group_by": "job",
        },
        "reason": "Find scheduled job entrypoints related to data persistence.",
    }
    if _has_tool_step(steps, "aggregate_query"):
        steps = [
            _merge_job_lookup_filters(step, question)
            if step.get("tool_name") == "aggregate_query"
            else step
            for step in steps
        ]
    else:
        steps.insert(0, job_step)
    if not _has_tool_step(steps, "vector_search"):
        steps.insert(0, vector_step)
    understanding["intent"] = "persistence_location"
    return {**plan, "steps": steps}


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


def apply_task_planning_defaults(
    plan: RetrievalPlan,
    understanding: QuestionUnderstanding,
    question: str,
    source_available: bool,
) -> RetrievalPlan:
    """Add configured task-planning retrieval steps when needed."""
    if understanding.get("intent") != task_planning_intent_name():
        return plan

    steps = list(plan.get("steps", []))
    for default_step in task_planning_default_steps(question):
        tool_name = _string(default_step.get("tool_name"), "")
        if not tool_name:
            continue
        if tool_name == "source_read" and not source_available:
            continue
        if _has_tool_step(steps, tool_name):
            continue
        steps.append(_normalize_step(default_step, question))
    steps = _order_task_planning_steps(steps, source_available)
    return {**plan, "steps": steps}


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
            _task_planning_guidance(understanding),
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


def _has_tool_step(steps: list[PlanStep], tool_name: str) -> bool:
    return any(step.get("tool_name") == tool_name for step in steps)


def _merge_job_lookup_filters(step: PlanStep, question: str) -> PlanStep:
    return {
        **step,
        "query": question,
        "filters": {
            **dict(step.get("filters", {})),
            "intent": "persistence_location",
            "group_by": "job",
        },
    }


def _looks_like_job_persistence_lookup(question: str) -> bool:
    intent_configs = load_intent_configs()
    intent = infer_query_intent(
        query=question,
        explicit_intent=None,
        group_by=None,
        intent_configs=intent_configs,
    )
    if intent != "persistence_location":
        return False
    lowered = question.lower()
    return any(term and term.lower() in lowered for term in _job_lookup_terms())


def _job_lookup_terms() -> list[str]:
    terms: list[str] = []
    for spec in load_aggregation_specs():
        if spec.group_by != "job":
            continue
        terms.extend([spec.group_by, *spec.aliases, *spec.annotation_names])
    return terms


def _looks_like_execution_trace_question(question: str) -> bool:
    config = load_execution_trace_config()
    lowered = question.lower()
    if _has_configured_entrypoint_suffix(lowered, config.entrypoint_suffixes):
        return True
    return any(
        term.lower() in lowered
        for term in config.trigger_terms
    )


def _has_configured_entrypoint_suffix(
    lowered_question: str,
    suffixes: list[str],
) -> bool:
    for suffix in suffixes:
        pattern = rf"\b[A-Za-z_][A-Za-z0-9_]*{re.escape(suffix.lower())}\b"
        if re.search(pattern, lowered_question):
            return True
    return False


def _order_task_planning_steps(
    steps: list[PlanStep],
    source_available: bool,
) -> list[PlanStep]:
    source_steps = [
        step for step in steps if step.get("tool_name") == "source_read"
    ]
    non_source_steps = [
        step for step in steps if step.get("tool_name") != "source_read"
    ]
    if not source_available or not source_steps:
        return _order_known_task_planning_steps(non_source_steps)
    ordered_steps = _order_known_task_planning_steps(non_source_steps)
    return [*ordered_steps, source_steps[0]]


def _order_known_task_planning_steps(steps: list[PlanStep]) -> list[PlanStep]:
    tool_order = {
        tool_name: index
        for index, tool_name in enumerate(task_planning_tool_order())
        if tool_name != "source_read"
    }
    return sorted(
        steps,
        key=lambda step: (
            tool_order.get(step.get("tool_name", ""), len(tool_order)),
            steps.index(step),
        ),
    )


def _task_planning_guidance(understanding: QuestionUnderstanding) -> str:
    if understanding.get("intent") != task_planning_intent_name():
        return "Task planning guidance: not applicable."
    return task_planning_prompt_guidance()


def _evidence_summary(evidence: list[RagEvidence]) -> str:
    lines: list[str] = []
    typed_evidence_available = any(item.source != "vector" for item in evidence)
    for index, item in enumerate(evidence, start=1):
        location = item.file_path or "unknown"
        if item.start_line is not None:
            end_line = item.end_line or item.start_line
            location = f"{location}:{item.start_line}-{end_line}"
        details = item.content_excerpt or ""
        route_details = _route_details(item)
        if route_details:
            details = f"{details}; {route_details}" if details else route_details
        if item.source == "execution_trace":
            detail_limit = 1600
        elif item.source == "vector":
            detail_limit = 1400
            details = _semantic_context_details(
                details,
                typed_evidence_available=typed_evidence_available,
            )
        else:
            detail_limit = 280
        if len(details) > detail_limit:
            details = f"{details[: detail_limit - 3]}..."
        lines.append(
            f"{index}. [{item.source}/{item.evidence_type}] "
            f"{item.symbol or ''} at {location}; {details}"
        )
    return "\n".join(lines) if lines else "No evidence yet."


def _semantic_context_details(
    details: str,
    *,
    typed_evidence_available: bool,
) -> str:
    prefix = (
        "semantic_context_only=true; use this as terminology/business-rule "
        "context, not as typed artifact candidates"
    )
    if typed_evidence_available:
        return (
            f"{prefix}; details omitted because typed relational evidence is "
            "available for artifact candidates."
        )
    if not details:
        return (
            f"{prefix}."
        )
    return (
        f"{prefix}; {details}"
    )


def _snippet_summary(snippets: list[SourceSnippet]) -> str:
    lines: list[str] = []
    for index, snippet in enumerate(snippets, start=1):
        lines.append(
            f"{index}. {snippet.file_path}:{snippet.start_line}-{snippet.end_line}\n"
            f"{snippet.content[:2500]}"
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
