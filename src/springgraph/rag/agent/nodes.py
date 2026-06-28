"""Node implementations for the planned Agentic RAG graph."""

import json
from pathlib import Path

from springgraph.rag.agent import planner
from springgraph.rag.agent.state import AgenticRagState, PlanStep
from springgraph.rag.config.loader import load_agentic_rag_config
from springgraph.rag.llm import invoke_agent_model
from springgraph.rag.memory.store import get_thread, update_thread
from springgraph.rag.schemas import RagEvidence, SourceSnippet
from springgraph.rag.target_trace import source_priority
from springgraph.rag.task_planning import task_planning_source_priority
from springgraph.rag.tools.registry import load_tool_registry
from springgraph.rag.tools.schemas import ToolInput


def load_runtime_config(state: AgenticRagState) -> AgenticRagState:
    """Load runtime config and tool metadata."""
    config = load_agentic_rag_config()
    registry = load_tool_registry()
    state["runtime_config"] = config
    state["tool_configs"] = registry.configs()
    state["tool_results"] = []
    state["used_tools"] = []
    state["observations"] = []
    state["evidence"] = []
    state["source_snippets"] = []
    state["conversation_history"] = []
    state["warnings"] = list(state.get("warnings", []))
    return state


def apply_request_defaults(state: AgenticRagState) -> AgenticRagState:
    """Apply request defaults from external config when the API omitted them."""
    config = state["runtime_config"]
    top_k = state.get("top_k", config.retrieval.default_top_k)
    graph_depth = state.get("graph_depth", config.retrieval.default_graph_depth)
    state["top_k"] = max(1, min(top_k, config.retrieval.max_top_k))
    state["graph_depth"] = max(0, min(graph_depth, config.retrieval.max_graph_depth))
    state["read_source"] = state.get(
        "read_source",
        config.source_reading.enabled_by_default,
    )
    state["source_available"] = bool(state["read_source"])
    state["source_reading_skipped_reason"] = (
        None if state["source_available"] else "read_source_disabled"
    )
    return state


def load_thread_memory(state: AgenticRagState) -> AgenticRagState:
    """Load minimal thread state."""
    memory_config = state["runtime_config"].memory
    if not memory_config.short_term_enabled:
        return state
    if not state.get("load_memory", False):
        return state
    memory = get_thread(
        state["thread_id"],
        project_id=state.get("project_id"),
        user_id=state.get("user_id"),
    )
    if not state.get("project_path") and memory.project_path is not None:
        state["project_path"] = memory.project_path
    state["conversation_history"] = _bounded_conversation_history(
        memory.messages,
        max_messages=memory_config.max_history_messages,
        max_chars=memory_config.max_history_chars,
    )
    if memory.last_question is not None:
        state["observations"].append(
            "Thread memory is available for follow-up context."
        )
    return state


def plan_retrieval(state: AgenticRagState) -> AgenticRagState:
    """Create question understanding and one structured retrieval plan."""
    # TODO: Make question understanding context-aware by passing concise thread
    # memory, recent evidence, recent symbols, and project library hints.
    understanding, retrieval_plan = planner.plan_question_retrieval(
        question=state["question"],
        available_tools=state["tool_configs"],
        source_available=state.get("source_available", False),
        memory_observations=state.get("observations", []),
    )
    state["question_understanding"] = understanding
    state["retrieval_plan"] = retrieval_plan
    return state


def execute_retrieval_plan(state: AgenticRagState) -> AgenticRagState:
    """Execute planned tool steps once, in order."""
    registry = load_tool_registry()
    steps = state.get("retrieval_plan", {}).get("steps", [])
    executed_source_read = False
    executed_signatures: set[str] = set()
    for step in steps:
        tool_name = step.get("tool_name", "")
        if not tool_name:
            continue
        signature = _step_signature(step)
        if signature in executed_signatures:
            state["observations"].append(
                f"Skipped duplicate planned tool step: {tool_name}"
            )
            continue
        executed_signatures.add(signature)
        if tool_name == "source_read":
            if executed_source_read:
                continue
            executed_source_read = True
            if not _has_file_evidence(state.get("evidence", [])):
                state["observations"].append(
                    "source_read skipped because no file evidence was available."
                )
                continue
        tool = registry.get(tool_name)
        if tool is None:
            state["warnings"].append(f"Unknown planned tool skipped: {tool_name}")
            continue
        evidence = _source_read_evidence_for_step(state, tool_name)
        result = tool.invoke(_tool_input(state, step, evidence=evidence))
        state["tool_results"].append(result)
        state["used_tools"].append(result.tool_name)
        state["observations"].append(result.summary)
        state["warnings"] = [*state.get("warnings", []), *result.warnings]
        state["evidence"] = _dedupe_evidence(
            [*state.get("evidence", []), *result.evidence]
        )
        state["source_snippets"] = _dedupe_source_snippets(
            [*state.get("source_snippets", []), *result.source_snippets]
        )
        if tool_name == "source_read" and not result.source_snippets:
            state["source_reading_skipped_reason"] = _source_skip_reason(
                result.warnings
            )

    if not executed_source_read and _should_auto_source_read(state):
        tool = registry.get("source_read")
        if tool is not None:
            source_read_step: PlanStep = {
                "tool_name": "source_read",
                "query": state["question"],
                "filters": {"trigger": "auto_target_trace"},
                "reason": "Read source for resolved target trace evidence.",
            }
            result = tool.invoke(
                _tool_input(
                    state,
                    source_read_step,
                    evidence=_prioritized_source_evidence(state.get("evidence", [])),
                )
            )
            state["tool_results"].append(result)
            state["used_tools"].append(result.tool_name)
            state["observations"].append(
                f"Auto source_read after target_trace: {result.summary}"
            )
            state["warnings"] = [*state.get("warnings", []), *result.warnings]
            state["source_snippets"] = _dedupe_source_snippets(
                [*state.get("source_snippets", []), *result.source_snippets]
            )
            executed_source_read = True
            if not result.source_snippets:
                state["source_reading_skipped_reason"] = _source_skip_reason(
                    result.warnings
                )

    if "source_read" not in state.get("used_tools", []):
        state["source_reading_skipped_reason"] = "not_requested_by_retrieval_plan"
    return state


def generate_final_answer(state: AgenticRagState) -> AgenticRagState:
    """Generate the final answer from accumulated evidence."""
    context_config = state["runtime_config"].context
    prompt = planner.build_final_prompt(
        question=state["question"],
        understanding=state["question_understanding"],
        evidence=state.get("evidence", [])[: context_config.max_evidence_items],
        source_snippets=state.get("source_snippets", [])[
            : context_config.max_source_snippets
        ],
        observations=state.get("observations", []),
        conversation_history=state.get("conversation_history", []),
        source_reading_skipped_reason=state.get("source_reading_skipped_reason"),
    )
    state["answer"] = invoke_agent_model(prompt)
    return state


def persist_turn_memory(state: AgenticRagState) -> AgenticRagState:
    """Persist minimal thread memory."""
    update_thread(
        state["thread_id"],
        state["project_id"],
        state.get("user_id"),
        state["project_path"],
        state["question"],
        state["answer"],
        len(state.get("evidence", [])),
    )
    return state


def _tool_input(
    state: AgenticRagState,
    step: PlanStep,
    evidence: list[RagEvidence] | None = None,
) -> ToolInput:
    return ToolInput(
        query=step.get("query", state["question"]),
        filters=step.get("filters", {}),
        project_id=state["project_id"],
        project_path=Path(state["project_path"]),
        top_k=state["top_k"],
        graph_depth=state["graph_depth"],
        source_available=state.get("source_available", False),
        max_source_files=state["runtime_config"].source_reading.max_files,
        max_source_lines=state["runtime_config"].source_reading.max_lines_per_file,
        source_line_padding=state["runtime_config"].source_reading.line_padding,
        evidence=evidence if evidence is not None else state.get("evidence", []),
        thread_id=state.get("thread_id"),
    )


def _source_read_evidence_for_step(
    state: AgenticRagState,
    tool_name: str,
) -> list[RagEvidence] | None:
    if tool_name != "source_read":
        return None
    intent = state.get("question_understanding", {}).get("intent")
    if intent != "task_planning":
        return None
    return _prioritized_task_planning_source_evidence(state.get("evidence", []))


def _has_file_evidence(evidence: list[RagEvidence]) -> bool:
    return any(item.file_path for item in evidence)


def _source_skip_reason(warnings: list[str]) -> str:
    for warning in warnings:
        if warning.startswith("source_read skipped:"):
            return warning.removeprefix("source_read skipped:").strip()
    return "source_read_returned_no_snippets"


def _should_auto_source_read(state: AgenticRagState) -> bool:
    if not state.get("source_available", False):
        return False
    evidence = state.get("evidence", [])
    if not _has_file_evidence(evidence):
        return False
    if "target_trace" in state.get("used_tools", []):
        return True
    intent = state.get("question_understanding", {}).get("intent")
    return intent == "persistence_location" and any(
        _is_trace_evidence(item) for item in evidence
    )


def _is_trace_evidence(item: RagEvidence) -> bool:
    if item.source == "target_trace":
        return True
    if item.evidence_type.startswith("target_relation:"):
        return True
    return item.evidence_type in {
        "edge:writes_table",
        "table_write",
        "table_mapping",
        "target_match",
    }


def _prioritized_source_evidence(evidence: list[RagEvidence]) -> list[RagEvidence]:
    return sorted(
        evidence,
        key=lambda item: (
            source_priority(item.evidence_type),
            item.file_path or "",
            item.start_line or 0,
            item.symbol or "",
        ),
    )


def _prioritized_task_planning_source_evidence(
    evidence: list[RagEvidence],
) -> list[RagEvidence]:
    indexed = list(enumerate(evidence))
    ranked = sorted(
        indexed,
        key=lambda item: (
            task_planning_source_priority(item[1].evidence_type),
            item[0],
        ),
    )
    return [item for _, item in ranked]


def _bounded_conversation_history(
    messages: list[dict[str, str]],
    max_messages: int,
    max_chars: int,
) -> list[dict[str, str]]:
    if max_messages <= 0 or max_chars <= 0:
        return []
    selected: list[dict[str, str]] = []
    remaining_chars = max_chars
    for message in reversed(messages[-max_messages:]):
        role = message.get("role", "").strip()
        content = message.get("content", "").strip()
        if not role or not content:
            continue
        if len(content) > remaining_chars:
            content = content[:remaining_chars].rstrip()
        if not content:
            continue
        selected.append({"role": role, "content": content})
        remaining_chars -= len(content)
        if remaining_chars <= 0:
            break
    return list(reversed(selected))


def _step_signature(step: PlanStep) -> str:
    tool_name = step.get("tool_name", "")
    filters = step.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}
    signature_filters = filters
    if tool_name == "aggregate_query":
        signature_filters = {
            "group_by": filters.get("group_by"),
            "module": filters.get("module"),
            "path_contains": filters.get("path_contains"),
        }
    return json.dumps(
        {
            "tool_name": tool_name,
            "filters": signature_filters,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _dedupe_evidence(items: list[RagEvidence]) -> list[RagEvidence]:
    seen: set[tuple[str, str | None, int | None, str | None]] = set()
    result: list[RagEvidence] = []
    for item in items:
        key = (item.evidence_type, item.file_path, item.start_line, item.symbol)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_source_snippets(items: list[SourceSnippet]) -> list[SourceSnippet]:
    seen: set[tuple[str, int, int]] = set()
    result: list[SourceSnippet] = []
    for item in items:
        key = (item.file_path, item.start_line, item.end_line)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
