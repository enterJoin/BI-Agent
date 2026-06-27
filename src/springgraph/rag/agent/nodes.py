"""Node implementations for the planned Agentic RAG graph."""

import json
from pathlib import Path

from springgraph.rag.agent import planner
from springgraph.rag.agent.state import AgenticRagState, PlanStep
from springgraph.rag.config.loader import load_agentic_rag_config
from springgraph.rag.llm import invoke_agent_model
from springgraph.rag.memory.store import get_thread, update_thread
from springgraph.rag.schemas import RagEvidence, SourceSnippet
from springgraph.rag.source_reader import check_source_path
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
    return state


def load_thread_memory(state: AgenticRagState) -> AgenticRagState:
    """Load minimal thread state."""
    if not state["runtime_config"].memory.short_term_enabled:
        return state
    memory = get_thread(state["thread_id"])
    if not state.get("project_path") and memory.project_path is not None:
        state["project_path"] = memory.project_path
    if memory.last_question is not None:
        state["observations"].append(
            f"Thread memory last question: {memory.last_question}"
        )
    return state


def check_project_source_path(state: AgenticRagState) -> AgenticRagState:
    """Apply the hard source path gate."""
    allowed, reason = check_source_path(Path(state["project_path"]))
    if not state.get("read_source", True):
        allowed = False
        reason = "read_source_disabled"
    state["source_available"] = allowed
    state["source_reading_skipped_reason"] = reason
    return state


def understand_question(state: AgenticRagState) -> AgenticRagState:
    """Create open-ended question understanding with LLM."""
    # TODO: Make question understanding context-aware by passing concise thread
    # memory, recent evidence, recent symbols, and project library hints.
    state["question_understanding"] = planner.understand_question(
        state["question"]
    )
    return state


def create_retrieval_plan(state: AgenticRagState) -> AgenticRagState:
    """Create one structured retrieval plan."""
    state["retrieval_plan"] = planner.create_retrieval_plan(
        question=state["question"],
        understanding=state["question_understanding"],
        available_tools=state["tool_configs"],
        source_available=state.get("source_available", False),
    )
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
        result = tool.invoke(_tool_input(state, step))
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
        source_reading_skipped_reason=state.get("source_reading_skipped_reason"),
    )
    state["answer"] = invoke_agent_model(prompt)
    return state


def persist_turn_memory(state: AgenticRagState) -> AgenticRagState:
    """Persist minimal thread memory."""
    update_thread(
        state["thread_id"],
        state["project_path"],
        state["question"],
        state["answer"],
        len(state.get("evidence", [])),
    )
    return state


def _tool_input(state: AgenticRagState, step: PlanStep) -> ToolInput:
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
        evidence=state.get("evidence", []),
        thread_id=state.get("thread_id"),
    )


def _has_file_evidence(evidence: list[RagEvidence]) -> bool:
    return any(item.file_path for item in evidence)


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
