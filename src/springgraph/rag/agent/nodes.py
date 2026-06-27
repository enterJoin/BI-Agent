"""Node implementations for the controlled Agentic RAG graph."""

from pathlib import Path

from springgraph.rag.agent import candidates, planner
from springgraph.rag.agent.state import AgenticRagState
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
    state["iteration"] = 0
    state["tool_call_count"] = 0
    state["no_new_evidence_rounds"] = 0
    state["last_source_read_evidence_count"] = 0
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


def propose_candidate_tools(state: AgenticRagState) -> AgenticRagState:
    """Rank available tools, assisted by embeddings."""
    ranked = candidates.propose_candidate_tools(
        state["question"],
        state["tool_configs"],
        state["runtime_config"],
    )
    state["candidate_tools"] = [tool.name for tool in ranked]
    return state


def decide_next_action(state: AgenticRagState) -> AgenticRagState:
    """Ask LLM to select the next tool or finish."""
    candidate_names = set(state.get("candidate_tools", []))
    candidate_configs = [
        tool for tool in state["tool_configs"] if tool.name in candidate_names
    ]
    action = planner.decide_next_action(
        question=state["question"],
        understanding=state["question_understanding"],
        candidate_tools=candidate_configs,
        evidence=state.get("evidence", []),
        source_available=state.get("source_available", False),
        observations=state.get("observations", []),
        judge_result=state.get("judge_result"),
    )
    if action.get("action") == "final_answer" and _needs_source_reading(state):
        action = {
            "action": "call_tool",
            "tool_name": "source_reader",
            "query": state["question"],
            "reason": "Source reading is enabled and file evidence is available.",
        }
    state["next_action"] = action
    return state


def execute_tool(state: AgenticRagState) -> AgenticRagState:
    """Execute the selected registered tool."""
    action = state.get("next_action", {})
    if action.get("action") != "call_tool":
        state["should_continue"] = False
        return state

    tool_name = action.get("tool_name", "")
    registry = load_tool_registry()
    tool = registry.get(tool_name)
    if tool is None:
        state["warnings"].append(f"Unknown tool skipped: {tool_name}")
        state["should_continue"] = False
        return state

    if tool_name == "source_reader" and not state.get("source_available", False):
        state["warnings"].append("source_reader blocked by source path gate.")
        state["should_continue"] = False
        return state
    if tool_name == "source_reader" and len(
        state.get("evidence", [])
    ) <= state.get("last_source_read_evidence_count", 0):
        state.setdefault("observations", []).append(
            "Source reader skipped because no new evidence was available."
        )
        state["no_new_evidence_rounds"] = state.get("no_new_evidence_rounds", 0) + 1
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1
        return state

    before_count = len(state.get("evidence", [])) + len(
        state.get("source_snippets", [])
    )
    result = tool.invoke(
        ToolInput(
            query=action.get("query", state["question"]),
            project_id=state["project_id"],
            project_path=Path(state["project_path"]),
            top_k=state["top_k"],
            graph_depth=state["graph_depth"],
            source_available=state.get("source_available", False),
            max_source_files=state["runtime_config"].source_reading.max_files,
            max_source_lines=state[
                "runtime_config"
            ].source_reading.max_lines_per_file,
            source_line_padding=state["runtime_config"].source_reading.line_padding,
            evidence=state.get("evidence", []),
            thread_id=state.get("thread_id"),
        )
    )
    state.setdefault("tool_results", []).append(result)
    state.setdefault("used_tools", []).append(result.tool_name)
    state.setdefault("observations", []).append(result.summary)
    state["warnings"] = [*state.get("warnings", []), *result.warnings]
    state["evidence"] = _dedupe_evidence(
        [*state.get("evidence", []), *result.evidence]
    )
    state["source_snippets"] = _dedupe_source_snippets(
        [
            *state.get("source_snippets", []),
            *result.source_snippets,
        ]
    )
    if tool_name == "source_reader":
        state["last_source_read_evidence_count"] = len(state.get("evidence", []))
    state["tool_call_count"] = state.get("tool_call_count", 0) + 1

    after_count = len(state.get("evidence", [])) + len(
        state.get("source_snippets", [])
    )
    if after_count == before_count:
        state["no_new_evidence_rounds"] = state.get("no_new_evidence_rounds", 0) + 1
    else:
        state["no_new_evidence_rounds"] = 0
    return state


def judge_evidence(state: AgenticRagState) -> AgenticRagState:
    """Judge whether evidence is enough to answer."""
    state["judge_result"] = planner.judge_evidence(
        question=state["question"],
        understanding=state["question_understanding"],
        evidence=state.get("evidence", []),
        source_snippets=state.get("source_snippets", []),
        observations=state.get("observations", []),
    )
    state["iteration"] = state.get("iteration", 0) + 1
    state["should_continue"] = _should_continue(state)
    return state


def generate_final_answer(state: AgenticRagState) -> AgenticRagState:
    """Generate the final answer from accumulated evidence."""
    prompt = planner.build_final_prompt(
        question=state["question"],
        understanding=state["question_understanding"],
        evidence=state.get("evidence", []),
        source_snippets=state.get("source_snippets", []),
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


def route_after_decision(state: AgenticRagState) -> str:
    """Route after next-action decision."""
    if state.get("next_action", {}).get("action") == "call_tool":
        return "execute_tool"
    return "generate_final_answer"


def route_after_judge(state: AgenticRagState) -> str:
    """Route after evidence judgment."""
    if state.get("should_continue", False):
        return "decide_next_action"
    return "generate_final_answer"


def _should_continue(state: AgenticRagState) -> bool:
    config = state["runtime_config"]
    if _needs_source_reading(state):
        return True
    if state.get("judge_result", {}).get("evidence_sufficient", False):
        return False
    if state.get("iteration", 0) >= config.loop.max_iterations:
        return False
    if state.get("tool_call_count", 0) >= config.loop.max_tool_calls:
        return False
    return (
        state.get("no_new_evidence_rounds", 0)
        < config.loop.stop_when_no_new_evidence_rounds
    )


def _needs_source_reading(state: AgenticRagState) -> bool:
    if not state.get("read_source", True):
        return False
    if not state.get("source_available", False):
        return False
    if state.get("source_snippets"):
        return False
    if len(state.get("evidence", [])) <= state.get(
        "last_source_read_evidence_count",
        0,
    ):
        return False
    return any(item.file_path for item in state.get("evidence", []))


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
        file_path = item.file_path
        start_line = item.start_line
        end_line = item.end_line
        key = (file_path, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
