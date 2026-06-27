"""Build the controlled Agentic RAG LangGraph."""

from typing import Any, Protocol, cast

from springgraph.rag.agent import nodes
from springgraph.rag.agent.state import AgenticRagState


class RunnableAgenticGraph(Protocol):
    """Protocol for compiled agentic graph objects."""

    def invoke(self, input: AgenticRagState) -> AgenticRagState:
        """Run the graph."""


def build_agentic_rag_graph() -> RunnableAgenticGraph:
    """Build the controlled Agentic RAG graph."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _FallbackAgenticGraph()

    graph = StateGraph(AgenticRagState)
    graph.add_node("load_runtime_config", nodes.load_runtime_config)
    graph.add_node("apply_request_defaults", nodes.apply_request_defaults)
    graph.add_node("load_thread_memory", nodes.load_thread_memory)
    graph.add_node("check_project_source_path", nodes.check_project_source_path)
    graph.add_node("understand_question", nodes.understand_question)
    graph.add_node("propose_candidate_tools", nodes.propose_candidate_tools)
    graph.add_node("decide_next_action", nodes.decide_next_action)
    graph.add_node("execute_tool", nodes.execute_tool)
    graph.add_node("judge_evidence", nodes.judge_evidence)
    graph.add_node("generate_final_answer", nodes.generate_final_answer)
    graph.add_node("persist_turn_memory", nodes.persist_turn_memory)

    graph.add_edge(START, "load_runtime_config")
    graph.add_edge("load_runtime_config", "apply_request_defaults")
    graph.add_edge("apply_request_defaults", "load_thread_memory")
    graph.add_edge("load_thread_memory", "check_project_source_path")
    graph.add_edge("check_project_source_path", "understand_question")
    graph.add_edge("understand_question", "propose_candidate_tools")
    graph.add_edge("propose_candidate_tools", "decide_next_action")
    graph.add_conditional_edges(
        "decide_next_action",
        nodes.route_after_decision,
        {
            "execute_tool": "execute_tool",
            "generate_final_answer": "generate_final_answer",
        },
    )
    graph.add_edge("execute_tool", "judge_evidence")
    graph.add_conditional_edges(
        "judge_evidence",
        nodes.route_after_judge,
        {
            "decide_next_action": "decide_next_action",
            "generate_final_answer": "generate_final_answer",
        },
    )
    graph.add_edge("generate_final_answer", "persist_turn_memory")
    graph.add_edge("persist_turn_memory", END)
    return cast(RunnableAgenticGraph, graph.compile())


class _FallbackAgenticGraph:
    """Sequential loop fallback when LangGraph is unavailable."""

    def invoke(self, input: AgenticRagState) -> AgenticRagState:
        state = cast(AgenticRagState, dict(input))
        for node in (
            nodes.load_runtime_config,
            nodes.apply_request_defaults,
            nodes.load_thread_memory,
            nodes.check_project_source_path,
            nodes.understand_question,
            nodes.propose_candidate_tools,
        ):
            state = cast(Any, node)(state)

        while True:
            state = nodes.decide_next_action(state)
            if nodes.route_after_decision(state) == "generate_final_answer":
                break
            state = nodes.execute_tool(state)
            state = nodes.judge_evidence(state)
            if nodes.route_after_judge(state) == "generate_final_answer":
                break

        state = nodes.generate_final_answer(state)
        return nodes.persist_turn_memory(state)
