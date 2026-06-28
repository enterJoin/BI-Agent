"""Build the controlled Agentic RAG LangGraph."""

from functools import lru_cache
from typing import Any, Protocol, cast

from springgraph.rag.agent import nodes
from springgraph.rag.agent.state import AgenticRagState


class RunnableAgenticGraph(Protocol):
    """Protocol for compiled agentic graph objects."""

    def invoke(self, input: AgenticRagState) -> AgenticRagState:
        """Run the graph."""


@lru_cache(maxsize=1)
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
    graph.add_node("resolve_query_context", nodes.resolve_query_context)
    graph.add_node("plan_retrieval", nodes.plan_retrieval)
    graph.add_node("execute_retrieval_plan", nodes.execute_retrieval_plan)
    graph.add_node("generate_final_answer", nodes.generate_final_answer)
    graph.add_node("persist_turn_memory", nodes.persist_turn_memory)

    graph.add_edge(START, "load_runtime_config")
    graph.add_edge("load_runtime_config", "apply_request_defaults")
    graph.add_edge("apply_request_defaults", "load_thread_memory")
    graph.add_edge("load_thread_memory", "resolve_query_context")
    graph.add_edge("resolve_query_context", "plan_retrieval")
    graph.add_edge("plan_retrieval", "execute_retrieval_plan")
    graph.add_edge("execute_retrieval_plan", "generate_final_answer")
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
            nodes.resolve_query_context,
            nodes.plan_retrieval,
            nodes.execute_retrieval_plan,
        ):
            state = cast(Any, node)(state)

        state = nodes.generate_final_answer(state)
        return nodes.persist_turn_memory(state)
