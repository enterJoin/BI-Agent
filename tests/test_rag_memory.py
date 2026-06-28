from springgraph.rag.agent import nodes
from springgraph.rag.agent.state import AgenticRagState
from springgraph.rag.config.models import (
    AgenticRagConfig,
    ContextConfig,
    MemoryConfig,
    RetrievalConfig,
    SafetyConfig,
    SourceReadingConfig,
)
from springgraph.rag.memory.store import get_thread
from springgraph.rag.schemas import RagEvidence, SourceSnippet
from springgraph.rag.tools.schemas import ToolInput, ToolResult


def test_load_thread_memory_bounds_conversation_history() -> None:
    thread_id = "thread:test-memory-bounds"
    memory = get_thread(thread_id)
    memory.messages = [
        {"role": "user", "content": "old1"},
        {"role": "assistant", "content": "old2"},
        {"role": "user", "content": "new1"},
        {"role": "assistant", "content": "new2"},
    ]
    memory.last_question = "new1"

    state = nodes.load_thread_memory(
        {
            "thread_id": thread_id,
            "load_memory": True,
            "project_path": "F:/demo",
            "runtime_config": _config(max_history_messages=3, max_history_chars=9),
            "observations": [],
        }
    )

    history = state["conversation_history"]

    assert len(history) <= 3
    assert sum(len(item["content"]) for item in history) <= 9
    assert history[-1] == {"role": "assistant", "content": "new2"}
    assert state["observations"] == [
        "Thread memory is available for follow-up context."
    ]


def test_generate_final_answer_includes_conversation_history(
    monkeypatch: object,
) -> None:
    captured: dict[str, str] = {}

    def fake_invoke_agent_model(prompt: str) -> str:
        captured["prompt"] = prompt
        return "answer"

    monkeypatch.setattr(nodes, "invoke_agent_model", fake_invoke_agent_model)
    state: AgenticRagState = {
        "question": "\u8fd9\u91cc\u7684\u5b83\u6307\u4ec0\u4e48",
        "question_understanding": {"task_goal": "follow up", "intent": "unknown"},
        "runtime_config": _config(max_history_messages=6, max_history_chars=4000),
        "evidence": [
            RagEvidence(
                evidence_type="table_usage",
                source="aggregate",
                file_path="OrderDao.java",
                start_line=10,
                end_line=10,
                symbol="db_table:oms_order",
                content_excerpt="table=oms_order",
            )
        ],
        "source_snippets": [],
        "observations": [],
        "conversation_history": [
            {
                "role": "user",
                "content": "\u8ba2\u5355\u4fe1\u606f\u5b58\u5728\u54ea\u91cc",
            },
            {"role": "assistant", "content": "\u5b58\u5728 oms_order"},
        ],
        "source_reading_skipped_reason": "not_requested_by_retrieval_plan",
    }

    result = nodes.generate_final_answer(state)

    assert result["answer"] == "answer"
    assert captured["prompt"].index("Current question:") < captured["prompt"].index(
        "Conversation history for reference only:"
    )
    assert "Conversation history for reference only:" in captured["prompt"]
    assert (
        "user: \u8ba2\u5355\u4fe1\u606f\u5b58\u5728\u54ea\u91cc"
        in captured["prompt"]
    )
    assert "assistant: \u5b58\u5728 oms_order" in captured["prompt"]
    assert "Evidence:" in captured["prompt"]


def test_resolved_contextual_question_is_used_for_planning(
    monkeypatch: object,
) -> None:
    calls: dict[str, object] = {}

    def fake_resolve_contextual_query(
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> dict[str, object]:
        calls["resolver_question"] = question
        calls["resolver_history"] = conversation_history
        return {
            "is_follow_up": True,
            "needs_context": True,
            "needs_clarification": False,
            "resolved_target": {
                "type": "job",
                "name": "tencentNineImageMappingHandler",
                "source": "history",
                "confidence": 0.95,
            },
            "rewritten_question": (
                "tencentNineImageMappingHandler "
                "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48"
            ),
            "retrieval_intent": "execution_flow",
            "preferred_tools": ["execution_trace"],
            "reason": "follow-up",
        }

    def fake_plan_question_retrieval(
        question: str,
        available_tools: list[object],
        source_available: bool,
        memory_observations: list[str],
        query_resolution: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        calls["planner_question"] = question
        calls["query_resolution"] = query_resolution
        return (
            {"task_goal": "trace", "intent": "execution_flow"},
            {"task_goal": "trace", "steps": []},
        )

    monkeypatch.setattr(
        nodes,
        "resolve_contextual_query",
        fake_resolve_contextual_query,
    )
    monkeypatch.setattr(
        nodes.planner,
        "plan_question_retrieval",
        fake_plan_question_retrieval,
    )
    state: AgenticRagState = {
        "question": "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48",
        "conversation_history": [
            {
                "role": "assistant",
                "content": (
                    "\u901a\u8fc7 `tencentNineImageMappingHandler` "
                    "\u8fd9\u4e2a Job \u5165\u5e93\u3002"
                ),
            }
        ],
        "tool_configs": [],
        "source_available": True,
        "observations": [],
    }

    state = nodes.resolve_query_context(state)
    state = nodes.plan_retrieval(state)

    assert calls["resolver_question"] == "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48"
    assert calls["planner_question"] == (
        "tencentNineImageMappingHandler "
        "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48"
    )
    assert state["question_understanding"]["rewritten_query"] == (
        "tencentNineImageMappingHandler "
        "\u8be6\u7ec6\u8fc7\u7a0b\u662f\u4ec0\u4e48"
    )


def test_generate_final_answer_includes_task_planning_guidance(
    monkeypatch: object,
) -> None:
    captured: dict[str, str] = {}

    def fake_invoke_agent_model(prompt: str) -> str:
        captured["prompt"] = prompt
        return "plan"

    monkeypatch.setattr(nodes, "invoke_agent_model", fake_invoke_agent_model)
    state: AgenticRagState = {
        "question": (
            "\u6211\u8981\u7ed9\u4f1a\u5458\u8868\u65b0\u589e\u751f\u65e5"
            "\u5b57\u6bb5\uff0c\u5e2e\u6211\u89c4\u5212\u600e\u4e48\u6539"
        ),
        "question_understanding": {
            "task_goal": "plan member birthday change",
            "intent": "task_planning",
        },
        "runtime_config": _config(max_history_messages=6, max_history_chars=4000),
        "evidence": [
            RagEvidence(
                evidence_type="table_usage",
                source="aggregate",
                file_path="MemberDao.java",
                start_line=15,
                end_line=15,
                symbol="db_table:member",
                content_excerpt="table=member; artifact=MemberDao",
            )
        ],
        "source_snippets": [],
        "observations": [],
        "conversation_history": [],
        "source_reading_skipped_reason": "not_requested_by_retrieval_plan",
    }

    result = nodes.generate_final_answer(state)

    assert result["answer"] == "plan"
    assert "Task planning answer sections:" in captured["prompt"]
    assert "\u4efb\u52a1\u7406\u89e3" in captured["prompt"]
    assert "Prefer adding a column when:" in captured["prompt"]
    assert "Prefer creating a new table when:" in captured["prompt"]


def test_execute_retrieval_plan_auto_reads_target_trace_source(
    monkeypatch: object,
) -> None:
    calls: list[str] = []

    class FakeTargetTraceTool:
        def invoke(self, tool_input: ToolInput) -> ToolResult:
            calls.append("target_trace")
            return ToolResult(
                tool_name="target_trace",
                summary="target trace",
                evidence=[
                    RagEvidence(
                        evidence_type="target_relation:writes_table",
                        source="target_trace",
                        file_path="src/main/java/OrderOperateHistoryDao.java",
                        start_line=15,
                        end_line=15,
                        symbol=(
                            "mapper:OrderOperateHistoryDao -> "
                            "db_table:oms_order_operate_history"
                        ),
                    )
                ],
            )

    class FakeSourceReadTool:
        def invoke(self, tool_input: ToolInput) -> ToolResult:
            calls.append("source_read")
            assert tool_input.evidence[0].evidence_type == (
                "target_relation:writes_table"
            )
            return ToolResult(
                tool_name="source_read",
                summary="source read",
                source_snippets=[
                    SourceSnippet(
                        file_path="src/main/java/OrderOperateHistoryDao.java",
                        start_line=12,
                        end_line=18,
                        content="interface OrderOperateHistoryDao",
                    )
                ],
            )

    class FakeRegistry:
        def get(self, name: str) -> object | None:
            return {
                "target_trace": FakeTargetTraceTool(),
                "source_read": FakeSourceReadTool(),
            }.get(name)

    monkeypatch.setattr(nodes, "load_tool_registry", lambda: FakeRegistry())
    state: AgenticRagState = {
        "thread_id": "thread-1",
        "project_path": "F:/demo",
        "project_id": "project-1",
        "question": "oms_order_operate_history table source",
        "top_k": 8,
        "graph_depth": 2,
        "source_available": True,
        "runtime_config": _config(max_history_messages=6, max_history_chars=4000),
        "retrieval_plan": {
            "steps": [
                {
                    "tool_name": "target_trace",
                    "query": "oms_order_operate_history table source",
                    "filters": {"intent": "persistence_location"},
                }
            ]
        },
        "question_understanding": {"intent": "persistence_location"},
        "tool_results": [],
        "used_tools": [],
        "observations": [],
        "evidence": [],
        "source_snippets": [],
        "warnings": [],
    }

    result = nodes.execute_retrieval_plan(state)

    assert calls == ["target_trace", "source_read"]
    assert result["used_tools"] == ["target_trace", "source_read"]
    assert result["source_snippets"][0].content == (
        "interface OrderOperateHistoryDao"
    )
    assert result.get("source_reading_skipped_reason") is None


def test_task_planning_source_read_prioritizes_table_evidence(
    monkeypatch: object,
) -> None:
    calls: list[str] = []

    class FakeSourceReadTool:
        def invoke(self, tool_input: ToolInput) -> ToolResult:
            calls.append("source_read")
            assert tool_input.evidence[0].evidence_type == "table_usage"
            return ToolResult(tool_name="source_read", summary="source read")

    class FakeRegistry:
        def get(self, name: str) -> object | None:
            if name == "source_read":
                return FakeSourceReadTool()
            return None

    monkeypatch.setattr(nodes, "load_tool_registry", lambda: FakeRegistry())
    state: AgenticRagState = {
        "thread_id": "thread-1",
        "project_path": "F:/demo",
        "project_id": "project-1",
        "question": "plan change",
        "top_k": 8,
        "graph_depth": 2,
        "source_available": True,
        "runtime_config": _config(max_history_messages=6, max_history_chars=4000),
        "retrieval_plan": {
            "steps": [
                {
                    "tool_name": "source_read",
                    "query": "plan change",
                    "filters": {"intent": "task_planning"},
                }
            ]
        },
        "question_understanding": {"intent": "task_planning"},
        "tool_results": [],
        "used_tools": [],
        "observations": [],
        "evidence": [
            RagEvidence(
                evidence_type="artifact",
                source="relational",
                file_path="Controller.java",
                symbol="class:MemberController",
            ),
            RagEvidence(
                evidence_type="table_usage",
                source="aggregate",
                file_path="MemberDao.java",
                symbol="db_table:member",
            ),
        ],
        "source_snippets": [],
        "warnings": [],
    }

    result = nodes.execute_retrieval_plan(state)

    assert calls == ["source_read"]
    assert result["used_tools"] == ["source_read"]


def _config(
    max_history_messages: int,
    max_history_chars: int,
) -> AgenticRagConfig:
    return AgenticRagConfig(
        retrieval=RetrievalConfig(
            default_top_k=8,
            max_top_k=50,
            default_graph_depth=2,
            max_graph_depth=3,
        ),
        source_reading=SourceReadingConfig(
            enabled_by_default=True,
            max_files=2,
            max_lines_per_file=40,
            line_padding=3,
        ),
        context=ContextConfig(max_evidence_items=60, max_source_snippets=2),
        memory=MemoryConfig(
            short_term_enabled=True,
            long_term_enabled=False,
            max_history_messages=max_history_messages,
            max_history_chars=max_history_chars,
        ),
        safety=SafetyConfig(
            require_evidence_citation=True,
            allow_shell_execution=False,
            allow_code_modification=False,
        ),
    )
