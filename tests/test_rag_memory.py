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
from springgraph.rag.schemas import RagEvidence


def test_load_thread_memory_bounds_conversation_history() -> None:
    thread_id = "thread:test-memory-bounds"
    memory = get_thread(thread_id)
    memory.messages = [
        {"role": "user", "content": "old1"},
        {"role": "assistant", "content": "old2"},
        {"role": "user", "content": "new1"},
        {"role": "assistant", "content": "new2"},
    ]

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
    assert "Conversation history:" in captured["prompt"]
    assert (
        "user: \u8ba2\u5355\u4fe1\u606f\u5b58\u5728\u54ea\u91cc"
        in captured["prompt"]
    )
    assert "assistant: \u5b58\u5728 oms_order" in captured["prompt"]
    assert "Evidence:" in captured["prompt"]


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
