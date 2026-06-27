from pathlib import Path
from typing import cast

from springgraph.rag import service
from springgraph.rag.agent.state import AgenticRagState
from springgraph.rag.schemas import RagEvidence, RagRequest, SourceSnippet


class FakeAgenticGraph:
    def invoke(self, input: AgenticRagState) -> AgenticRagState:
        state = cast(AgenticRagState, dict(input))
        state["answer"] = "agentic answer"
        state["question_understanding"] = {
            "task_goal": "追踪数据入库链路",
            "business_terms": ["素材"],
            "technical_terms": ["Mapper"],
            "entities": [],
            "sub_questions": ["入口在哪里"],
        }
        state["evidence"] = [
            RagEvidence(
                evidence_type="symbol",
                source="relational",
                file_path="src/main/java/DemoMapper.java",
                start_line=10,
                end_line=20,
                symbol="DemoMapper.insert",
            )
        ]
        state["source_snippets"] = [
            SourceSnippet(
                file_path="src/main/java/DemoMapper.java",
                start_line=10,
                end_line=20,
                content="insert",
            )
        ]
        state["used_tools"] = ["aggregate_query", "source_read"]
        state["observations"] = ["Aggregate table query returned 1 evidence items."]
        state["source_reading_skipped_reason"] = None
        return state


def test_ask_project_agentic_mode_uses_agentic_graph(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        service,
        "build_agentic_rag_graph",
        lambda: FakeAgenticGraph(),
    )

    answer = service.ask_project(
        RagRequest(
            question="素材数据怎么入库？",
            project_path=str(tmp_path),
            mode="agentic",
        )
    )

    assert answer.mode == "agentic"
    assert answer.intent == "追踪数据入库链路"
    assert answer.used_tools == ["aggregate_query", "source_read"]
    assert answer.used_relational_search is True
    assert answer.used_source_reading is True
    assert answer.expanded_queries[:3] == ["素材数据怎么入库？", "素材", "Mapper"]
