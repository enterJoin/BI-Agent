from pathlib import Path

from springgraph.rag.config.models import ToolConfig
from springgraph.rag.schemas import RagEvidence
from springgraph.rag.tools import implementations
from springgraph.rag.tools.implementations import AggregateQueryTool
from springgraph.rag.tools.schemas import ToolInput, ToolResult


def test_aggregate_query_treats_store_question_as_table_query(
    monkeypatch: object,
) -> None:
    captured: dict[str, ToolInput] = {}

    def fake_aggregate_tables(
        config: ToolConfig,
        tool_input: ToolInput,
    ) -> ToolResult:
        captured["tool_input"] = tool_input
        return ToolResult(
            tool_name=config.name,
            summary="table aggregation",
            evidence=[
                RagEvidence(
                    evidence_type="table_usage",
                    source="aggregate",
                    symbol="db_table:oms_order",
                )
            ],
        )

    monkeypatch.setattr(
        implementations,
        "_aggregate_tables",
        fake_aggregate_tables,
    )
    tool = AggregateQueryTool(
        ToolConfig(
            name="aggregate_query",
            description="",
            capabilities=[],
            requires=["project_id"],
        )
    )

    result = tool.invoke(
        ToolInput(
            query="\u8ba2\u5355\u4fe1\u606f\u662f\u5728\u54ea\u91cc\u5b58\u5165\u7684",
            filters={},
            project_id="project-1",
            project_path=Path("F:/demo"),
            top_k=8,
            graph_depth=2,
            source_available=True,
            max_source_files=3,
            max_source_lines=80,
            source_line_padding=3,
        )
    )

    assert captured["tool_input"].query == (
        "\u8ba2\u5355\u4fe1\u606f\u662f\u5728\u54ea\u91cc\u5b58\u5165\u7684"
    )
    assert result.summary == "table aggregation"
    assert result.evidence[0].symbol == "db_table:oms_order"
