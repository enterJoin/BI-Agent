from pathlib import Path
from typing import Any

from springgraph.models import Edge, File, Symbol
from springgraph.rag.config.models import ToolConfig
from springgraph.rag.schemas import RagEvidence
from springgraph.rag.tools import implementations
from springgraph.rag.tools.implementations import AggregateQueryTool, TargetTraceTool
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


def test_target_trace_returns_target_and_write_relation(
    monkeypatch: object,
) -> None:
    target = Symbol(
        id="symbol-table",
        project_id="project-1",
        file_id="file-entity",
        kind="db_table",
        name="oms_order_operate_history",
        qualified_name="db_table:oms_order_operate_history",
        start_line=18,
        end_line=18,
        meta={"table": "oms_order_operate_history"},
    )
    source = Symbol(
        id="symbol-dao",
        project_id="project-1",
        file_id="file-dao",
        kind="mapper",
        name="OrderOperateHistoryDao",
        qualified_name="mapper:OrderOperateHistoryDao",
        start_line=15,
        end_line=15,
        meta={},
    )
    entity_file = File(
        id="file-entity",
        project_id="project-1",
        path="guli-order-service/src/main/java/OrderOperateHistoryEntity.java",
        module_name="guli-order-service",
        service_name="guli-order-service",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )
    dao_file = File(
        id="file-dao",
        project_id="project-1",
        path="guli-order-service/src/main/java/OrderOperateHistoryDao.java",
        module_name="guli-order-service",
        service_name="guli-order-service",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )
    edge = Edge(
        id=1,
        project_id="project-1",
        source_id="symbol-dao",
        target_id="symbol-table",
        kind="writes_table",
        line=15,
        confidence=1.0,
        resolved_by="extractor",
        meta={"source": "BaseMapper"},
    )

    class FakeSession:
        def execute(self, statement: object) -> Any:
            return None

    class FakeScope:
        def __enter__(self) -> FakeSession:
            return FakeSession()

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(implementations, "session_scope", lambda: FakeScope())
    monkeypatch.setattr(
        implementations,
        "_resolve_target_symbols",
        lambda **_: [(target, entity_file)],
    )
    monkeypatch.setattr(
        implementations,
        "_trace_target_relations",
        lambda **_: [(edge, source, target, dao_file)],
    )

    tool = TargetTraceTool(
        ToolConfig(
            name="target_trace",
            description="",
            capabilities=[],
            requires=["project_id"],
        )
    )

    result = tool.invoke(
        ToolInput(
            query="oms_order_operate_history table source",
            filters={
                "intent": "persistence_location",
                "target": "oms_order_operate_history",
            },
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

    assert result.tool_name == "target_trace"
    assert [item.evidence_type for item in result.evidence] == [
        "target_relation:writes_table",
        "target_match",
    ]
    assert result.evidence[0].file_path == (
        "guli-order-service/src/main/java/OrderOperateHistoryDao.java"
    )
    assert result.evidence[0].metadata["edge_kind"] == "writes_table"
