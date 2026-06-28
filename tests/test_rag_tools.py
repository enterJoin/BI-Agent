from pathlib import Path
from typing import Any

from springgraph.models import Edge, File, Symbol
from springgraph.rag.config.models import ScopeFallbackConfig, ToolConfig
from springgraph.rag.schemas import RagEvidence
from springgraph.rag.tools import implementations
from springgraph.rag.tools.implementations import (
    AggregateQueryTool,
    ExecutionTraceTool,
    TargetTraceTool,
)
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


def test_execution_trace_reads_full_method_body(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "src/main/java/demo/Flow.java"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\n".join(
            [
                "package demo;",
                "public class Flow {",
                "  public void run() {",
                "    service.load();",
                "    if (service.empty()) {",
                "      return;",
                "    }",
                "    for (String item : items) {",
                "      if (item == null) {",
                "        continue;",
                "      }",
                "      kafkaService.send(topic, item);",
                "    }",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    file_row = File(
        id="file-flow",
        project_id="project-1",
        path="src/main/java/demo/Flow.java",
        module_name="demo",
        service_name="demo",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )
    symbol = Symbol(
        id="method-run",
        project_id="project-1",
        file_id="file-flow",
        kind="method",
        name="run",
        qualified_name="demo.Flow.run",
        start_line=3,
        end_line=14,
        meta={},
    )

    class FakeSession:
        pass

    class FakeScope:
        def __enter__(self) -> FakeSession:
            return FakeSession()

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(implementations, "session_scope", lambda: FakeScope())
    monkeypatch.setattr(
        implementations,
        "_resolve_execution_targets",
        lambda **_: [(symbol, file_row)],
    )
    monkeypatch.setattr(
        implementations,
        "_resolve_downstream_methods",
        lambda **_: [],
    )
    monkeypatch.setattr(
        implementations,
        "_resolve_sql_statement_rows",
        lambda **_: [],
    )
    monkeypatch.setattr(
        implementations,
        "_trace_method_table_relations",
        lambda **_: [],
    )

    tool = ExecutionTraceTool(
        ToolConfig(
            name="execution_trace",
            description="",
            capabilities=[],
            requires=["project_id", "project_path"],
        )
    )

    result = tool.invoke(
        ToolInput(
            query="run执行详细步骤是什么",
            filters={},
            project_id="project-1",
            project_path=tmp_path,
            top_k=8,
            graph_depth=2,
            source_available=True,
            max_source_files=3,
            max_source_lines=80,
            source_line_padding=3,
        )
    )

    assert result.tool_name == "execution_trace"
    assert result.source_snippets[0].start_line == 3
    assert result.source_snippets[0].end_line == 14
    assert "continue;" in result.source_snippets[0].content
    assert "control" in (result.evidence[0].content_excerpt or "")
    assert "message" in (result.evidence[0].content_excerpt or "")


def test_aggregate_tables_keeps_local_table_results_without_fallback(
    monkeypatch: object,
) -> None:
    table = Symbol(
        id="symbol-table",
        project_id="project-1",
        file_id="file-entity",
        kind="db_table",
        name="ums_member",
        qualified_name="db_table:ums_member",
        start_line=18,
        end_line=18,
        meta={"table": "ums_member"},
    )
    file_row = File(
        id="file-entity",
        project_id="project-1",
        path="guli-member-service/src/main/java/MemberEntity.java",
        module_name="guli-member-service",
        service_name="guli-member-service",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )

    class FakeRows:
        def all(self) -> list[tuple[Symbol, File]]:
            return [(table, file_row)]

    class FakeSession:
        def execute(self, statement: object) -> FakeRows:
            return FakeRows()

    class FakeScope:
        def __enter__(self) -> FakeSession:
            return FakeSession()

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(implementations, "session_scope", lambda: FakeScope())
    monkeypatch.setattr(implementations, "_resolve_module", lambda **_: "svc")

    def fail_fallback(**_: object) -> list[RagEvidence]:
        raise AssertionError("fallback should not run when local tables exist")

    monkeypatch.setattr(implementations, "_module_scope_fallback", fail_fallback)

    result = implementations._aggregate_tables(
        ToolConfig(
            name="aggregate_query",
            description="",
            capabilities=[],
            requires=["project_id"],
        ),
        ToolInput(
            query="guli-member-service系统的表都有哪些",
            filters={},
            project_id="project-1",
            project_path=Path("F:/demo"),
            top_k=8,
            graph_depth=2,
            source_available=True,
            max_source_files=3,
            max_source_lines=80,
            source_line_padding=3,
        ),
    )

    assert [item.evidence_type for item in result.evidence] == ["table_usage"]
    assert result.evidence[0].metadata["table"] == "ums_member"


def test_aggregate_tables_adds_module_fallback_when_local_tables_empty(
    monkeypatch: object,
) -> None:
    fallback = RagEvidence(
        evidence_type="module_scope_empty",
        source="module_scope",
        symbol="module:guli-auth-service",
    )

    class FakeRows:
        def all(self) -> list[tuple[Symbol, File]]:
            return []

    class FakeSession:
        def execute(self, statement: object) -> FakeRows:
            return FakeRows()

    class FakeScope:
        def __enter__(self) -> FakeSession:
            return FakeSession()

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(implementations, "session_scope", lambda: FakeScope())
    monkeypatch.setattr(
        implementations,
        "_resolve_module",
        lambda **_: "guli-auth-service",
    )
    monkeypatch.setattr(
        implementations,
        "_module_scope_fallback",
        lambda **_: [fallback],
    )

    result = implementations._aggregate_tables(
        ToolConfig(
            name="aggregate_query",
            description="",
            capabilities=[],
            requires=["project_id"],
        ),
        ToolInput(
            query="guli-auth-service系统的表都有哪些",
            filters={},
            project_id="project-1",
            project_path=Path("F:/demo"),
            top_k=8,
            graph_depth=2,
            source_available=True,
            max_source_files=3,
            max_source_lines=80,
            source_line_padding=3,
        ),
    )

    assert [item.evidence_type for item in result.evidence] == [
        "module_scope_empty"
    ]


def test_module_scope_fallback_reports_related_service_and_tables(
    monkeypatch: object,
) -> None:
    remote = Symbol(
        id="remote-member",
        project_id="project-1",
        file_id="file-feign",
        kind="remote_service",
        name="guli-member-service",
        qualified_name="remote_service:guli-member-service",
        start_line=9,
        end_line=9,
        meta={},
    )
    remote_file = File(
        id="file-feign",
        project_id="project-1",
        path="guli-auth-service/src/main/java/MemberFeignService.java",
        module_name="guli-auth-service",
        service_name="guli-auth-service",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )
    related_table = RagEvidence(
        evidence_type="related_service_table",
        source="module_scope",
        symbol="db_table:ums_member",
        metadata={"related_service": "guli-member-service"},
    )

    monkeypatch.setattr(
        implementations,
        "load_scope_fallback_config",
        lambda: ScopeFallbackConfig(
            enabled=True,
            related_service_symbol_kinds=["remote_service"],
            related_service_limit=3,
            related_table_limit_per_service=20,
        ),
    )
    monkeypatch.setattr(implementations, "_module_has_files", lambda *_: True)
    monkeypatch.setattr(
        implementations,
        "_module_related_service_rows",
        lambda **_: [(remote, remote_file)],
    )
    monkeypatch.setattr(
        implementations,
        "_related_service_table_evidence",
        lambda **_: [related_table],
    )

    result = implementations._module_scope_fallback(
        session=object(),
        project_id="project-1",
        module="guli-auth-service",
        query="guli-auth-service系统的表都有哪些",
        scope="tables",
        include_related_tables=True,
        table_limit=8,
    )

    assert [item.evidence_type for item in result] == [
        "module_scope_empty",
        "module_related_service",
        "related_service_table",
    ]
    assert result[1].metadata["related_service"] == "guli-member-service"


def test_rank_table_rows_prioritizes_query_table_terms() -> None:
    member_file = File(
        id="file-member",
        project_id="project-1",
        path="guli-member-service/src/main/java/MemberDao.java",
        module_name="guli-member-service",
        service_name="guli-member-service",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )
    growth_file = File(
        id="file-growth",
        project_id="project-1",
        path="guli-member-service/src/main/java/GrowthChangeHistoryDao.java",
        module_name="guli-member-service",
        service_name="guli-member-service",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )
    member = Symbol(
        id="symbol-member",
        project_id="project-1",
        file_id="file-member",
        kind="db_table",
        name="member",
        qualified_name="db_table:member",
        start_line=15,
        end_line=15,
        meta={"table": "member"},
    )
    growth = Symbol(
        id="symbol-growth",
        project_id="project-1",
        file_id="file-growth",
        kind="db_table",
        name="growth_change_history",
        qualified_name="db_table:growth_change_history",
        start_line=15,
        end_line=15,
        meta={"table": "growth_change_history"},
    )

    rows = implementations._rank_table_rows(
        [(growth, growth_file), (member, member_file)],
        "member user birthday field",
    )

    assert rows[0][0].qualified_name == "db_table:member"
