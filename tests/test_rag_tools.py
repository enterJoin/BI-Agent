from pathlib import Path
from typing import Any

from springgraph.models import Edge, File, Symbol
from springgraph.rag.config.models import (
    AggregationSpecConfig,
    ScopeFallbackConfig,
    ToolConfig,
)
from springgraph.rag.schemas import RagEvidence
from springgraph.rag.tools import implementations
from springgraph.rag.tools.implementations import (
    AggregateQueryTool,
    ExecutionTraceTool,
    TargetTraceTool,
)
from springgraph.rag.tools.schemas import ToolInput, ToolResult
from springgraph.refinement import _extractors


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


def test_aggregate_query_groups_jobs_without_table_aggregation(
    monkeypatch: object,
) -> None:
    job = Symbol(
        id="job-symbol",
        project_id="project-1",
        file_id="job-file",
        kind="annotation_usage",
        name="@XxlJob tencentMaterialReportHandler",
        qualified_name=(
            "annotation_usage:XxlJob:tencentMaterialReportHandler:"
            "com.demo.TencentReportJob.tencentMaterialReportHandler:0"
        ),
        start_line=65,
        end_line=65,
        meta={
            "annotation": "XxlJob",
            "values": {"value": "tencentMaterialReportHandler"},
            "target_qualified_name": (
                "com.demo.TencentReportJob.tencentMaterialReportHandler"
            ),
        },
    )
    file_row = File(
        id="job-file",
        project_id="project-1",
        path="demo/src/main/java/com/demo/quartz/TencentReportJob.java",
        module_name="demo",
        service_name="demo",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )

    class FakeRows:
        def all(self) -> list[tuple[Symbol, File]]:
            return [(job, file_row)]

    class FakeSession:
        def execute(self, statement: object) -> FakeRows:
            return FakeRows()

    class FakeScope:
        def __enter__(self) -> FakeSession:
            return FakeSession()

        def __exit__(self, *_: object) -> None:
            return None

    def fail_aggregate_tables(
        config: ToolConfig,
        tool_input: ToolInput,
    ) -> ToolResult:
        raise AssertionError("job aggregation must not use table aggregation")

    monkeypatch.setattr(implementations, "session_scope", lambda: FakeScope())
    monkeypatch.setattr(implementations, "_resolve_module", lambda **_: "demo")
    monkeypatch.setattr(implementations, "_aggregate_tables", fail_aggregate_tables)

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
            query="\u5e7f\u70b9\u901a\u4e5d\u56fe\u7d20\u6750\u6d88\u8017\u662f\u54ea\u4e2aJob\u5165\u5e93",
            filters={"group_by": "job"},
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

    assert result.evidence[0].evidence_type == "job_entrypoint"
    assert result.evidence[0].metadata["job"] == "tencentMaterialReportHandler"
    assert result.evidence[0].metadata["symbol_id"] == "job-symbol"


def test_query_terms_adds_generic_cjk_ngrams() -> None:
    terms = implementations._query_terms(
        "\u5e7f\u70b9\u901a\u4e5d\u56fe\u7d20\u6750\u6d88\u8017\u6570\u636e "
        "TencentReportService material_sync"
    )

    assert "\u5e7f\u70b9\u901a" in terms
    assert "\u4e5d\u56fe" in terms
    assert "\u7d20\u6750" in terms
    assert "\u6d88\u8017" in terms
    assert "Tencent" in terms
    assert "Report" in terms
    assert "Service" in terms
    assert "material" in terms
    assert "sync" in terms


def test_execution_target_candidates_drop_bare_entrypoint_suffix() -> None:
    candidates = implementations._execution_target_candidates(
        "tencentNineImageMappingHandler \u8be6\u7ec6\u8fc7\u7a0b",
        {},
    )

    assert "tencentNineImageMappingHandler" in candidates
    assert "Handler" not in candidates


def test_insert_sql_does_not_treat_duplicate_update_columns_as_tables() -> None:
    sql = """
        INSERT INTO nine_image_mapping
        (component_id, image_id, nine_order, file_id, images_md5)
        VALUES
        (?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
        file_id = VALUES(file_id),
        images_md5 = VALUES(images_md5)
    """

    assert _extractors._tables_from_sql(sql, "insert") == [
        ("writes_table", "nine_image_mapping")
    ]


def test_job_name_is_job_aggregation_alias() -> None:
    spec = implementations._aggregation_spec_for("job_name")

    assert spec is not None
    assert spec.group_by == "job"


def test_rank_aggregation_rows_filters_to_strong_hint_match() -> None:
    spec = implementations._aggregation_spec_for("job")
    assert spec is not None
    file_row = File(
        id="file-1",
        project_id="project-1",
        path="bijobserv/src/main/java/TencentDataInfoJob.java",
        module_name="bijobserv",
        service_name="bijobserv",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )
    exact_job = Symbol(
        id="job-1",
        project_id="project-1",
        file_id="file-1",
        kind="annotation_usage",
        name="@XxlJob adCreativeInfoV3Handler",
        qualified_name="annotation_usage:XxlJob:adCreativeInfoV3Handler",
        start_line=10,
        end_line=10,
        meta={"values": {"value": "adCreativeInfoV3Handler"}},
    )
    related_job = Symbol(
        id="job-2",
        project_id="project-1",
        file_id="file-1",
        kind="annotation_usage",
        name="@XxlJob huaweiSyncAdCreativeHandler",
        qualified_name="annotation_usage:XxlJob:huaweiSyncAdCreativeHandler",
        start_line=20,
        end_line=20,
        meta={"values": {"value": "huaweiSyncAdCreativeHandler"}},
    )

    ranked = implementations._rank_aggregation_rows(
        [(related_job, file_row), (exact_job, file_row)],
        "广告创意数据 adCreativeInfoV3Handler creative",
        spec,
        term_weights={"adcreativeinfov3handler": 8, "creative": 8},
    )

    assert [symbol.name for symbol, _ in ranked] == [
        "@XxlJob adCreativeInfoV3Handler"
    ]


def test_aggregation_config_covers_known_symbol_kinds() -> None:
    configured_kinds = {
        kind
        for spec in implementations.load_aggregation_specs()
        for kind in spec.symbol_kinds
    }

    assert {
        "annotation_usage",
        "bean",
        "cache_key",
        "class",
        "config",
        "constructor",
        "data_contract",
        "db_column",
        "db_table",
        "enum",
        "field",
        "file",
        "import",
        "interface",
        "mapper",
        "method",
        "mq_exchange",
        "mq_queue",
        "mq_topic",
        "oauth_provider",
        "package",
        "parameter",
        "permission_rule",
        "remote_service",
        "resource",
        "route",
        "service",
        "sql_statement",
    } <= configured_kinds
    assert "mybatis_statement" not in configured_kinds


def test_aggregate_query_uses_configured_group_by_without_code_branch(
    monkeypatch: object,
) -> None:
    controller = Symbol(
        id="controller-symbol",
        project_id="project-1",
        file_id="controller-file",
        kind="class",
        name="OrderController",
        qualified_name="com.demo.OrderController",
        start_line=12,
        end_line=80,
        meta={},
    )
    file_row = File(
        id="controller-file",
        project_id="project-1",
        path="demo/src/main/java/com/demo/OrderController.java",
        module_name="demo",
        service_name="demo",
        language="java",
        content_hash="hash",
        size_bytes=1,
    )

    class FakeRows:
        def all(self) -> list[tuple[Symbol, File]]:
            return [(controller, file_row)]

    class FakeSession:
        def execute(self, statement: object) -> FakeRows:
            return FakeRows()

    class FakeScope:
        def __enter__(self) -> FakeSession:
            return FakeSession()

        def __exit__(self, *_: object) -> None:
            return None

    spec = AggregationSpecConfig(
        group_by="controller",
        aliases=("controllers",),
        symbol_kinds=("class",),
        evidence_type="controller_entry",
        label="controller",
    )
    monkeypatch.setattr(implementations, "load_aggregation_specs", lambda: (spec,))
    monkeypatch.setattr(implementations, "session_scope", lambda: FakeScope())
    monkeypatch.setattr(implementations, "_resolve_module", lambda **_: None)

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
            query="order controllers",
            filters={"group_by": "controllers"},
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

    assert result.evidence[0].evidence_type == "controller_entry"
    assert result.evidence[0].metadata["group_by"] == "controller"
    assert result.evidence[0].metadata["controller"] == "OrderController"


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


def test_infer_module_ignores_library_vector_evidence() -> None:
    evidence = [
        RagEvidence(
            evidence_type="vector_chunk:project_knowledge",
            source="vector",
            file_path="library/03-reporting-and-data-sources.md",
            metadata={
                "module_name": "library",
                "metadata": {"chunk_type": "project_knowledge"},
            },
        ),
        RagEvidence(
            evidence_type="method",
            source="aggregate",
            file_path="bijobserv/src/main/java/com/demo/Job.java",
            metadata={"module_name": "bijobserv"},
        ),
    ]

    assert implementations._infer_module(evidence) == "bijobserv"


def test_infer_module_returns_none_for_only_library_evidence() -> None:
    evidence = [
        RagEvidence(
            evidence_type="vector_chunk:project_knowledge_parent",
            source="vector",
            file_path="library/关键词.md",
            metadata={
                "module_name": "library",
                "metadata": {"chunk_type": "project_knowledge_parent"},
            },
        )
    ]

    assert implementations._infer_module(evidence) is None


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
