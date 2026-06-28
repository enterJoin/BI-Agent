from springgraph.refinement import _typed_artifacts
from springgraph.vector_search import _resolve_chunk_types


def test_vector_search_resolves_artifact_types_to_chunk_types() -> None:
    assert _resolve_chunk_types(
        ["project_knowledge"],
        ["job_entrypoint", "artifact_db_table"],
    ) == [
        "project_knowledge",
        "artifact_job_entrypoint",
        "artifact_db_table",
    ]


def test_xxljob_annotation_becomes_job_entrypoint_artifact() -> None:
    symbol = _typed_artifacts._SymbolRow(
        id="symbol-1",
        kind="annotation_usage",
        name="@XxlJob syncOrder",
        qualified_name="annotation_usage:XxlJob:syncOrder",
        language="java",
        start_line=10,
        end_line=10,
        signature=None,
        annotations=[],
        metadata={"annotation": "XxlJob"},
        file_path="demo/src/main/java/Job.java",
        module_name="demo",
        service_name="demo",
    )

    assert _typed_artifacts._artifact_type(symbol, set()) == "job_entrypoint"


def test_other_annotation_usage_is_summarized_only() -> None:
    symbol = _typed_artifacts._SymbolRow(
        id="symbol-1",
        kind="annotation_usage",
        name="@RequestMapping /orders",
        qualified_name="annotation_usage:RequestMapping:/orders",
        language="java",
        start_line=10,
        end_line=10,
        signature=None,
        annotations=[],
        metadata={"annotation": "RequestMapping", "values": {"value": "/orders"}},
        file_path="demo/src/main/java/OrderController.java",
        module_name="demo",
        service_name="demo",
    )

    assert _typed_artifacts._artifact_type(symbol, set()) is None


def test_annotation_usage_without_value_is_summarized_only() -> None:
    symbol = _typed_artifacts._SymbolRow(
        id="symbol-1",
        kind="annotation_usage",
        name="@Override",
        qualified_name="annotation_usage:Override",
        language="java",
        start_line=10,
        end_line=10,
        signature=None,
        annotations=[],
        metadata={"annotation": "Override", "values": {}},
        file_path="demo/src/main/java/OrderService.java",
        module_name="demo",
        service_name="demo",
    )

    assert _typed_artifacts._artifact_type(symbol, set()) is None


def test_method_needs_non_structural_relation_or_annotation() -> None:
    symbol = _typed_artifacts._SymbolRow(
        id="symbol-1",
        kind="method",
        name="calculate",
        qualified_name="com.demo.OrderService.calculate",
        language="java",
        start_line=10,
        end_line=20,
        signature="void calculate()",
        annotations=[],
        metadata={},
        file_path="demo/src/main/java/OrderService.java",
        module_name="demo",
        service_name="demo",
    )

    assert _typed_artifacts._artifact_type(symbol, set()) is None
    assert _typed_artifacts._artifact_type(symbol, {"symbol-1"}) == "method"


def test_unknown_non_summary_kind_is_kept_as_typed_artifact() -> None:
    symbol = _typed_artifacts._SymbolRow(
        id="symbol-1",
        kind="future_kind",
        name="future",
        qualified_name="future:future",
        language="java",
        start_line=10,
        end_line=10,
        signature=None,
        annotations=[],
        metadata={},
        file_path="demo/src/main/java/Future.java",
        module_name="demo",
        service_name="demo",
    )

    assert _typed_artifacts._artifact_type(symbol, set()) == "future_kind"


def test_summary_kinds_are_not_single_artifacts() -> None:
    symbol = _typed_artifacts._SymbolRow(
        id="symbol-1",
        kind="field",
        name="platType",
        qualified_name="com.demo.Report.platType",
        language="java",
        start_line=10,
        end_line=10,
        signature=None,
        annotations=[],
        metadata={},
        file_path="demo/src/main/java/Report.java",
        module_name="demo",
        service_name="demo",
    )

    assert _typed_artifacts._artifact_type(symbol, set()) is None
