from pathlib import Path

from springgraph.extractor.java_tree_sitter import extract_java_file


def test_extracts_controller_route_and_refs() -> None:
    root = Path(__file__).parent / "fixtures" / "sample_springboot"
    path = root / "src/main/java/com/example/demo/controller/UserController.java"
    result = extract_java_file(
        "project:test",
        path,
        "src/main/java/com/example/demo/controller/UserController.java",
        "demo",
        "demo",
    )
    qualified = {symbol.qualified_name for symbol in result.symbols}
    assert "com.example.demo.controller.UserController" in qualified
    assert "com.example.demo.controller.UserController.<init>" in qualified
    assert "com.example.demo.controller.UserController.getUser" in qualified
    assert any(symbol.name == "GET /api/users/{id}" for symbol in result.symbols)
    refs = {(ref.reference_name, ref.reference_kind) for ref in result.unresolved_refs}
    assert ("UserService", "injects") in refs
    assert ("UserService.findById", "calls") in refs


def test_extracts_request_mapping_routes_with_commented_annotations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "OrderController.java"
    path.write_text(
        "\n".join(
            [
                "package com.example.order.controller;",
                "",
                "import org.springframework.web.bind.annotation.RequestMapping;",
                "import org.springframework.web.bind.annotation.RequestMethod;",
                "import org.springframework.web.bind.annotation.RestController;",
                "",
                "@RestController",
                '@RequestMapping("order/order")',
                "public class OrderController {",
                "    @RequestMapping(\"/list\")",
                '    //@RequiresPermissions("order:order:list")',
                "    public R list() { return R.ok(); }",
                "",
                "    @RequestMapping(value = \"/submit\", method = RequestMethod.POST)",
                "    public R submit() { return R.ok(); }",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    result = extract_java_file(
        "project:test",
        path,
        "order-service/src/main/java/com/example/order/controller/OrderController.java",
        "order-service",
        "order-service",
    )

    routes = {
        symbol.name: symbol for symbol in result.symbols if symbol.kind == "route"
    }
    assert "ANY /order/order/list" in routes
    assert "POST /order/order/submit" in routes
    assert routes["ANY /order/order/list"].metadata["class_mapping"] == "order/order"
    assert routes["ANY /order/order/list"].metadata["method_mapping"] == "/list"


def test_extracts_annotation_usages_and_values(tmp_path: Path) -> None:
    path = tmp_path / "JobHandler.java"
    path.write_text(
        "\n".join(
            [
                "package com.example.job;",
                "",
                "import com.xxl.job.core.handler.annotation.XxlJob;",
                "",
                '@Component("jobHandler")',
                "public class JobHandler {",
                '    @XXLJob("syncOrderJob")',
                "    @KafkaListener("
                'topics = {"order-topic", "pay-topic"}, groupId = "order-group")',
                "    public void sync("
                '@RequestParam(value = "id", required = false) Long id) {',
                "    }",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    result = extract_java_file(
        "project:test",
        path,
        "job-service/src/main/java/com/example/job/JobHandler.java",
        "job-service",
        "job-service",
    )

    annotation_symbols = [
        symbol for symbol in result.symbols if symbol.kind == "annotation_usage"
    ]
    by_annotation = {
        str(symbol.metadata["annotation"]): symbol for symbol in annotation_symbols
    }

    assert by_annotation["Component"].metadata["values"] == {"value": "jobHandler"}
    assert by_annotation["XXLJob"].name == "@XXLJob syncOrderJob"
    assert by_annotation["XXLJob"].metadata["values"] == {"value": "syncOrderJob"}
    assert by_annotation["XXLJob"].metadata["target_kind"] == "method"
    assert (
        by_annotation["XXLJob"].metadata["target_qualified_name"]
        == "com.example.job.JobHandler.sync"
    )
    assert by_annotation["KafkaListener"].metadata["values"] == {
        "topics": ["order-topic", "pay-topic"],
        "groupId": "order-group",
    }
    assert by_annotation["RequestParam"].metadata["values"] == {
        "value": "id",
        "required": "false",
    }
    assert by_annotation["RequestParam"].metadata["target_kind"] == "parameter"

    annotation_ids = {symbol.id for symbol in annotation_symbols}
    target_ids = {
        symbol.id
        for symbol in result.symbols
        if symbol.qualified_name
        in {
            "com.example.job.JobHandler",
            "com.example.job.JobHandler.sync",
            "com.example.job.JobHandler.sync.id",
        }
    }
    assert any(
        edge.kind == "annotated_by"
        and edge.source_id in target_ids
        and edge.target_id in annotation_ids
        for edge in result.edges
    )
    assert any(
        edge.kind == "annotates"
        and edge.source_id in annotation_ids
        and edge.target_id in target_ids
        for edge in result.edges
    )
