from pathlib import Path

from springgraph.refinement._embedder import HashEmbedder
from springgraph.refinement._extractors import extract_file_facts
from springgraph.refinement._repository import merge_facts
from springgraph.refinement._scanner import scan_refinement_files

FIXTURES = Path(__file__).parent / "fixtures" / "refinement_workspace"


def test_scans_refinement_files() -> None:
    files = scan_refinement_files(FIXTURES)
    paths = {item.relative_path for item in files}
    assert (
        "order-service/src/main/java/com/example/order/entity/OrderEntity.java"
        in paths
    )
    assert "order-service/src/main/resources/mapper/order/OrderDao.xml" in paths
    assert "auth-service/src/main/resources/templates/login.html" in paths


def test_extracts_table_mapper_sql_and_oauth_facts() -> None:
    facts = []
    for item in scan_refinement_files(FIXTURES):
        facts.append(
            extract_file_facts(
                item.path,
                item.relative_path,
                item.language,
                item.module_name,
                item.service_name,
            )
        )
    merged = merge_facts(facts)
    symbol_names = {symbol.qualified_name for symbol in merged.symbols}
    edge_kinds = {
        (edge.source_key, edge.target_key, edge.kind) for edge in merged.edges
    }
    chunk_types = {chunk.chunk_type for chunk in merged.chunks}

    assert "db_table:oms_order" in symbol_names
    assert "db_column:oms_order.order_sn" in symbol_names
    assert "oauth_provider:Gitee" in symbol_names
    assert any(
        target == "db_table:oms_order" and kind == "reads_table"
        for _, target, kind in edge_kinds
    )
    assert any(
        target == "db_table:oms_order_operate_history" and kind == "writes_table"
        for _, target, kind in edge_kinds
    )
    assert "mybatis_sql" in chunk_types
    assert "third_party_login" in chunk_types
    assert all("should-not-leak" not in chunk.content for chunk in merged.chunks)


def test_extracts_message_topic_and_rocket_tag_facts(tmp_path: Path) -> None:
    source_path = tmp_path / "MessagePublisher.java"
    source_path.write_text(
        "\n".join(
            [
                "class MessagePublisher {",
                "  void send() {",
                "    String topic = BIConstants.MATERIAL_REPORT_TOPIC;",
                "    kafkaService.send(topic, payload);",
                "    rocketMQTemplate.syncSend(\"order-topic:paid\", payload);",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    facts = extract_file_facts(
        source_path,
        "src/main/java/demo/MessagePublisher.java",
        "java",
        "demo",
        "demo",
    )
    symbol_names = {symbol.qualified_name for symbol in facts.symbols}
    edges = {(edge.target_key, edge.kind) for edge in facts.edges}

    assert "mq_topic:BIConstants.MATERIAL_REPORT_TOPIC" in symbol_names
    assert "mq_topic:order-topic" in symbol_names
    assert "mq_tag:order-topic:paid" in symbol_names
    assert (
        "mq_topic:BIConstants.MATERIAL_REPORT_TOPIC",
        "publishes",
    ) in edges
    assert ("mq_topic:order-topic", "publishes") in edges


def test_hash_embedder_returns_1024_dimensions() -> None:
    vector = HashEmbedder().embed("orders read oms_order and oauth provider gitee")
    assert len(vector) == 1024
    assert any(item != 0 for item in vector)
