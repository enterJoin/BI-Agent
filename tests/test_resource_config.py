from pathlib import Path

from springgraph.resource_config import extract_config_file


def test_extracts_resources_from_yaml() -> None:
    root = Path(__file__).parent / "fixtures" / "sample_workspace"
    path = root / "data-ingestion/src/main/resources/application.yml"
    result = extract_config_file(
        "project:test",
        path,
        "data-ingestion/src/main/resources/application.yml",
        "data-ingestion",
        "data-ingestion",
    )
    resources = {
        symbol.metadata.get("normalized_name")
        for symbol in result.symbols
        if symbol.kind == "resource"
    }
    assert "mysql://10.0.0.1:3306/biz" in resources
    assert "redis://10.0.0.2:6379/0" in resources
    assert "rabbitmq://10.0.0.3:5672" in resources
