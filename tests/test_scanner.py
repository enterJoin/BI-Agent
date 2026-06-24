from pathlib import Path

from springgraph.scanner import scan_project

FIXTURES = Path(__file__).parent / "fixtures"


def test_scans_single_project_java_files() -> None:
    files = scan_project(FIXTURES / "sample_springboot")
    java_paths = [item.relative_path for item in files if item.language == "java"]
    assert len(java_paths) == 7
    assert "src/main/java/com/example/demo/controller/UserController.java" in java_paths


def test_scans_workspace_modules_and_configs() -> None:
    files = scan_project(FIXTURES / "sample_workspace")
    paths = {item.relative_path: item for item in files}
    assert "data-ingestion/src/main/resources/application.yml" in paths
    assert "business-api/src/main/resources/application.properties" in paths
    assert (
        paths["data-ingestion/src/main/resources/application.yml"].module_name
        == "data-ingestion"
    )
    assert (
        paths["business-api/src/main/resources/application.properties"].service_name
        == "business-api"
    )
