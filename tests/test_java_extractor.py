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
