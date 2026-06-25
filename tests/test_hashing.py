from pathlib import Path

from springgraph.hashing import project_id, route_symbol_id, symbol_id
from springgraph.refinement._embedder import HashEmbedder


def test_ids_are_stable() -> None:
    root = Path(__file__).parent
    assert project_id(root) == project_id(root)


def test_different_symbols_get_different_ids() -> None:
    first = symbol_id("project:a", "A.java", "class", "a.A", 1)
    second = symbol_id("project:a", "A.java", "class", "a.B", 1)
    assert first != second


def test_route_id_is_sensitive() -> None:
    first = route_symbol_id("project:a", "GET", "/a", "A.get")
    second = route_symbol_id("project:a", "POST", "/a", "A.get")
    assert first != second


def test_hash_embedder_handles_cjk_queries() -> None:
    vector = HashEmbedder().embed("订单系统和购物车系统有什么交集")
    assert len(vector) == 1024
    assert any(item != 0 for item in vector)
