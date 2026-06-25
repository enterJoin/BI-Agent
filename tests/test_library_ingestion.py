from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

from springgraph.hashing import project_id
from springgraph.refinement import _library
from springgraph.refinement._embedder import HashEmbedder
from springgraph.refinement._library import (
    extract_library_file_facts,
    scan_library_files,
)
from springgraph.vector_search import search_project_vectors

FIXTURES = Path(__file__).parent / "fixtures" / "sample_workspace"


def test_scans_library_markdown_files() -> None:
    files = scan_library_files(FIXTURES)
    paths = {item.relative_path for item in files}

    assert "library/gulimall-business-knowledge.md" in paths


def test_extracts_parent_and_overlapped_child_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(_library, "TARGET_CHUNK_TOKENS", 5)
    monkeypatch.setattr(_library, "MAX_CHUNK_TOKENS", 6)
    monkeypatch.setattr(_library, "OVERLAP_TOKENS", 2)
    markdown = tmp_path / "checkout.md"
    markdown.write_text(
        "\n".join(
            [
                "# Checkout Knowledge",
                "",
                "order order-service checkout payment",
                "",
                "cart cart-service checkout route",
                "",
                "stock coupon member gateway",
            ]
        ),
        encoding="utf-8",
    )

    facts = extract_library_file_facts(markdown, "library/checkout.md")

    parents = [
        chunk
        for chunk in facts.chunks
        if chunk.metadata.get("chunk_role") == "parent"
    ]
    children = [
        chunk
        for chunk in facts.chunks
        if chunk.metadata.get("chunk_role") == "child"
    ]
    assert len(parents) == 1
    assert len(children) == 3
    assert parents[0].chunk_type == "project_knowledge_parent"
    assert parents[0].metadata["child_count"] == 3
    assert parents[0].metadata["overlap_tokens"] == 2
    assert all(
        child.metadata["parent_doc_id"] == parents[0].metadata["parent_doc_id"]
        for child in children
    )
    assert "Context after:" in children[0].content
    assert "Context before:" in children[1].content


def test_vector_search_can_rank_library_business_knowledge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedder = HashEmbedder()
    project_id_value = project_id(tmp_path)
    library_content = (
        "Chunk type: project_knowledge_parent\n"
        "Source: library/gulimall-business-knowledge.md\n"
        "Heading: 订单与购物车交集\n"
        "Content: order orders order-service cart cart-service checkout "
        "gulimall_order_route gulimall_cart_route oms_order oms_order_item"
    )
    unrelated_content = (
        "Chunk type: project_knowledge_parent\n"
        "Content: product sku catalog user profile oauth provider"
    )
    rows = [
        _vector_row(
            chunk_id="chunk-library",
            project_id=project_id_value,
            file_path="library/gulimall-business-knowledge.md",
            title="Gulimall 业务知识库 > 订单与购物车交集",
            content=library_content,
            embedding=embedder.embed(library_content),
            metadata={
                "source_type": "library",
                "chunk_role": "parent",
                "parent_title": "Gulimall 业务知识库 > 订单与购物车交集",
            },
        ),
        _vector_row(
            chunk_id="chunk-unrelated",
            project_id=project_id_value,
            file_path="library/other.md",
            title="Other",
            content=unrelated_content,
            embedding=embedder.embed(unrelated_content),
            metadata={"source_type": "library", "chunk_role": "parent"},
        ),
    ]
    session = _FallbackVectorSession(rows)

    @contextmanager
    def session_scope() -> Iterator[_FallbackVectorSession]:
        yield session

    monkeypatch.setattr("springgraph.vector_search.create_embedder", lambda: embedder)
    monkeypatch.setattr("springgraph.vector_search.session_scope", session_scope)

    result = search_project_vectors(
        query="订单系统和购物车系统有什么交集？",
        project_path=tmp_path,
        limit=2,
    )

    assert result.project_id == project_id_value
    assert result.matches[0].chunk_id == "chunk-library"
    assert result.matches[0].file_path == "library/gulimall-business-knowledge.md"
    assert result.matches[0].metadata["chunk_role"] == "parent"
    assert session.rolled_back is True


def _vector_row(
    chunk_id: str,
    project_id: str,
    file_path: str,
    title: str,
    content: str,
    embedding: list[float],
    metadata: dict[str, object],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "project_id": project_id,
        "file_path": file_path,
        "title": title,
        "chunk_type": "project_knowledge_parent",
        "content": content,
        "language": "markdown",
        "start_line": 1,
        "end_line": 10,
        "module_name": "library",
        "service_name": None,
        "symbol_qualified_name": None,
        "metadata": metadata,
        "embedding_text": _format_vector(embedding),
    }


def _format_vector(vector: list[float]) -> str:
    return "[" + ",".join(f"{item:.8f}" for item in vector) + "]"


class _FallbackVectorSession:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls = 0
        self.rolled_back = False

    def execute(
        self,
        _statement: object,
        _params: dict[str, object],
    ) -> "_FakeResult":
        self.calls += 1
        if self.calls == 1:
            raise SQLAlchemyError("pgvector unavailable")
        return _FakeResult(self.rows)

    def rollback(self) -> None:
        self.rolled_back = True


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def mappings(self) -> list[dict[str, Any]]:
        return self.rows
