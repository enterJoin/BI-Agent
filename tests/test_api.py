from pathlib import Path

from fastapi.testclient import TestClient

from springgraph import api
from springgraph.refinement._types import RefinementResult
from springgraph.vector_search import VectorSearchMatch, VectorSearchResult


def test_refine_endpoint_returns_refinement_summary(
    monkeypatch: object, tmp_path: Path
) -> None:
    expected = RefinementResult(
        project_id="project-1",
        index_run_id=101,
        embedding_job_id=202,
        files_seen=3,
        symbols_upserted=4,
        edges_upserted=5,
        chunks_upserted=6,
        embeddings_upserted=7,
        errors=[],
    )
    monkeypatch.setattr(api, "refine_project", lambda _: expected)

    client = TestClient(api.app)
    response = client.post("/api/refine", json={"project_path": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {
        "project_id": "project-1",
        "index_run_id": 101,
        "embedding_job_id": 202,
        "files_seen": 3,
        "symbols_upserted": 4,
        "edges_upserted": 5,
        "chunks_upserted": 6,
        "embeddings_upserted": 7,
        "errors": [],
    }


def test_vector_search_endpoint_returns_chunks(
    monkeypatch: object, tmp_path: Path
) -> None:
    expected = VectorSearchResult(
        query="订单系统和购物车系统有什么交集？",
        project_id="project-1",
        embedding_model="local-hash-embedding-v1",
        embedding_dim=1024,
        matches=[
            VectorSearchMatch(
                chunk_id="chunk-1",
                project_id="project-1",
                file_path="order-service/src/main/java/A.java",
                title="Order Service",
                chunk_type="mybatis_sql",
                content="select * from oms_order",
                score=0.91,
                distance=0.09,
                language="java",
                start_line=10,
                end_line=20,
                module_name="order-service",
                service_name="order-service",
                symbol_qualified_name="com.example.OrderService",
                metadata={"table": "oms_order"},
            )
        ],
    )
    monkeypatch.setattr(api, "search_project_vectors", lambda **_: expected)

    client = TestClient(api.app)
    response = client.post(
        "/api/vector-search",
        json={
            "query": "订单系统和购物车系统有什么交集？",
            "project_path": str(tmp_path),
            "limit": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "订单系统和购物车系统有什么交集？"
    assert payload["project_id"] == "project-1"
    assert payload["matches"][0]["chunk_id"] == "chunk-1"
    assert payload["matches"][0]["metadata"] == {"table": "oms_order"}


def test_vector_search_endpoint_accepts_project_id(monkeypatch: object) -> None:
    expected = VectorSearchResult(
        query="订单系统和购物车系统有什么交集？",
        project_id="project-1",
        embedding_model="local-hash-embedding-v1",
        embedding_dim=1024,
        matches=[],
    )
    monkeypatch.setattr(api, "search_project_vectors", lambda **_: expected)

    client = TestClient(api.app)
    response = client.post(
        "/api/vector-search",
        json={
            "query": "订单系统和购物车系统有什么交集？",
            "project_id": "project-1",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == "project-1"
    assert payload["matches"] == []


def test_vector_search_endpoint_requires_project_identifier() -> None:
    client = TestClient(api.app)
    response = client.post(
        "/api/vector-search",
        json={
            "query": "订单系统和购物车系统有什么交集？",
            "limit": 5,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "project_id or project_path is required."
    }


def test_vector_search_endpoint_accepts_project_id_real_shape(
    monkeypatch: object,
) -> None:
    expected = VectorSearchResult(
        query="订单系统和购物车系统有什么交集？",
        project_id="project:f180a56f5c2d6c1c9d064558ba21cdfa",
        embedding_model="local-hash-embedding-v1",
        embedding_dim=1024,
        matches=[
            VectorSearchMatch(
                chunk_id="chunk-order",
                project_id="project:f180a56f5c2d6c1c9d064558ba21cdfa",
                file_path="guli-order-service/src/main/java/com/example/Order.java",
                title="Order Service",
                chunk_type="mapper_method",
                content="Chunk type: mapper_method",
                score=0.9,
                distance=0.1,
                language="java",
                start_line=10,
                end_line=10,
                module_name="guli-order-service",
                service_name="guli-order-service",
                symbol_qualified_name="mapper:OrderDao",
                metadata={"service": "guli-order-service"},
            )
        ],
    )
    monkeypatch.setattr(api, "search_project_vectors", lambda **_: expected)

    client = TestClient(api.app)
    response = client.post(
        "/api/vector-search",
        json={
            "query": "订单系统和购物车系统有什么交集？",
            "project_id": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == "project:f180a56f5c2d6c1c9d064558ba21cdfa"
    assert payload["matches"][0]["service_name"] == "guli-order-service"
