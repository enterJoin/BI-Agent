from pathlib import Path

from fastapi.testclient import TestClient

from springgraph import api
from springgraph.rag.schemas import (
    RagAnswer,
    RagEvidence,
    RagStreamEvent,
    SourceSnippet,
)
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


def test_vector_search_endpoint_accepts_json_string_body(
    monkeypatch: object,
) -> None:
    expected = VectorSearchResult(
        query="订单系统都用到了哪些表？",
        project_id="project-1",
        embedding_model="local-hash-embedding-v1",
        embedding_dim=1024,
        matches=[],
    )
    monkeypatch.setattr(api, "search_project_vectors", lambda **_: expected)

    client = TestClient(api.app)
    response = client.post(
        "/api/vector-search",
        content=(
            '{"query":"订单系统都用到了哪些表？",'
            '"project_id":"project-1","limit":10}'
        ).encode(),
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )

    assert response.status_code == 200
    assert response.json()["project_id"] == "project-1"


def test_rag_ask_endpoint_uses_request_defaults(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_ask_project(request: object) -> RagAnswer:
        captured["request"] = request
        return RagAnswer(
            answer="问题意图识别为 route_lookup。",
            thread_id="thread-1",
            project_id="project-1",
            project_path=str(tmp_path),
            intent="route_lookup",
            rewritten_query="订单创建接口",
            expanded_queries=["订单创建接口"],
            used_vector_search=True,
            used_relational_search=True,
            used_source_reading=True,
            source_reading_skipped_reason=None,
            evidence=[
                RagEvidence(
                    evidence_type="route",
                    source="relational",
                    file_path="src/main/java/OrderController.java",
                    start_line=10,
                    end_line=20,
                    symbol="OrderController.create",
                    score=1.0,
                    content_excerpt="@PostMapping",
                    metadata={"kind": "route"},
                )
            ],
            source_snippets=[
                SourceSnippet(
                    file_path="src/main/java/OrderController.java",
                    start_line=10,
                    end_line=20,
                    content="@PostMapping",
                )
            ],
            warnings=[],
        )

    monkeypatch.setattr(api, "ask_project", fake_ask_project)

    client = TestClient(api.app)
    response = client.post(
        "/api/rag/ask",
        json={
            "question": "订单创建接口在哪里实现",
            "project_path": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "route_lookup"
    assert payload["evidence"][0]["file_path"] == "src/main/java/OrderController.java"
    request = captured["request"]
    assert request.top_k == 8  # type: ignore[attr-defined]
    assert request.graph_depth == 2  # type: ignore[attr-defined]
    assert request.read_source is True  # type: ignore[attr-defined]


def test_rag_ask_endpoint_accepts_project_id(
    monkeypatch: object,
) -> None:
    captured: dict[str, object] = {}

    def fake_ask_project(request: object) -> RagAnswer:
        captured["request"] = request
        return RagAnswer(
            answer="命中项目。",
            thread_id="thread-1",
            project_id="project-1",
            project_path="F:\\demo",
            intent="semantic_qna",
            rewritten_query="项目",
            expanded_queries=["项目"],
            used_vector_search=True,
            used_relational_search=True,
            used_source_reading=False,
            source_reading_skipped_reason="project_path_not_found",
            evidence=[],
            source_snippets=[],
            warnings=[],
        )

    monkeypatch.setattr(api, "ask_project", fake_ask_project)

    client = TestClient(api.app)
    response = client.post(
        "/api/rag/ask",
        json={
            "question": "这个项目有哪些接口？",
            "project_id": "project-1",
            "project_path": "F:\\should-be-ignored-by-service",
        },
    )

    assert response.status_code == 200
    request = captured["request"]
    assert request.project_id == "project-1"  # type: ignore[attr-defined]
    assert request.project_path == "F:\\should-be-ignored-by-service"  # type: ignore[attr-defined]


def test_rag_ask_endpoint_accepts_agentic_mode(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_ask_project(request: object) -> RagAnswer:
        captured["request"] = request
        return RagAnswer(
            answer="agentic",
            thread_id="thread-1",
            project_id="project-1",
            project_path=str(tmp_path),
            intent="追踪数据入库链路",
            rewritten_query="素材怎么入库",
            expanded_queries=["素材怎么入库"],
            used_vector_search=True,
            used_relational_search=True,
            used_source_reading=False,
            source_reading_skipped_reason="project_path_not_found",
            evidence=[],
            source_snippets=[],
            warnings=[],
            mode="agentic",
            used_tools=["artifact_search"],
            observations=["Artifact search returned 0 evidence items."],
        )

    monkeypatch.setattr(api, "ask_project", fake_ask_project)

    client = TestClient(api.app)
    response = client.post(
        "/api/rag/ask",
        json={
            "question": "素材怎么入库",
            "project_path": str(tmp_path),
            "mode": "agentic",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "agentic"
    assert payload["used_tools"] == ["artifact_search"]
    request = captured["request"]
    assert request.mode == "agentic"  # type: ignore[attr-defined]


def test_rag_ask_stream_endpoint_returns_sse_events(
    monkeypatch: object,
) -> None:
    captured: dict[str, object] = {}

    def fake_stream_ask_project(request: object) -> list[RagStreamEvent]:
        captured["request"] = request
        return [
            RagStreamEvent(
                event="status",
                data={"stage": "planning", "message": "Creating plan."},
            ),
            RagStreamEvent(
                event="final",
                data={"answer": {"answer": "ok", "used_tools": []}},
            ),
        ]

    monkeypatch.setattr(api, "stream_ask_project", fake_stream_ask_project)

    client = TestClient(api.app)
    response = client.post(
        "/api/rag/ask/stream",
        json={
            "question": "订单服务都用到了哪些表？",
            "project_id": "project-1",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert (
        'event: status\ndata: {"stage": "planning", "message": "Creating plan."}'
        in response.text
    )
    assert (
        'event: final\ndata: {"answer": {"answer": "ok", "used_tools": []}}'
        in response.text
    )
    assert "event: done\ndata: {}" in response.text
    request = captured["request"]
    assert request.project_id == "project-1"  # type: ignore[attr-defined]


def test_rag_ask_endpoint_requires_project_identifier() -> None:
    client = TestClient(api.app)
    response = client.post(
        "/api/rag/ask",
        json={"question": "这个项目有哪些接口？"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "project_id or project_path is required."
    }
