import json
from typing import Any

from springgraph.config import Settings
from springgraph.refinement import _embedder
from springgraph.refinement._embedder import (
    HashEmbedder,
    OpenAICompatibleEmbedder,
    create_embedder,
)


def test_openai_compatible_embedder_sends_dimensions(
    monkeypatch: object,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(_embedder, "urlopen", fake_urlopen)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://example.test/v1/embeddings",
        api_key="test-key",
        model_name="text-embedding-3-large",
        dimensions=3,
        timeout_seconds=12.0,
        batch_size=8,
        max_concurrency=4,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )

    result = embedder.embed("hello")

    assert result == [0.1, 0.2, 0.3]
    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["body"] == {
        "model": "text-embedding-3-large",
        "input": ["hello"],
        "dimensions": 3,
    }
    assert captured["timeout"] == 12.0


def test_openai_compatible_embedder_batches_inputs(
    monkeypatch: object,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(_embedder, "urlopen", fake_urlopen)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://example.test/v1",
        api_key="test-key",
        model_name="text-embedding-3-large",
        dimensions=2,
        timeout_seconds=12.0,
        batch_size=8,
        max_concurrency=4,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )

    result = embedder.embed_batch(["first", "second"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["body"] == {
        "model": "text-embedding-3-large",
        "input": ["first", "second"],
        "dimensions": 2,
    }
    assert captured["timeout"] == 12.0


def test_openai_compatible_embedder_retries_connection_reset(
    monkeypatch: object,
) -> None:
    attempts = 0

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionResetError("reset")
        return FakeResponse()

    monkeypatch.setattr(_embedder, "urlopen", fake_urlopen)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://example.test/v1",
        api_key="test-key",
        model_name="text-embedding-3-large",
        dimensions=2,
        timeout_seconds=12.0,
        batch_size=8,
        max_concurrency=4,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )

    result = embedder.embed("hello")

    assert result == [0.1, 0.2]
    assert attempts == 2


def test_hash_embedder_batches_inputs() -> None:
    embedder = HashEmbedder(dimensions=4)

    result = embedder.embed_batch(["orders", "tables"])

    assert result == [embedder.embed("orders"), embedder.embed("tables")]


def test_create_embedder_accepts_apirouter_provider() -> None:
    settings = Settings(
        embedding_provider="apirouter",
        embedding_model="text-embedding-3-large",
        embedding_dim=1024,
        embedding_base_url="https://example.test/v1",
        embedding_api_key="test-key",
    )

    embedder = create_embedder(settings)

    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model_name == "text-embedding-3-large"
    assert embedder.dimensions == 1024
    assert embedder.batch_size == 8
    assert embedder.max_concurrency == 4
