import json
from typing import Any

from springgraph.config import Settings
from springgraph.refinement import _embedder
from springgraph.refinement._embedder import OpenAICompatibleEmbedder, create_embedder


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
    )

    result = embedder.embed("hello")

    assert result == [0.1, 0.2, 0.3]
    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["body"] == {
        "model": "text-embedding-3-large",
        "input": "hello",
        "dimensions": 3,
    }
    assert captured["timeout"] == 12.0


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
