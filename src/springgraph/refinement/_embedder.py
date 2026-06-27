"""Embedding providers for refinement chunks."""

import json
from hashlib import sha256
from math import sqrt
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from springgraph.config import Settings, get_settings

DEFAULT_EMBEDDING_DIMENSIONS = 1024


class Embedder(Protocol):
    """Embedding provider contract used by the refinement repository."""

    model_name: str
    dimensions: int

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for text."""


class HashEmbedder:
    """Small deterministic embedder used when no external model is configured."""

    def __init__(self, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.model_name = "local-hash-embedding-v1"

    def embed(self, text: str) -> list[float]:
        """Return a normalized deterministic embedding for text."""
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            token_digest = sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(token_digest[:4], "big") % self.dimensions
            sign = 1.0 if token_digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = sqrt(sum(item * item for item in vector))
        if norm == 0:
            return vector
        return [item / norm for item in vector]


class OpenAICompatibleEmbedder:
    """Embedding provider for OpenAI-compatible /v1/embeddings APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        dimensions: int,
        timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector from the configured remote provider."""
        body = {
            "model": self.model_name,
            "input": text,
            "dimensions": self.dimensions,
        }
        request = Request(
            self._endpoint_url(),
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Embedding provider returned HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Embedding provider request failed: {exc}") from exc

        vector = payload.get("data", [{}])[0].get("embedding")
        if not isinstance(vector, list):
            raise RuntimeError("Embedding provider response did not contain a vector.")
        result = [float(item) for item in vector]
        if len(result) != self.dimensions:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected {self.dimensions}, got {len(result)}."
            )
        return result

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _endpoint_url(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"


def create_embedder(settings: Settings | None = None) -> Embedder:
    """Create the configured embedding provider."""
    resolved = settings or get_settings()
    provider = resolved.embedding_provider.strip().lower()
    if provider in {"local", "hash", "local_hash"}:
        return HashEmbedder(dimensions=resolved.embedding_dim)
    if provider in {
        "openai",
        "openai_compatible",
        "openai-compatible",
        "compatible",
        "apirouter",
    }:
        return OpenAICompatibleEmbedder(
            base_url=resolved.embedding_base_url,
            api_key=resolved.embedding_api_key,
            model_name=resolved.embedding_model,
            dimensions=resolved.embedding_dim,
            timeout_seconds=resolved.embedding_timeout_seconds,
        )
    raise ValueError(f"Unsupported embedding provider: {resolved.embedding_provider}")


def _tokens(text: str) -> list[str]:
    token: list[str] = []
    cjk_token: list[str] = []
    tokens: list[str] = []

    def flush_word() -> None:
        nonlocal token
        if token:
            tokens.append("".join(token))
            token = []

    def flush_cjk() -> None:
        nonlocal cjk_token
        if not cjk_token:
            return
        tokens.extend(cjk_token)
        tokens.extend(
            "".join(cjk_token[index : index + 2])
            for index in range(len(cjk_token) - 1)
        )
        cjk_token = []

    for char in text.lower():
        if _is_cjk(char):
            flush_word()
            cjk_token.append(char)
            continue
        flush_cjk()
        if char.isalnum() or char in {"_", "-", ".", "/", ":"}:
            token.append(char)
        else:
            flush_word()
    flush_word()
    flush_cjk()
    return tokens


def _is_cjk(char: str) -> bool:
    return (
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )
