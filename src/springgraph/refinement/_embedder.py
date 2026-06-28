"""Embedding providers for refinement chunks."""

import json
from hashlib import sha256
from math import sqrt
from time import sleep
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from springgraph.config import Settings, get_settings

DEFAULT_EMBEDDING_DIMENSIONS = 1024
TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class Embedder(Protocol):
    """Embedding provider contract used by the refinement repository."""

    model_name: str
    dimensions: int
    batch_size: int
    max_concurrency: int

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for texts in the same order."""


class HashEmbedder:
    """Small deterministic embedder used when no external model is configured."""

    def __init__(
        self,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        batch_size: int = 256,
    ) -> None:
        self.dimensions = dimensions
        self.model_name = "local-hash-embedding-v1"
        self.batch_size = max(1, batch_size)
        self.max_concurrency = 1

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

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings for texts."""
        return [self.embed(text) for text in texts]


class OpenAICompatibleEmbedder:
    """Embedding provider for OpenAI-compatible /v1/embeddings APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        dimensions: int,
        timeout_seconds: float,
        batch_size: int,
        max_concurrency: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, batch_size)
        self.max_concurrency = max(1, max_concurrency)
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector from the configured remote provider."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors from the configured remote provider."""
        if not texts:
            return []
        body = {
            "model": self.model_name,
            "input": texts,
            "dimensions": self.dimensions,
        }
        payload = self._post_embeddings(body)
        return self._parse_vectors(payload, len(texts))

    def _post_embeddings(self, body: dict[str, object]) -> dict[str, object]:
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            request = Request(
                self._endpoint_url(),
                data=json.dumps(body).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return cast_json_object(response.read())
            except HTTPError as exc:
                if not self._should_retry_http(exc, attempt, attempts):
                    details = exc.read().decode("utf-8", errors="ignore")
                    raise RuntimeError(
                        f"Embedding provider returned HTTP {exc.code}: {details}"
                    ) from exc
                last_error = exc
            except (
                ConnectionError,
                ConnectionResetError,
                OSError,
                TimeoutError,
                URLError,
            ) as exc:
                if attempt >= attempts - 1:
                    raise RuntimeError(
                        f"Embedding provider request failed after {attempts} "
                        f"attempts: {exc}"
                    ) from exc
                last_error = exc
            self._sleep_before_retry(attempt)
        raise RuntimeError(
            f"Embedding provider request failed after {attempts} attempts: "
            f"{last_error}"
        )

    def _parse_vectors(
        self,
        payload: dict[str, object],
        expected_count: int,
    ) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding provider response did not contain data.")
        items = sorted(data, key=_embedding_index)
        if len(items) != expected_count:
            raise RuntimeError(
                "Embedding response count mismatch: "
                f"expected {expected_count}, got {len(items)}."
            )
        return [self._parse_vector(item) for item in items]

    def _parse_vector(self, item: object) -> list[float]:
        if not isinstance(item, dict):
            raise RuntimeError("Embedding provider response item was not an object.")
        vector = item.get("embedding")
        if not isinstance(vector, list):
            raise RuntimeError("Embedding provider response did not contain a vector.")
        result = [float(value) for value in vector]
        if len(result) != self.dimensions:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected {self.dimensions}, got {len(result)}."
            )
        return result

    def _should_retry_http(
        self,
        exc: HTTPError,
        attempt: int,
        attempts: int,
    ) -> bool:
        return attempt < attempts - 1 and exc.code in TRANSIENT_HTTP_STATUS_CODES

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        sleep(self.retry_backoff_seconds * (2**attempt))

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
            batch_size=resolved.embedding_batch_size,
            max_concurrency=resolved.embedding_max_concurrency,
            max_retries=resolved.embedding_max_retries,
            retry_backoff_seconds=resolved.embedding_retry_backoff_seconds,
        )
    raise ValueError(f"Unsupported embedding provider: {resolved.embedding_provider}")


def cast_json_object(raw: bytes) -> dict[str, object]:
    """Decode a provider JSON response as an object."""
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Embedding provider response was not a JSON object.")
    return payload


def _embedding_index(item: object) -> int:
    if not isinstance(item, dict):
        return 0
    index = item.get("index")
    return int(index) if isinstance(index, int | str) else 0


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
