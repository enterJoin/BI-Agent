"""Vector-only semantic retrieval over refined code chunks."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from springgraph.db import session_scope
from springgraph.hashing import project_id
from springgraph.refinement._embedder import create_embedder

MAX_LIMIT = 100
FALLBACK_CANDIDATE_LIMIT = 1_000


class VectorSearchError(RuntimeError):
    """Raised when vector retrieval cannot be completed."""


@dataclass(frozen=True)
class VectorSearchMatch:
    """One vector-retrieved chunk."""

    chunk_id: str
    project_id: str
    file_path: str
    title: str
    chunk_type: str
    content: str
    score: float
    distance: float
    language: str
    start_line: int | None
    end_line: int | None
    module_name: str | None
    service_name: str | None
    symbol_qualified_name: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    """Vector search response payload before HTTP serialization."""

    query: str
    project_id: str
    embedding_model: str
    embedding_dim: int
    matches: list[VectorSearchMatch]


def search_project_vectors(
    query: str,
    project_id_value: str | None = None,
    project_path: str | Path | None = None,
    limit: int = 10,
) -> VectorSearchResult:
    """Search active chunk embeddings with the configured embedding provider."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    normalized_limit = _normalize_limit(limit)
    embedder = create_embedder()
    query_embedding = embedder.embed(normalized_query)
    resolved_project_id = _resolve_project_id(project_id_value, project_path)

    with session_scope() as session:
        matches = search_vectors_with_session(
            session=session,
            query_embedding=query_embedding,
            embedding_model=embedder.model_name,
            embedding_dim=embedder.dimensions,
            project_id_value=resolved_project_id,
            limit=normalized_limit,
        )

    return VectorSearchResult(
        query=normalized_query,
        project_id=resolved_project_id,
        embedding_model=embedder.model_name,
        embedding_dim=embedder.dimensions,
        matches=matches,
    )


def search_vectors_with_session(
    session: Session,
    query_embedding: list[float],
    embedding_model: str,
    embedding_dim: int,
    project_id_value: str | None = None,
    limit: int = 10,
) -> list[VectorSearchMatch]:
    """Search embeddings in an existing session."""
    normalized_limit = _normalize_limit(limit)
    try:
        return _search_with_pgvector(
            session=session,
            query_embedding=query_embedding,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            project_id_value=project_id_value,
            limit=normalized_limit,
        )
    except SQLAlchemyError:
        session.rollback()
        try:
            return _search_with_python_fallback(
                session=session,
                query_embedding=query_embedding,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                project_id_value=project_id_value,
                limit=normalized_limit,
            )
        except SQLAlchemyError as fallback_exc:
            raise VectorSearchError(
                "Vector search failed. Verify the semantic refinement tables "
                "exist and the database is reachable."
            ) from fallback_exc
        except ValueError as fallback_exc:
            raise VectorSearchError(
                "Vector search failed because stored embeddings could not be parsed."
            ) from fallback_exc


def _search_with_pgvector(
    session: Session,
    query_embedding: list[float],
    embedding_model: str,
    embedding_dim: int,
    project_id_value: str | None,
    limit: int,
) -> list[VectorSearchMatch]:
    project_filter = ""
    params: dict[str, object] = {
        "embedding": _format_vector(query_embedding),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "limit": limit,
    }
    if project_id_value is not None:
        project_filter = "AND e.project_id = :project_id"
        params["project_id"] = project_id_value

    rows = session.execute(
        text(
            f"""
            WITH query_vector AS (
                SELECT CAST(:embedding AS vector) AS embedding
            )
            SELECT
                c.id AS chunk_id,
                e.project_id AS project_id,
                f.path AS file_path,
                c.title AS title,
                c.chunk_type AS chunk_type,
                c.content AS content,
                c.language AS language,
                c.start_line AS start_line,
                c.end_line AS end_line,
                f.module_name AS module_name,
                f.service_name AS service_name,
                s.qualified_name AS symbol_qualified_name,
                c.metadata AS metadata,
                e.embedding <=> query_vector.embedding AS distance
            FROM chunk_embeddings e
            JOIN code_chunks c ON c.id = e.chunk_id
            JOIN files f ON f.id = c.file_id
            LEFT JOIN symbols s ON s.id = c.symbol_id
            CROSS JOIN query_vector
            WHERE e.status = 'active'
              AND c.status = 'active'
              AND e.embedding_model = :embedding_model
              AND e.embedding_dim = :embedding_dim
              {project_filter}
            ORDER BY e.embedding <=> query_vector.embedding
            LIMIT :limit
            """
        ),
        params,
    ).mappings()
    return [_match_from_row(row) for row in rows]


def _search_with_python_fallback(
    session: Session,
    query_embedding: list[float],
    embedding_model: str,
    embedding_dim: int,
    project_id_value: str | None,
    limit: int,
) -> list[VectorSearchMatch]:
    project_filter = ""
    params: dict[str, object] = {
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "candidate_limit": FALLBACK_CANDIDATE_LIMIT,
    }
    if project_id_value is not None:
        project_filter = "AND e.project_id = :project_id"
        params["project_id"] = project_id_value

    rows = list(
        session.execute(
            text(
                f"""
                SELECT
                    c.id AS chunk_id,
                    e.project_id AS project_id,
                    f.path AS file_path,
                    c.title AS title,
                    c.chunk_type AS chunk_type,
                    c.content AS content,
                    c.language AS language,
                    c.start_line AS start_line,
                    c.end_line AS end_line,
                    f.module_name AS module_name,
                    f.service_name AS service_name,
                    s.qualified_name AS symbol_qualified_name,
                    c.metadata AS metadata,
                    e.embedding::text AS embedding_text
                FROM chunk_embeddings e
                JOIN code_chunks c ON c.id = e.chunk_id
                JOIN files f ON f.id = c.file_id
                LEFT JOIN symbols s ON s.id = c.symbol_id
                WHERE e.status = 'active'
                  AND c.status = 'active'
                  AND e.embedding_model = :embedding_model
                  AND e.embedding_dim = :embedding_dim
                  {project_filter}
                LIMIT :candidate_limit
                """
            ),
            params,
        ).mappings()
    )

    scored_rows: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        embedding = _parse_vector(str(row["embedding_text"]))
        similarity = _cosine_similarity(query_embedding, embedding)
        scored_rows.append((similarity, row))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    return [
        _match_from_row(row, distance=1.0 - score)
        for score, row in scored_rows[:limit]
    ]


def _match_from_row(
    row: Mapping[str, Any],
    distance: float | None = None,
) -> VectorSearchMatch:
    resolved_distance = float(row["distance"] if distance is None else distance)
    return VectorSearchMatch(
        chunk_id=str(row["chunk_id"]),
        project_id=str(row["project_id"]),
        file_path=str(row["file_path"]),
        title=str(row["title"]),
        chunk_type=str(row["chunk_type"]),
        content=str(row["content"]),
        score=1.0 - resolved_distance,
        distance=resolved_distance,
        language=str(row["language"]),
        start_line=_optional_int(row.get("start_line")),
        end_line=_optional_int(row.get("end_line")),
        module_name=_optional_str(row.get("module_name")),
        service_name=_optional_str(row.get("service_name")),
        symbol_qualified_name=_optional_str(row.get("symbol_qualified_name")),
        metadata=_metadata(row.get("metadata")),
    )


def _resolve_project_id(
    project_id_value: str | None,
    project_path: str | Path | None,
) -> str:
    if project_id_value is not None:
        normalized_project_id = project_id_value.strip()
        if not normalized_project_id:
            raise ValueError("project_id must not be empty when provided.")
        return normalized_project_id
    if project_path is None:
        raise ValueError("project_id or project_path is required.")
    path = Path(project_path)
    if not str(path).strip():
        raise ValueError("project_path must not be empty.")
    return project_id(path.expanduser().resolve())


def _normalize_limit(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))


def _format_vector(vector: list[float]) -> str:
    return "[" + ",".join(f"{item:.8f}" for item in vector) + "]"


def _parse_vector(raw: str) -> list[float]:
    values = raw.strip().strip("[]")
    if not values:
        return []
    return [float(item) for item in values.split(",")]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(item * item for item in left))
    right_norm = sqrt(sum(item * item for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(
        left_item * right_item
        for left_item, right_item in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _metadata(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}
