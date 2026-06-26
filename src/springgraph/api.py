"""FastAPI application for project refinement and vector retrieval."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from springgraph.config import get_settings
from springgraph.logging_config import configure_logging
from springgraph.refinement import refine_project
from springgraph.vector_search import (
    VectorSearchError,
    VectorSearchMatch,
    search_project_vectors,
)

logger = logging.getLogger(__name__)


class RefineProjectRequest(BaseModel):
    """Request body for project refinement."""

    project_path: str = Field(..., min_length=1)


class RefineProjectResponse(BaseModel):
    """Response returned after refinement completes."""

    project_id: str
    index_run_id: int
    embedding_job_id: int
    files_seen: int
    symbols_upserted: int
    edges_upserted: int
    chunks_upserted: int
    embeddings_upserted: int
    errors: list[str]


class VectorSearchRequest(BaseModel):
    """Request body for vector-only semantic retrieval."""

    query: str = Field(..., min_length=1)
    project_id: str | None = None
    project_path: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class VectorSearchMatchResponse(BaseModel):
    """One retrieved chunk returned by the vector endpoint."""

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


class VectorSearchResponse(BaseModel):
    """Response returned by vector-only semantic retrieval."""

    query: str
    project_id: str
    embedding_model: str
    embedding_dim: int
    matches: list[VectorSearchMatchResponse]


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    configure_logging(get_settings().log_level)
    app = FastAPI(
        title="springgraph API",
        version="0.1.0",
        description="Refine Java Spring projects and retrieve semantic chunks.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/refine", response_model=RefineProjectResponse)
    def refine_endpoint(request: RefineProjectRequest) -> RefineProjectResponse:
        project_path = _validated_project_path(request.project_path)
        try:
            result = refine_project(project_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Project refinement failed for path: %s", project_path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return RefineProjectResponse(**asdict(result))

    @app.post("/api/vector-search", response_model=VectorSearchResponse)
    async def vector_search_endpoint(
        request: Request,
    ) -> VectorSearchResponse:
        raw_body = await request.body()
        request_model = _parse_vector_search_request(raw_body)
        if request_model.project_id is None and request_model.project_path is None:
            raise HTTPException(
                status_code=400,
                detail="project_id or project_path is required.",
            )
        if (
            request_model.project_id is not None
            and not request_model.project_id.strip()
        ):
            raise HTTPException(
                status_code=400,
                detail="project_id must not be empty when provided.",
            )
        if (
            request_model.project_path is not None
            and not request_model.project_path.strip()
        ):
            raise HTTPException(
                status_code=400,
                detail="project_path must not be empty when provided.",
            )
        try:
            result = search_project_vectors(
                query=request_model.query,
                project_id_value=request_model.project_id,
                project_path=request_model.project_path,
                limit=request_model.limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VectorSearchError as exc:
            logger.exception("Vector search failed for query: %s", request_model.query)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected vector search failure.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return VectorSearchResponse(
            query=result.query,
            project_id=result.project_id,
            embedding_model=result.embedding_model,
            embedding_dim=result.embedding_dim,
            matches=[_match_response(match) for match in result.matches],
        )

    return app


def run() -> None:
    """Run the API with Uvicorn."""
    import uvicorn

    uvicorn.run("springgraph.api:app", host="0.0.0.0", port=8000)


def _validated_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"project_path does not exist: {path}",
        )
    if not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"project_path must be a directory: {path}",
        )
    return path


def _match_response(match: VectorSearchMatch) -> VectorSearchMatchResponse:
    return VectorSearchMatchResponse(**asdict(match))


def _parse_vector_search_request(raw_body: bytes) -> VectorSearchRequest:
    if not raw_body:
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object.",
        )
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object.",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object.",
        )
    try:
        return VectorSearchRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


app = create_app()


if __name__ == "__main__":
    run()
