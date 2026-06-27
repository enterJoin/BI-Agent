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
from springgraph.rag.llm import LlmConfigurationError, LlmInvocationError
from springgraph.rag.schemas import RagAnswer, RagRequest
from springgraph.rag.service import RagRequestError, ask_project
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


class RagAskRequest(BaseModel):
    """Request body for RAG question answering."""

    question: str = Field(..., min_length=1)
    project_id: str | None = None
    project_path: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    graph_depth: int = Field(default=2, ge=0, le=3)
    read_source: bool = True
    mode: str = "agentic"


class RagEvidenceResponse(BaseModel):
    """One evidence item returned by RAG."""

    evidence_type: str
    source: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None
    score: float
    content_excerpt: str | None = None
    metadata: dict[str, Any]


class SourceSnippetResponse(BaseModel):
    """One source snippet returned by RAG."""

    file_path: str
    start_line: int
    end_line: int
    content: str


class RagAskResponse(BaseModel):
    """Response returned by RAG question answering."""

    answer: str
    thread_id: str
    project_id: str
    project_path: str
    intent: str
    rewritten_query: str
    expanded_queries: list[str]
    used_vector_search: bool
    used_relational_search: bool
    used_source_reading: bool
    source_reading_skipped_reason: str | None
    evidence: list[RagEvidenceResponse]
    source_snippets: list[SourceSnippetResponse]
    warnings: list[str]
    mode: str = "agentic"
    used_tools: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


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

    @app.post("/api/rag/ask", response_model=RagAskResponse)
    def rag_ask_endpoint(request: RagAskRequest) -> RagAskResponse:
        try:
            result = ask_project(
                request=RagRequest(
                    question=request.question,
                    project_id=request.project_id,
                    project_path=request.project_path,
                    thread_id=request.thread_id,
                    user_id=request.user_id,
                    top_k=request.top_k,
                    graph_depth=request.graph_depth,
                    read_source=request.read_source,
                    mode=request.mode,
                )
            )
        except RagRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LlmConfigurationError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except LlmInvocationError as exc:
            logger.exception("RAG LLM invocation failed.")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG ask failed for question: %s", request.question)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return _rag_response(result)

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


def _rag_response(answer: RagAnswer) -> RagAskResponse:
    return RagAskResponse(
        answer=answer.answer,
        thread_id=answer.thread_id,
        project_id=answer.project_id,
        project_path=answer.project_path,
        intent=answer.intent,
        rewritten_query=answer.rewritten_query,
        expanded_queries=answer.expanded_queries,
        used_vector_search=answer.used_vector_search,
        used_relational_search=answer.used_relational_search,
        used_source_reading=answer.used_source_reading,
        source_reading_skipped_reason=answer.source_reading_skipped_reason,
        evidence=[
            RagEvidenceResponse(**asdict(item)) for item in answer.evidence
        ],
        source_snippets=[
            SourceSnippetResponse(**asdict(item))
            for item in answer.source_snippets
        ],
        warnings=answer.warnings,
        mode=answer.mode,
        used_tools=answer.used_tools,
        observations=answer.observations,
    )


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
