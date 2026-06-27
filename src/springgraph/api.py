"""FastAPI application for project refinement and vector retrieval."""

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field, ValidationError
from sqlalchemy import desc, select

from springgraph.config import get_settings
from springgraph.db import session_scope
from springgraph.logging_config import configure_logging
from springgraph.models import Project
from springgraph.rag.llm import LlmConfigurationError, LlmInvocationError
from springgraph.rag.memory.store import (
    RagMessageRecord,
    RagThreadRecord,
    create_chat_thread,
    delete_chat_thread,
    ensure_chat_thread,
    list_chat_messages,
    list_chat_threads,
    update_chat_thread_title,
)
from springgraph.rag.schemas import RagAnswer, RagRequest, RagStreamEvent
from springgraph.rag.service import RagRequestError, ask_project, stream_ask_project
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


class ProjectResponse(BaseModel):
    """Indexed project returned by project APIs."""

    id: str
    root_path: str
    name: str
    created_at: str
    updated_at: str


class VectorSearchRequest(BaseModel):
    """Request body for vector-only semantic retrieval."""

    query: str = Field(..., min_length=1)
    project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("project_id", "projectId"),
    )
    project_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("project_path", "projectPath"),
    )
    user_id: str | None = Field(
        default="1",
        validation_alias=AliasChoices("user_id", "userId"),
    )
    thread_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
    )
    title: str | None = None
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
    user_id: str | None = None
    thread_id: str | None = None
    title: str | None = None
    embedding_model: str
    embedding_dim: int
    matches: list[VectorSearchMatchResponse]


class RagAskRequest(BaseModel):
    """Request body for RAG question answering."""

    question: str = Field(..., min_length=1)
    project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("project_id", "projectId"),
    )
    project_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("project_path", "projectPath"),
    )
    thread_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
    )
    user_id: str | None = Field(
        default="1",
        validation_alias=AliasChoices("user_id", "userId"),
    )
    title: str | None = None
    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        validation_alias=AliasChoices("top_k", "topK"),
    )
    graph_depth: int = Field(
        default=2,
        ge=0,
        le=3,
        validation_alias=AliasChoices("graph_depth", "graphDepth"),
    )
    read_source: bool = Field(
        default=True,
        validation_alias=AliasChoices("read_source", "readSource"),
    )
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


class RagThreadCreateRequest(BaseModel):
    """Request body for creating a RAG chat thread."""

    project_id: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("project_id", "projectId"),
    )
    user_id: str | None = Field(
        default="1",
        validation_alias=AliasChoices("user_id", "userId"),
    )
    thread_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
    )
    title: str | None = None
    first_question: str | None = Field(
        default=None,
        validation_alias=AliasChoices("first_question", "firstQuestion"),
    )


class RagThreadUpdateTitleRequest(BaseModel):
    """Request body for updating a RAG chat thread title."""

    project_id: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("project_id", "projectId"),
    )
    user_id: str | None = Field(
        default="1",
        validation_alias=AliasChoices("user_id", "userId"),
    )
    title: str = Field(..., min_length=1)


class RagThreadResponse(BaseModel):
    """Persisted RAG chat thread returned by APIs."""

    id: int
    project_id: str
    user_id: str
    thread_id: str
    title: str
    created_at: str
    updated_at: str


class RagMessageResponse(BaseModel):
    """Persisted RAG chat message returned by APIs."""

    id: int
    thread_id: str
    role: str
    content: str
    created_at: str


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

    @app.get("/api/projects", response_model=list[ProjectResponse])
    def list_projects_endpoint() -> list[ProjectResponse]:
        try:
            projects = _list_project_records()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Project list API failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return [_project_response(project) for project in projects]

    @app.get("/api/projects/{project_id}", response_model=ProjectResponse)
    def get_project_endpoint(project_id: str) -> ProjectResponse:
        try:
            project = _get_project_record(project_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Project detail API failed: project_id=%s", project_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if project is None:
            raise HTTPException(
                status_code=404,
                detail=f"project_id was not found: {project_id}",
            )
        return _project_response(project)

    @app.post("/api/refine", response_model=RefineProjectResponse)
    def refine_endpoint(request: RefineProjectRequest) -> RefineProjectResponse:
        project_path = _validated_project_path(request.project_path)
        started_at = perf_counter()
        logger.info("Refine API request started: project_path=%s", project_path)
        try:
            result = refine_project(project_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Project refinement failed for path: %s", project_path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.info(
            "Refine API request completed: project_id=%s, files_seen=%s, "
            "symbols_upserted=%s, edges_upserted=%s, chunks_upserted=%s, "
            "embeddings_upserted=%s, elapsed_seconds=%.3f",
            result.project_id,
            result.files_seen,
            result.symbols_upserted,
            result.edges_upserted,
            result.chunks_upserted,
            result.embeddings_upserted,
            perf_counter() - started_at,
        )

        return RefineProjectResponse(**asdict(result))

    @app.post("/api/vector_search", response_model=VectorSearchResponse)
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

        thread_id = _resolved_thread_id(request_model.thread_id)
        title = request_model.title or request_model.query
        try:
            ensure_chat_thread(
                project_id=result.project_id,
                user_id=request_model.user_id,
                thread_id=thread_id,
                title=title,
            )
        except ValueError as exc:
            raise _chat_http_exception(exc) from exc

        return VectorSearchResponse(
            query=result.query,
            project_id=result.project_id,
            user_id=request_model.user_id,
            thread_id=thread_id,
            title=title,
            embedding_model=result.embedding_model,
            embedding_dim=result.embedding_dim,
            matches=[_match_response(match) for match in result.matches],
        )

    @app.post("/api/rag/ask", response_model=RagAskResponse)
    def rag_ask_endpoint(request: RagAskRequest) -> RagAskResponse:
        try:
            result = ask_project(request=_rag_request_from_api(request))
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

    @app.post("/api/rag/ask/stream")
    def rag_ask_stream_endpoint(request: RagAskRequest) -> StreamingResponse:
        rag_request = _rag_request_from_api(request)
        return StreamingResponse(
            _sse_rag_events(rag_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/rag/threads", response_model=RagThreadResponse)
    def create_rag_thread_endpoint(
        request: RagThreadCreateRequest,
    ) -> RagThreadResponse:
        try:
            title = request.title or request.first_question
            thread = create_chat_thread(
                project_id=request.project_id,
                user_id=request.user_id,
                title=title,
                thread_id=request.thread_id,
            )
        except ValueError as exc:
            raise _chat_http_exception(exc) from exc
        return _thread_response(thread)

    @app.get("/api/rag/threads", response_model=list[RagThreadResponse])
    def list_rag_threads_endpoint(
        request: Request,
    ) -> list[RagThreadResponse]:
        project_id = _required_query_param(request, "projectId", "project_id")
        user_id = _optional_query_param(request, "userId", "user_id")
        try:
            threads = list_chat_threads(project_id=project_id, user_id=user_id)
        except ValueError as exc:
            raise _chat_http_exception(exc) from exc
        return [_thread_response(thread) for thread in threads]

    @app.get(
        "/api/rag/threads/{thread_id}/messages",
        response_model=list[RagMessageResponse],
    )
    def list_rag_messages_endpoint(
        thread_id: str,
        request: Request,
    ) -> list[RagMessageResponse]:
        project_id = _required_query_param(request, "projectId", "project_id")
        user_id = _optional_query_param(request, "userId", "user_id")
        try:
            messages = list_chat_messages(
                project_id=project_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        except ValueError as exc:
            raise _chat_http_exception(exc) from exc
        return [_message_response(message) for message in messages]

    @app.delete("/api/rag/threads/{thread_id}")
    def delete_rag_thread_endpoint(
        thread_id: str,
        request: Request,
    ) -> dict[str, str]:
        project_id = _required_query_param(request, "projectId", "project_id")
        user_id = _optional_query_param(request, "userId", "user_id")
        try:
            delete_chat_thread(
                project_id=project_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        except ValueError as exc:
            raise _chat_http_exception(exc) from exc
        return {"status": "deleted"}

    @app.patch(
        "/api/rag/threads/{thread_id}/title",
        response_model=RagThreadResponse,
    )
    def update_rag_thread_title_endpoint(
        thread_id: str,
        request: RagThreadUpdateTitleRequest,
    ) -> RagThreadResponse:
        try:
            thread = update_chat_thread_title(
                project_id=request.project_id,
                user_id=request.user_id,
                thread_id=thread_id,
                title=request.title,
            )
        except ValueError as exc:
            raise _chat_http_exception(exc) from exc
        return _thread_response(thread)

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


def _list_project_records() -> list[Project]:
    with session_scope() as session:
        rows = session.execute(
            select(Project).order_by(desc(Project.updated_at), desc(Project.created_at))
        ).scalars()
        return list(rows)


def _get_project_record(project_id: str) -> Project | None:
    normalized_project_id = project_id.strip()
    if not normalized_project_id:
        raise HTTPException(status_code=400, detail="project_id must not be empty.")
    with session_scope() as session:
        return session.get(Project, normalized_project_id)


def _project_response(project: Project) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        root_path=project.root_path,
        name=project.name,
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat(),
    )


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


def _thread_response(thread: RagThreadRecord) -> RagThreadResponse:
    return RagThreadResponse(
        id=thread.id,
        project_id=thread.project_id,
        user_id=thread.user_id,
        thread_id=thread.thread_id,
        title=thread.title,
        created_at=thread.created_at.isoformat(),
        updated_at=thread.updated_at.isoformat(),
    )


def _message_response(message: RagMessageRecord) -> RagMessageResponse:
    return RagMessageResponse(
        id=message.id,
        thread_id=message.thread_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at.isoformat(),
    )


def _chat_http_exception(exc: ValueError) -> HTTPException:
    message = str(exc)
    status_code = 404 if "not found" in message else 400
    return HTTPException(status_code=status_code, detail=message)


def _resolved_thread_id(thread_id: str | None) -> str:
    normalized = thread_id.strip() if thread_id else ""
    if normalized:
        return normalized
    return f"thread:{uuid4().hex}"


def _required_query_param(request: Request, *names: str) -> str:
    value = _optional_query_param(request, *names)
    if value is None:
        raise HTTPException(status_code=422, detail=f"{names[0]} is required.")
    return value


def _optional_query_param(request: Request, *names: str) -> str | None:
    for name in names:
        value = request.query_params.get(name)
        if value is not None and value.strip():
            return value
    return None


def _rag_request_from_api(request: RagAskRequest) -> RagRequest:
    return RagRequest(
        question=request.question,
        project_id=request.project_id,
        project_path=request.project_path,
        thread_id=request.thread_id,
        user_id=request.user_id,
        title=request.title,
        top_k=request.top_k,
        graph_depth=request.graph_depth,
        read_source=request.read_source,
        mode=request.mode,
    )


def _sse_rag_events(request: RagRequest) -> Iterator[str]:
    try:
        for event in stream_ask_project(request):
            yield _format_sse(event)
        yield _format_sse(RagStreamEvent(event="done", data={}))
    except RagRequestError as exc:
        yield _format_sse(
            RagStreamEvent(
                event="error",
                data={"status_code": 400, "detail": str(exc)},
            )
        )
    except LlmConfigurationError as exc:
        yield _format_sse(
            RagStreamEvent(
                event="error",
                data={"status_code": 500, "detail": str(exc)},
            )
        )
    except LlmInvocationError as exc:
        logger.exception("Streaming RAG LLM invocation failed.")
        yield _format_sse(
            RagStreamEvent(
                event="error",
                data={"status_code": 502, "detail": str(exc)},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming RAG ask failed.")
        yield _format_sse(
            RagStreamEvent(
                event="error",
                data={"status_code": 500, "detail": str(exc)},
            )
        )


def _format_sse(event: RagStreamEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


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
