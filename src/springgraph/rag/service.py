"""Public RAG service entry point."""

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from uuid import uuid4

from springgraph.db import session_scope
from springgraph.hashing import project_id
from springgraph.models import Project
from springgraph.rag.agent import nodes
from springgraph.rag.agent.graph import build_agentic_rag_graph
from springgraph.rag.agent.state import AgenticRagState
from springgraph.rag.memory.store import ensure_chat_thread
from springgraph.rag.schemas import (
    RagAnswer,
    RagEvidence,
    RagRequest,
    RagStreamEvent,
    SourceSnippet,
)


class RagRequestError(ValueError):
    """Raised when a RAG request is invalid."""


_PROJECT_CACHE_TTL_SECONDS = 300.0
_PROJECT_PATH_CACHE: dict[str, tuple[float, str]] = {}


def ask_project(request: RagRequest) -> RagAnswer:
    """Answer a code question using the Agentic RAG flow."""
    question = request.question.strip()
    if not question:
        raise RagRequestError("question must not be empty.")

    thread_id = request.thread_id or f"thread:{uuid4().hex}"
    project_id_value, project_path = _resolve_project_reference(request)
    mode = request.mode.strip().lower()
    if mode != "agentic":
        raise RagRequestError(f"Unsupported RAG mode: {request.mode}")
    _ensure_chat_thread_if_project_exists(
        project_id=project_id_value,
        user_id=request.user_id,
        thread_id=thread_id,
        title=request.title or question,
    )
    return _ask_project_agentic(
        request=request,
        question=question,
        thread_id=thread_id,
        project_id_value=project_id_value,
        project_path=project_path,
    )


def _ask_project_agentic(
    request: RagRequest,
    question: str,
    thread_id: str,
    project_id_value: str,
    project_path: Path,
) -> RagAnswer:
    """Answer a code question using controlled Agentic RAG."""
    graph = build_agentic_rag_graph()
    final_state = graph.invoke(
        _initial_state(
            request=request,
            question=question,
            thread_id=thread_id,
            project_id_value=project_id_value,
            project_path=project_path,
        )
    )
    return _answer_from_state(
        final_state=final_state,
        question=question,
        thread_id=thread_id,
        project_id_value=project_id_value,
        project_path=project_path,
    )


def stream_ask_project(request: RagRequest) -> Iterator[RagStreamEvent]:
    """Stream a code question using the planned Agentic RAG flow."""
    question = request.question.strip()
    if not question:
        raise RagRequestError("question must not be empty.")

    thread_id = request.thread_id or f"thread:{uuid4().hex}"
    project_id_value, project_path = _resolve_project_reference(request)
    mode = request.mode.strip().lower()
    if mode != "agentic":
        raise RagRequestError(f"Unsupported RAG mode: {request.mode}")
    _ensure_chat_thread_if_project_exists(
        project_id=project_id_value,
        user_id=request.user_id,
        thread_id=thread_id,
        title=request.title or question,
    )

    state = _initial_state(
        request=request,
        question=question,
        thread_id=thread_id,
        project_id_value=project_id_value,
        project_path=project_path,
    )
    yield _event(
        "status",
        stage="request_resolved",
        message="RAG request resolved.",
        project_id=project_id_value,
        thread_id=thread_id,
    )

    state = nodes.load_runtime_config(state)
    state = nodes.apply_request_defaults(state)
    state = nodes.load_thread_memory(state)
    state = nodes.resolve_query_context(state)
    yield _event(
        "query_resolution",
        query_resolution=state.get("query_resolution", {}),
        contextual_question=state.get("contextual_question", question),
    )
    yield _event(
        "status",
        stage="planning",
        message="Creating retrieval plan.",
    )

    state = nodes.plan_retrieval(state)
    yield _event(
        "plan",
        query_resolution=state.get("query_resolution", {}),
        contextual_question=state.get("contextual_question", question),
        question_understanding=state.get("question_understanding", {}),
        retrieval_plan=state.get("retrieval_plan", {}),
    )

    yield _event(
        "status",
        stage="retrieval",
        message="Executing retrieval tools.",
    )
    state = nodes.execute_retrieval_plan(state)
    for result in state.get("tool_results", []):
        yield _event(
            "tool_result",
            tool_name=result.tool_name,
            summary=result.summary,
            evidence_count=len(result.evidence),
            source_snippet_count=len(result.source_snippets),
            warnings=result.warnings,
        )
    yield _event(
        "evidence",
        evidence_count=len(state.get("evidence", [])),
        source_snippet_count=len(state.get("source_snippets", [])),
        used_tools=list(state.get("used_tools", [])),
        observations=list(state.get("observations", [])),
    )

    yield _event(
        "status",
        stage="answering",
        message="Generating final answer.",
    )
    state = nodes.generate_final_answer(state)
    state = nodes.persist_turn_memory(state)
    answer = _answer_from_state(
        final_state=state,
        question=question,
        thread_id=thread_id,
        project_id_value=project_id_value,
        project_path=project_path,
    )
    yield _event("final", answer=asdict(answer))


def _initial_state(
    request: RagRequest,
    question: str,
    thread_id: str,
    project_id_value: str,
    project_path: Path,
) -> AgenticRagState:
    return {
        "thread_id": thread_id,
        "load_memory": request.thread_id is not None,
        "user_id": request.user_id,
        "title": request.title,
        "project_path": str(project_path),
        "project_id": project_id_value,
        "question": question,
        "top_k": request.top_k,
        "graph_depth": request.graph_depth,
        "read_source": request.read_source,
        "warnings": [],
    }


def _event(event: str, **data: object) -> RagStreamEvent:
    return RagStreamEvent(event=event, data=data)


def _ensure_chat_thread_if_project_exists(
    project_id: str,
    user_id: str | None,
    thread_id: str,
    title: str,
) -> None:
    try:
        ensure_chat_thread(
            project_id=project_id,
            user_id=user_id,
            thread_id=thread_id,
            title=title,
        )
    except ValueError as exc:
        if "project_id was not found" not in str(exc):
            raise


def _answer_from_state(
    final_state: AgenticRagState,
    question: str,
    thread_id: str,
    project_id_value: str,
    project_path: Path,
) -> RagAnswer:
    understanding = final_state.get("question_understanding", {})
    intent = str(understanding.get("task_goal", "agentic_rag"))
    rewritten_query = str(
        final_state.get("contextual_question")
        or understanding.get("rewritten_query")
        or question
    )
    expanded_queries = _agentic_expanded_queries(
        question=question,
        understanding=understanding,
        rewritten_query=rewritten_query,
    )
    return RagAnswer(
        answer=final_state["answer"],
        thread_id=thread_id,
        project_id=project_id_value,
        project_path=str(project_path),
        intent=intent,
        rewritten_query=rewritten_query,
        expanded_queries=expanded_queries,
        used_vector_search=False,
        used_relational_search=bool(
            {
                "artifact_search",
                "relation_search",
                "aggregate_query",
            }.intersection(final_state.get("used_tools", []))
        ),
        used_source_reading=bool(final_state.get("source_snippets")),
        source_reading_skipped_reason=final_state.get(
            "source_reading_skipped_reason"
        ),
        evidence=list(final_state.get("evidence", [])),
        source_snippets=list(final_state.get("source_snippets", [])),
        warnings=list(final_state.get("warnings", [])),
        mode="agentic",
        used_tools=list(final_state.get("used_tools", [])),
        observations=list(final_state.get("observations", [])),
    )


def _resolve_project_reference(request: RagRequest) -> tuple[str, Path]:
    """Resolve project_id first, then project_path."""
    if request.project_id is not None:
        normalized_project_id = request.project_id.strip()
        if not normalized_project_id:
            raise RagRequestError("project_id must not be empty when provided.")
        cached_path = _cached_project_path(normalized_project_id)
        if cached_path is not None:
            return normalized_project_id, cached_path
        with session_scope() as session:
            project = session.get(Project, normalized_project_id)
            if project is None:
                raise RagRequestError(
                    f"project_id was not found: {normalized_project_id}"
                )
            project_path = Path(project.root_path).expanduser()
            _PROJECT_PATH_CACHE[normalized_project_id] = (
                monotonic(),
                str(project_path),
            )
            return normalized_project_id, project_path

    if request.project_path is None:
        raise RagRequestError("project_id or project_path is required.")

    normalized_project_path = request.project_path.strip()
    if not normalized_project_path:
        raise RagRequestError("project_path must not be empty when provided.")
    project_path = Path(normalized_project_path).expanduser()
    return project_id(project_path.resolve()), project_path


def _cached_project_path(project_id_value: str) -> Path | None:
    item = _PROJECT_PATH_CACHE.get(project_id_value)
    if item is None:
        return None
    cached_at, raw_path = item
    if monotonic() - cached_at > _PROJECT_CACHE_TTL_SECONDS:
        _PROJECT_PATH_CACHE.pop(project_id_value, None)
        return None
    return Path(raw_path)


def answer_to_dict(answer: RagAnswer) -> dict[str, object]:
    """Convert an answer dataclass into a JSON-serializable dictionary."""
    return asdict(answer)


def evidence_to_dict(item: RagEvidence) -> dict[str, object]:
    """Convert evidence to a dictionary."""
    return asdict(item)


def source_snippet_to_dict(item: SourceSnippet) -> dict[str, object]:
    """Convert source snippet to a dictionary."""
    return asdict(item)


def _agentic_expanded_queries(
    question: str,
    understanding: object,
    rewritten_query: str | None = None,
) -> list[str]:
    if not isinstance(understanding, dict):
        return [query for query in (question, rewritten_query) if query]
    values: list[str] = [question]
    if rewritten_query:
        values.append(rewritten_query)
    for key in ("business_terms", "technical_terms", "entities", "sub_questions"):
        raw = understanding.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if str(item).strip())
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
