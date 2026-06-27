"""Public RAG service entry point."""

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from springgraph.db import session_scope
from springgraph.hashing import project_id
from springgraph.models import Project
from springgraph.rag.agent.graph import build_agentic_rag_graph
from springgraph.rag.schemas import RagAnswer, RagEvidence, RagRequest, SourceSnippet


class RagRequestError(ValueError):
    """Raised when a RAG request is invalid."""


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
        {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "project_path": str(project_path),
            "project_id": project_id_value,
            "question": question,
            "top_k": request.top_k,
            "graph_depth": request.graph_depth,
            "read_source": request.read_source,
            "warnings": [],
        }
    )
    understanding = final_state.get("question_understanding", {})
    intent = str(understanding.get("task_goal", "agentic_rag"))
    expanded_queries = _agentic_expanded_queries(question, understanding)
    return RagAnswer(
        answer=final_state["answer"],
        thread_id=thread_id,
        project_id=project_id_value,
        project_path=str(project_path),
        intent=intent,
        rewritten_query=question,
        expanded_queries=expanded_queries,
        used_vector_search="vector_search" in final_state.get("used_tools", []),
        used_relational_search=bool(
            {
                "relational_search",
                "call_graph_search",
                "db_mapping_search",
                "config_search",
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
        with session_scope() as session:
            project = session.get(Project, normalized_project_id)
            if project is None:
                raise RagRequestError(
                    f"project_id was not found: {normalized_project_id}"
                )
            return normalized_project_id, Path(project.root_path).expanduser()

    if request.project_path is None:
        raise RagRequestError("project_id or project_path is required.")

    normalized_project_path = request.project_path.strip()
    if not normalized_project_path:
        raise RagRequestError("project_path must not be empty when provided.")
    project_path = Path(normalized_project_path).expanduser()
    return project_id(project_path.resolve()), project_path


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
) -> list[str]:
    if not isinstance(understanding, dict):
        return [question]
    values: list[str] = [question]
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
