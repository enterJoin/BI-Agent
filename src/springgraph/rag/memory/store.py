"""Thread memory and persisted chat storage for Agentic RAG."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from springgraph.db import session_scope
from springgraph.models import Project, RagMessage, RagThread

DEFAULT_USER_ID = "1"
DEFAULT_THREAD_TITLE = "New chat"


@dataclass
class ThreadMemory:
    """Minimal per-thread state."""

    project_path: str | None = None
    last_question: str | None = None
    last_evidence_count: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RagThreadRecord:
    """Persisted chat thread returned by APIs."""

    id: int
    project_id: str
    user_id: str
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class RagMessageRecord:
    """Persisted chat message returned by APIs."""

    id: int
    thread_id: str
    role: str
    content: str
    created_at: datetime


_THREADS: dict[str, ThreadMemory] = {}


def get_thread(
    thread_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
) -> ThreadMemory:
    """Return existing thread memory or create a new one."""
    if project_id is None:
        return _THREADS.setdefault(thread_id, ThreadMemory())
    normalized_user_id = normalize_user_id(user_id)
    with session_scope() as session:
        thread = _find_thread(session, project_id, normalized_user_id, thread_id)
        if thread is None:
            return _THREADS.setdefault(thread_id, ThreadMemory())
        project = session.get(Project, project_id)
        rows = session.execute(
            select(RagMessage)
            .where(RagMessage.thread_db_id == thread.id)
            .order_by(RagMessage.created_at, RagMessage.id)
        ).scalars()
        messages = [
            {"role": row.role, "content": row.content}
            for row in rows
        ]
        last_question = _last_user_message(messages)
        return ThreadMemory(
            project_path=project.root_path if project is not None else None,
            last_question=last_question,
            messages=messages,
        )


def update_thread(
    thread_id: str,
    project_id: str,
    user_id: str | None,
    project_path: str,
    question: str,
    answer: str,
    evidence_count: int,
) -> None:
    """Persist the current turn into database-backed thread memory."""
    normalized_user_id = normalize_user_id(user_id)
    with session_scope() as session:
        if session.get(Project, project_id) is None:
            _update_in_process_thread(
                thread_id,
                project_path,
                question,
                answer,
                evidence_count,
            )
            return
        thread = _get_or_create_thread(
            session=session,
            project_id=project_id,
            user_id=normalized_user_id,
            thread_id=thread_id,
            title=question,
        )
        session.add_all(
            [
                RagMessage(
                    thread_db_id=thread.id,
                    role="user",
                    content=question,
                ),
                RagMessage(
                    thread_db_id=thread.id,
                    role="assistant",
                    content=answer,
                ),
            ]
        )
        thread.updated_at = func.now()


def create_chat_thread(
    project_id: str,
    user_id: str | None,
    title: str | None = None,
    thread_id: str | None = None,
) -> RagThreadRecord:
    """Create one chat thread."""
    normalized_user_id = normalize_user_id(user_id)
    normalized_thread_id = thread_id.strip() if thread_id and thread_id.strip() else ""
    if not normalized_thread_id:
        normalized_thread_id = f"thread:{uuid4().hex}"
    normalized_title = _normalize_title(title, DEFAULT_THREAD_TITLE)
    with session_scope() as session:
        _require_project(session, project_id)
        existing = _find_thread(
            session,
            project_id,
            normalized_user_id,
            normalized_thread_id,
        )
        if existing is not None:
            raise ValueError(f"chat thread already exists: {normalized_thread_id}")
        thread = RagThread(
            project_id=project_id,
            user_id=normalized_user_id,
            thread_id=normalized_thread_id,
            title=normalized_title,
        )
        session.add(thread)
        session.flush()
        return _thread_record(thread)


def ensure_chat_thread(
    project_id: str,
    user_id: str | None,
    thread_id: str,
    title: str,
) -> RagThreadRecord:
    """Return an existing chat thread or create it."""
    normalized_user_id = normalize_user_id(user_id)
    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id:
        raise ValueError("thread_id must not be empty.")
    normalized_title = _normalize_title(title, DEFAULT_THREAD_TITLE)
    with session_scope() as session:
        _require_project(session, project_id)
        thread = _find_thread(
            session,
            project_id,
            normalized_user_id,
            normalized_thread_id,
        )
        if thread is None:
            thread = RagThread(
                project_id=project_id,
                user_id=normalized_user_id,
                thread_id=normalized_thread_id,
                title=normalized_title,
            )
            session.add(thread)
            session.flush()
        return _thread_record(thread)


def list_chat_threads(
    project_id: str,
    user_id: str | None,
) -> list[RagThreadRecord]:
    """List chat threads for a project and user."""
    normalized_user_id = normalize_user_id(user_id)
    with session_scope() as session:
        _require_project(session, project_id)
        rows = session.execute(
            select(RagThread)
            .where(RagThread.project_id == project_id)
            .where(RagThread.user_id == normalized_user_id)
            .order_by(desc(RagThread.updated_at), desc(RagThread.id))
        ).scalars()
        return [_thread_record(row) for row in rows]


def list_chat_messages(
    project_id: str,
    user_id: str | None,
    thread_id: str,
) -> list[RagMessageRecord]:
    """List chat messages for one thread."""
    normalized_user_id = normalize_user_id(user_id)
    with session_scope() as session:
        thread = _require_thread(session, project_id, normalized_user_id, thread_id)
        rows = session.execute(
            select(RagMessage)
            .where(RagMessage.thread_db_id == thread.id)
            .order_by(RagMessage.created_at, RagMessage.id)
        ).scalars()
        return [_message_record(thread.thread_id, row) for row in rows]


def delete_chat_thread(
    project_id: str,
    user_id: str | None,
    thread_id: str,
) -> None:
    """Delete one chat thread and its messages."""
    normalized_user_id = normalize_user_id(user_id)
    with session_scope() as session:
        thread = _require_thread(session, project_id, normalized_user_id, thread_id)
        session.delete(thread)


def update_chat_thread_title(
    project_id: str,
    user_id: str | None,
    thread_id: str,
    title: str,
) -> RagThreadRecord:
    """Update one chat thread title."""
    normalized_user_id = normalize_user_id(user_id)
    normalized_title = _normalize_title(title, DEFAULT_THREAD_TITLE)
    with session_scope() as session:
        thread = _require_thread(session, project_id, normalized_user_id, thread_id)
        thread.title = normalized_title
        thread.updated_at = func.now()
        session.flush()
        return _thread_record(thread)


def normalize_user_id(user_id: str | None) -> str:
    """Normalize optional API user_id for stable unique keys."""
    if user_id is None or not user_id.strip():
        return DEFAULT_USER_ID
    return user_id.strip()


def _get_or_create_thread(
    session: Session,
    project_id: str,
    user_id: str,
    thread_id: str,
    title: str,
) -> RagThread:
    thread = _find_thread(session, project_id, user_id, thread_id)
    if thread is not None:
        return thread
    thread = RagThread(
        project_id=project_id,
        user_id=user_id,
        thread_id=thread_id,
        title=_normalize_title(title, DEFAULT_THREAD_TITLE),
    )
    session.add(thread)
    session.flush()
    return thread


def _find_thread(
    session: Session,
    project_id: str,
    user_id: str,
    thread_id: str,
) -> RagThread | None:
    return session.execute(
        select(RagThread)
        .where(RagThread.project_id == project_id)
        .where(RagThread.user_id == user_id)
        .where(RagThread.thread_id == thread_id)
    ).scalar_one_or_none()


def _require_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError(f"project_id was not found: {project_id}")
    return project


def _require_thread(
    session: Session,
    project_id: str,
    user_id: str,
    thread_id: str,
) -> RagThread:
    thread = _find_thread(session, project_id, user_id, thread_id)
    if thread is None:
        raise ValueError(f"chat thread was not found: {thread_id}")
    return thread


def _thread_record(thread: RagThread) -> RagThreadRecord:
    return RagThreadRecord(
        id=thread.id,
        project_id=thread.project_id,
        user_id=thread.user_id,
        thread_id=thread.thread_id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


def _message_record(
    thread_id: str,
    message: RagMessage,
) -> RagMessageRecord:
    return RagMessageRecord(
        id=message.id,
        thread_id=thread_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def _last_user_message(messages: list[dict[str, str]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content")
    return None


def _normalize_title(title: str | None, fallback: str) -> str:
    if title is None or not title.strip():
        return fallback
    return title.strip()


def _update_in_process_thread(
    thread_id: str,
    project_path: str,
    question: str,
    answer: str,
    evidence_count: int,
) -> None:
    memory = _THREADS.setdefault(thread_id, ThreadMemory())
    memory.project_path = project_path
    memory.last_question = question
    memory.last_evidence_count = evidence_count
    memory.messages.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    memory.messages = memory.messages[-10:]
