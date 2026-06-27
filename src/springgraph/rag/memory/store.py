"""Small in-process thread memory for Agentic RAG."""

from dataclasses import dataclass, field


@dataclass
class ThreadMemory:
    """Minimal per-thread state."""

    project_path: str | None = None
    last_question: str | None = None
    last_evidence_count: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)


_THREADS: dict[str, ThreadMemory] = {}


def get_thread(thread_id: str) -> ThreadMemory:
    """Return existing thread memory or create a new one."""
    return _THREADS.setdefault(thread_id, ThreadMemory())


def update_thread(
    thread_id: str,
    project_path: str,
    question: str,
    answer: str,
    evidence_count: int,
) -> None:
    """Persist the current turn into in-process memory."""
    memory = get_thread(thread_id)
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
