"""Shared RAG data structures."""

from dataclasses import dataclass, field

RagIntent = str
RagMode = str


@dataclass(frozen=True)
class RagRequest:
    """Input accepted by the RAG service."""

    question: str
    project_id: str | None = None
    project_path: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    top_k: int = 8
    graph_depth: int = 2
    read_source: bool = True
    mode: RagMode = "agentic"


@dataclass(frozen=True)
class RagPlan:
    """Query plan produced before retrieval."""

    intent: RagIntent
    original_question: str
    rewritten_query: str
    expanded_queries: list[str]
    entities: list[str] = field(default_factory=list)
    use_vector_search: bool = True
    use_relational_search: bool = True
    relation_expansion: bool = True
    need_source_reading: bool = True
    top_k: int = 8
    graph_depth: int = 2


@dataclass(frozen=True)
class RagEvidence:
    """One evidence item retrieved from vector or relational storage."""

    evidence_type: str
    source: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None
    score: float = 0.0
    content_excerpt: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSnippet:
    """Source code snippet read from an available project directory."""

    file_path: str
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True)
class RagAnswer:
    """Response returned by the RAG service and HTTP API."""

    answer: str
    thread_id: str
    project_id: str
    project_path: str
    intent: RagIntent
    rewritten_query: str
    expanded_queries: list[str]
    used_vector_search: bool
    used_relational_search: bool
    used_source_reading: bool
    source_reading_skipped_reason: str | None
    evidence: list[RagEvidence]
    source_snippets: list[SourceSnippet]
    warnings: list[str] = field(default_factory=list)
    mode: RagMode = "agentic"
    used_tools: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
