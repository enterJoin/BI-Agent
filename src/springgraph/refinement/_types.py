"""Internal value objects for Java system refinement."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RefinedFile:
    """A file participating in semantic refinement."""

    path: Path
    relative_path: str
    language: str
    module_name: str | None
    service_name: str | None
    size_bytes: int
    modified_at: datetime


@dataclass(frozen=True)
class SymbolFact:
    """A symbol to upsert into the graph."""

    key: str
    kind: str
    name: str
    qualified_name: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    metadata: dict[str, object] = field(default_factory=dict)
    annotations: list[str] = field(default_factory=list)
    signature: str | None = None


@dataclass(frozen=True)
class EdgeFact:
    """A relationship between two symbol facts."""

    source_key: str
    target_key: str
    kind: str
    line: int | None = None
    confidence: float = 1.0
    resolved_by: str = "refinement"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkFact:
    """A semantic chunk to upsert and embed."""

    key: str
    file_path: str
    symbol_key: str | None
    chunk_type: str
    title: str
    content: str
    language: str
    start_line: int | None
    end_line: int | None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementFacts:
    """Extracted refinement facts."""

    symbols: list[SymbolFact] = field(default_factory=list)
    edges: list[EdgeFact] = field(default_factory=list)
    chunks: list[ChunkFact] = field(default_factory=list)


@dataclass(frozen=True)
class RefinementResult:
    """Summary returned by the public refinement method."""

    project_id: str
    index_run_id: int
    embedding_job_id: int
    files_seen: int
    symbols_upserted: int
    edges_upserted: int
    chunks_upserted: int
    embeddings_upserted: int
    errors: list[str]
