"""SQLAlchemy persistence models."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import UserDefinedType


class Base(DeclarativeBase):
    """Base model class."""


JsonDict = dict[str, Any]
JsonList = list[Any]


class VectorType(UserDefinedType[list[float]]):
    """Minimal pgvector SQLAlchemy type without adding a runtime dependency."""

    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **_: object) -> str:
        """Return the PostgreSQL column type."""
        return f"vector({self.dimensions})"

    def bind_processor(
        self,
        dialect: object,
    ) -> Callable[[list[float] | None], str | None]:
        """Serialize Python vectors into pgvector's text input format."""

        def process(value: list[float] | None) -> str | None:
            if value is None:
                return None
            return "[" + ",".join(f"{item:.8f}" for item in value) + "]"

        return process

    def result_processor(
        self,
        dialect: object,
        coltype: object,
    ) -> Callable[[object], list[float] | None]:
        """Deserialize pgvector text output into floats."""

        def process(value: object) -> list[float] | None:
            if value is None:
                return None
            text = str(value).strip("[]")
            if not text:
                return []
            return [float(item) for item in text.split(",")]

        return process


class Project(Base):
    """Indexed project root."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    root_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class File(Base):
    """Indexed source or config file."""

    __tablename__ = "files"
    __table_args__ = (UniqueConstraint("project_id", "path"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    module_name: Mapped[str | None] = mapped_column(Text)
    service_name: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    error: Mapped[str | None] = mapped_column(Text)


class Symbol(Base):
    """Code graph symbol."""

    __tablename__ = "symbols"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "file_id", "kind", "qualified_name", "start_line"
        ),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        Text, ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    qualified_name: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    start_column: Mapped[int] = mapped_column(Integer, nullable=False)
    end_column: Mapped[int] = mapped_column(Integer, nullable=False)
    signature: Mapped[str | None] = mapped_column(Text)
    docstring: Mapped[str | None] = mapped_column(Text)
    annotations: Mapped[JsonList] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    modifiers: Mapped[JsonList] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    meta: Mapped[JsonDict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Edge(Base):
    """Code graph edge."""

    __tablename__ = "edges"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "source_id", "target_id", "kind", "line", "column_no"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(
        Text, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[str] = mapped_column(
        Text, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    line: Mapped[int | None] = mapped_column(Integer)
    column_no: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="1.0"
    )
    resolved_by: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="extractor"
    )
    meta: Mapped[JsonDict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UnresolvedRef(Base):
    """Unresolved reference for second-pass resolution."""

    __tablename__ = "unresolved_refs"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "from_symbol_id",
            "reference_name",
            "reference_kind",
            "line",
            "column_no",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    from_symbol_id: Mapped[str] = mapped_column(
        Text, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        Text, ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    reference_name: Mapped[str] = mapped_column(Text, nullable=False)
    reference_kind: Mapped[str] = mapped_column(Text, nullable=False)
    line: Mapped[int] = mapped_column(Integer, nullable=False)
    column_no: Mapped[int] = mapped_column(Integer, nullable=False)
    candidates: Mapped[JsonList] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    meta: Mapped[JsonDict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class IndexRun(Base):
    """Indexing run status."""

    __tablename__ = "index_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    files_seen: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    files_indexed: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    symbols_created: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    edges_created: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    unresolved_created: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    errors: Mapped[JsonList] = mapped_column(JSONB, nullable=False, server_default="[]")


class CodeChunk(Base):
    """Semantic text chunk generated from indexed code facts."""

    __tablename__ = "code_chunks"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "file_id",
            "symbol_id",
            "chunk_type",
            "content_hash",
            "template_version",
        ),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        Text, ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    symbol_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("symbols.id", ondelete="SET NULL")
    )
    index_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("index_runs.id", ondelete="SET NULL")
    )
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    parser_version: Mapped[str] = mapped_column(Text, nullable=False)
    template_version: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[JsonDict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChunkEmbedding(Base):
    """Vector embedding for a semantic code chunk."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id", "embedding_model", "embedding_dim", "content_hash"
        ),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(
        Text, ForeignKey("code_chunks.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(VectorType(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EmbeddingJob(Base):
    """Embedding build job status."""

    __tablename__ = "embedding_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    index_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("index_runs.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    chunks_total: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    chunks_embedded: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    errors: Mapped[JsonList] = mapped_column(JSONB, nullable=False, server_default="[]")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
