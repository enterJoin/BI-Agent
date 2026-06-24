"""Persistence helpers for semantic refinement."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Table, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from springgraph.hashing import content_hash, digest, file_row_id
from springgraph.models import (
    ChunkEmbedding,
    CodeChunk,
    Edge,
    EmbeddingJob,
    File,
    Project,
    Symbol,
)
from springgraph.refinement._embedder import Embedder
from springgraph.refinement._types import (
    ChunkFact,
    EdgeFact,
    RefinedFile,
    RefinementFacts,
    SymbolFact,
)

PARSER_VERSION = "semantic-refinement-v1"
TEMPLATE_VERSION = "semantic-refinement-v1"


class RefinementRepository:
    """Write refinement facts into the graph and vector tables."""

    def __init__(self, session: Session, project_id_value: str) -> None:
        self.session = session
        self.project_id = project_id_value

    def upsert_project(self, root: Path) -> None:
        """Ensure project row exists."""
        stmt = insert(Project).values(
            id=self.project_id,
            root_path=str(root.resolve()),
            name=root.resolve().name,
            updated_at=datetime.now(tz=UTC),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[Project.id],
            set_={
                "root_path": str(root.resolve()),
                "name": root.resolve().name,
                "updated_at": datetime.now(tz=UTC),
            },
        )
        self.session.execute(stmt)

    def clear_refinement_outputs(self) -> None:
        """Remove previous derived refinement outputs for this project."""
        self.session.execute(
            delete(ChunkEmbedding).where(ChunkEmbedding.project_id == self.project_id)
        )
        self.session.execute(
            delete(CodeChunk).where(CodeChunk.project_id == self.project_id)
        )
        self.session.execute(
            delete(Edge).where(
                Edge.project_id == self.project_id,
                Edge.resolved_by == "refinement",
            )
        )

    def upsert_files(self, files: list[RefinedFile]) -> dict[str, str]:
        """Upsert file rows and return relative path to row ID mapping."""
        file_ids: dict[str, str] = {}
        for item in files:
            source = _read_text(item.path)
            row_id = file_row_id(self.project_id, item.relative_path)
            file_ids[item.relative_path] = row_id
            values = {
                "id": row_id,
                "project_id": self.project_id,
                "path": item.relative_path,
                "module_name": item.module_name,
                "service_name": item.service_name,
                "language": item.language,
                "content_hash": content_hash(source),
                "size_bytes": item.size_bytes,
                "modified_at": item.modified_at,
                "indexed_at": datetime.now(tz=UTC),
                "error": None,
            }
            stmt = insert(File).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[File.project_id, File.path],
                set_=values,
            )
            self.session.execute(stmt)
        return file_ids

    def upsert_facts(
        self, facts: RefinementFacts, file_ids: dict[str, str]
    ) -> tuple[int, int, dict[str, str]]:
        """Upsert symbols and edges."""
        key_to_id: dict[str, str] = {}
        for fact in facts.symbols:
            symbol_id_value = _symbol_id(self.project_id, fact)
            key_to_id[fact.key] = symbol_id_value
            file_id = file_ids[fact.file_path]
            existing = self.session.get(Symbol, symbol_id_value)
            if existing is not None:
                existing.meta = dict(existing.meta) | fact.metadata
                existing.updated_at = datetime.now(tz=UTC)
                key_to_id[fact.key] = existing.id
                continue
            values: dict[str, Any] = {
                "id": symbol_id_value,
                "project_id": self.project_id,
                "file_id": file_id,
                "kind": fact.kind,
                "name": fact.name,
                "qualified_name": fact.qualified_name,
                "language": fact.language,
                "start_line": max(1, fact.start_line),
                "end_line": max(1, fact.end_line),
                "start_column": 0,
                "end_column": 0,
                "signature": fact.signature,
                "docstring": None,
                "annotations": fact.annotations,
                "modifiers": [],
                "metadata": fact.metadata,
                "updated_at": datetime.now(tz=UTC),
            }
            table = cast(Table, Symbol.__table__)
            stmt = insert(table).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    table.c.project_id,
                    table.c.file_id,
                    table.c.kind,
                    table.c.qualified_name,
                    table.c.start_line,
                ],
                set_={
                    key: getattr(stmt.excluded, key)
                    for key in values
                    if key != "id"
                },
            )
            result = self.session.execute(stmt.returning(table.c.id))
            key_to_id[fact.key] = str(result.scalar_one())

        edge_count = 0
        for edge in facts.edges:
            source_id = key_to_id.get(edge.source_key)
            target_id = key_to_id.get(edge.target_key)
            if source_id is None or target_id is None:
                continue
            values = {
                "project_id": self.project_id,
                "source_id": source_id,
                "target_id": target_id,
                "kind": edge.kind,
                "line": edge.line,
                "column_no": 0,
                "confidence": edge.confidence,
                "resolved_by": edge.resolved_by,
                "metadata": edge.metadata,
            }
            table = cast(Table, Edge.__table__)
            stmt = insert(table).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    table.c.project_id,
                    table.c.source_id,
                    table.c.target_id,
                    table.c.kind,
                    table.c.line,
                    table.c.column_no,
                ],
                set_={
                    "confidence": stmt.excluded.confidence,
                    "resolved_by": stmt.excluded.resolved_by,
                    "metadata": stmt.excluded.metadata,
                },
            )
            self.session.execute(stmt)
            edge_count += 1
        return len(facts.symbols), edge_count, key_to_id

    def create_embedding_job(
        self, index_run_id: int, embedder: Embedder, chunks_total: int
    ) -> EmbeddingJob:
        """Create an embedding job row."""
        job = EmbeddingJob(
            project_id=self.project_id,
            index_run_id=index_run_id,
            status="running",
            embedding_model=embedder.model_name,
            embedding_dim=embedder.dimensions,
            chunks_total=chunks_total,
            chunks_embedded=0,
            errors=[],
        )
        self.session.add(job)
        self.session.flush()
        return job

    def upsert_chunks_and_embeddings(
        self,
        chunks: list[ChunkFact],
        file_ids: dict[str, str],
        key_to_id: dict[str, str],
        index_run_id: int,
        embedder: Embedder,
    ) -> tuple[int, int]:
        """Upsert code chunks and deterministic embeddings."""
        chunk_count = 0
        embedding_count = 0
        for chunk in chunks:
            chunk_id = _chunk_id(self.project_id, chunk)
            content_digest = content_hash(chunk.content)
            symbol_id_value = None
            if chunk.symbol_key is not None:
                symbol_id_value = key_to_id.get(chunk.symbol_key)
            values: dict[str, Any] = {
                "id": chunk_id,
                "project_id": self.project_id,
                "file_id": file_ids[chunk.file_path],
                "symbol_id": symbol_id_value,
                "index_run_id": index_run_id,
                "chunk_type": chunk.chunk_type,
                "title": chunk.title,
                "content": chunk.content,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content_hash": content_digest,
                "status": "active",
                "parser_version": str(
                    chunk.metadata.get("parser_version", PARSER_VERSION)
                ),
                "template_version": str(
                    chunk.metadata.get("template_version", TEMPLATE_VERSION)
                ),
                "metadata": chunk.metadata,
                "updated_at": datetime.now(tz=UTC),
            }
            table = cast(Table, CodeChunk.__table__)
            stmt = insert(table).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[table.c.id],
                set_={key: getattr(stmt.excluded, key) for key in values},
            )
            self.session.execute(stmt)
            chunk_count += 1

            embedding = embedder.embed(chunk.content)
            embedding_id = _embedding_id(
                self.project_id,
                chunk_id,
                embedder.model_name,
                embedder.dimensions,
                content_digest,
            )
            embedding_values = {
                "id": embedding_id,
                "chunk_id": chunk_id,
                "project_id": self.project_id,
                "embedding_model": embedder.model_name,
                "embedding_dim": embedder.dimensions,
                "embedding": embedding,
                "content_hash": content_digest,
                "status": "active",
            }
            embedding_table = cast(Table, ChunkEmbedding.__table__)
            embedding_stmt = insert(embedding_table).values(**embedding_values)
            embedding_stmt = embedding_stmt.on_conflict_do_update(
                index_elements=[embedding_table.c.id],
                set_={
                    key: getattr(embedding_stmt.excluded, key)
                    for key in embedding_values
                },
            )
            self.session.execute(embedding_stmt)
            embedding_count += 1
        return chunk_count, embedding_count

    def finish_embedding_job(
        self, job: EmbeddingJob, chunks_embedded: int, errors: list[str]
    ) -> None:
        """Mark an embedding job complete."""
        job.status = "completed" if not errors else "completed_with_errors"
        job.chunks_embedded = chunks_embedded
        job.errors = errors
        job.finished_at = datetime.now(tz=UTC)

    def find_latest_index_run_id(self) -> int:
        """Return the latest index run ID for this project."""
        run_id = self.session.scalar(
            select(EmbeddingJob.index_run_id)
            .where(EmbeddingJob.project_id == self.project_id)
            .order_by(EmbeddingJob.id.desc())
        )
        return int(run_id or 0)


def merge_facts(items: list[RefinementFacts]) -> RefinementFacts:
    """Merge per-file facts into a single collection, preserving order."""
    symbols: list[SymbolFact] = []
    edges: list[EdgeFact] = []
    chunks: list[ChunkFact] = []
    seen_symbols: set[str] = set()
    seen_edges: set[tuple[str, str, str, int | None]] = set()
    seen_chunks: set[str] = set()
    for item in items:
        for symbol in item.symbols:
            if symbol.key in seen_symbols:
                continue
            seen_symbols.add(symbol.key)
            symbols.append(symbol)
        for edge in item.edges:
            marker = (edge.source_key, edge.target_key, edge.kind, edge.line)
            if marker in seen_edges:
                continue
            seen_edges.add(marker)
            edges.append(edge)
        for chunk in item.chunks:
            if chunk.key in seen_chunks:
                continue
            seen_chunks.add(chunk.key)
            chunks.append(chunk)
    return RefinementFacts(symbols=symbols, edges=edges, chunks=chunks)


def _symbol_id(project_id_value: str, fact: SymbolFact) -> str:
    return f"{fact.kind}:{digest(f'{project_id_value}:{fact.key}', 32)}"


def _chunk_id(project_id_value: str, chunk: ChunkFact) -> str:
    return f"chunk:{digest(f'{project_id_value}:{chunk.key}', 32)}"


def _embedding_id(
    project_id_value: str,
    chunk_id: str,
    model_name: str,
    dimensions: int,
    content_digest: str,
) -> str:
    raw = f"{project_id_value}:{chunk_id}:{model_name}:{dimensions}:{content_digest}"
    return f"embedding:{digest(raw, 32)}"


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
