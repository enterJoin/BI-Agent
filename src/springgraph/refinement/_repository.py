"""Persistence helpers for semantic refinement."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileChangePlan:
    """Files grouped by whether semantic refinement must be rebuilt."""

    changed: list[RefinedFile]
    unchanged: list[RefinedFile]
    content_hashes: dict[str, str]


@dataclass(frozen=True)
class _EmbeddingInput:
    """Chunk data needed to build one embedding row."""

    chunk_id: str
    content: str
    content_hash: str


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

    def delete_missing_files(self, current_paths: set[str]) -> int:
        """Delete file rows that no longer exist in the scanned project."""
        statement = delete(File).where(File.project_id == self.project_id)
        if current_paths:
            statement = statement.where(File.path.not_in(current_paths))
        result = self.session.execute(statement)
        return int(getattr(result, "rowcount", 0) or 0)

    def plan_file_changes(self, files: list[RefinedFile]) -> FileChangePlan:
        """Compare scanned files with stored hashes for incremental refinement."""
        existing_rows = self.session.execute(
            select(File.path, File.content_hash, File.error).where(
                File.project_id == self.project_id
            )
        ).all()
        existing = {
            str(path): (str(content_hash_value), error)
            for path, content_hash_value, error in existing_rows
        }
        changed: list[RefinedFile] = []
        unchanged: list[RefinedFile] = []
        content_hashes: dict[str, str] = {}
        for item in files:
            source = _read_text(item.path)
            digest_value = content_hash(source)
            content_hashes[item.relative_path] = digest_value
            stored = existing.get(item.relative_path)
            if stored is not None and stored[0] == digest_value and stored[1] is None:
                unchanged.append(item)
            else:
                changed.append(item)
        return FileChangePlan(
            changed=changed,
            unchanged=unchanged,
            content_hashes=content_hashes,
        )

    def file_ids_for(self, files: list[RefinedFile]) -> dict[str, str]:
        """Return relative path to deterministic file row ID mapping."""
        return {
            item.relative_path: file_row_id(self.project_id, item.relative_path)
            for item in files
        }

    def upsert_files(
        self,
        files: list[RefinedFile],
        content_hashes: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Upsert file rows and return relative path to row ID mapping."""
        file_ids: dict[str, str] = {}
        for item in files:
            row_id = file_row_id(self.project_id, item.relative_path)
            file_ids[item.relative_path] = row_id
            digest_value = (
                content_hashes[item.relative_path]
                if content_hashes and item.relative_path in content_hashes
                else content_hash(_read_text(item.path))
            )
            values = {
                "id": row_id,
                "project_id": self.project_id,
                "path": item.relative_path,
                "module_name": item.module_name,
                "service_name": item.service_name,
                "language": item.language,
                "content_hash": digest_value,
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

    def clear_file_refinement_outputs(self, file_ids: list[str]) -> None:
        """Remove stale semantic chunks and outgoing refinement edges for files."""
        if not file_ids:
            return
        source_symbol_ids = select(Symbol.id).where(
            Symbol.project_id == self.project_id,
            Symbol.file_id.in_(file_ids),
        )
        self.session.execute(
            delete(Edge).where(
                Edge.project_id == self.project_id,
                Edge.resolved_by == "refinement",
                Edge.source_id.in_(source_symbol_ids),
            )
        )
        self.session.execute(
            delete(CodeChunk).where(
                CodeChunk.project_id == self.project_id,
                CodeChunk.file_id.in_(file_ids),
            )
        )

    def typed_artifact_chunks_exist(self, template_version: str) -> bool:
        """Return whether typed artifact vector chunks exist for this project."""
        row = self.session.execute(
            select(CodeChunk.id)
            .where(CodeChunk.project_id == self.project_id)
            .where(CodeChunk.chunk_type.like("artifact_%"))
            .where(CodeChunk.template_version == template_version)
            .limit(1)
        ).first()
        return row is not None

    def delete_typed_artifact_chunks(self) -> int:
        """Delete generated typed artifact chunks for this project."""
        result = self.session.execute(
            delete(CodeChunk).where(
                CodeChunk.project_id == self.project_id,
                CodeChunk.chunk_type.like("artifact_%"),
            )
        )
        return int(getattr(result, "rowcount", 0) or 0)

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
        started_at = perf_counter()
        logger.info(
            "Chunk and embedding upsert started: project_id=%s, chunks=%s, "
            "embedding_model=%s, embedding_dim=%s, batch_size=%s, "
            "max_concurrency=%s",
            self.project_id,
            len(chunks),
            embedder.model_name,
            embedder.dimensions,
            embedder.batch_size,
            embedder.max_concurrency,
        )
        chunk_count = 0
        embedding_count = 0
        embedding_inputs: list[_EmbeddingInput] = []
        for chunk in chunks:
            chunk_id = _chunk_id(self.project_id, chunk)
            content_digest = content_hash(chunk.content)
            symbol_id_value = None
            if chunk.symbol_key is not None:
                symbol_id_value = key_to_id.get(chunk.symbol_key)
            elif isinstance(chunk.metadata.get("symbol_id"), str):
                symbol_id_value = str(chunk.metadata["symbol_id"])
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
            embedding_inputs.append(
                _EmbeddingInput(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    content_hash=content_digest,
                )
            )

        embedding_count = self._upsert_embedding_inputs(embedding_inputs, embedder)
        logger.info(
            "Chunk and embedding upsert completed: project_id=%s, "
            "chunks_upserted=%s, embeddings_upserted=%s, elapsed_seconds=%.3f",
            self.project_id,
            chunk_count,
            embedding_count,
            perf_counter() - started_at,
        )
        return chunk_count, embedding_count

    def _upsert_embedding(
        self,
        chunk_id: str,
        content_digest: str,
        embedding: list[float],
        embedder: Embedder,
    ) -> None:
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

    def _upsert_embedding_inputs(
        self,
        inputs: list[_EmbeddingInput],
        embedder: Embedder,
    ) -> int:
        if not inputs:
            return 0
        batches = _batched(inputs, embedder.batch_size)
        if embedder.max_concurrency <= 1 or len(batches) <= 1:
            return self._upsert_embedding_batches(batches, embedder)
        return self._upsert_embedding_batches_concurrently(batches, embedder)

    def _upsert_embedding_batches(
        self,
        batches: list[list[_EmbeddingInput]],
        embedder: Embedder,
    ) -> int:
        embedding_count = 0
        for batch in batches:
            embeddings = embedder.embed_batch([item.content for item in batch])
            embedding_count += self._upsert_embedding_batch(
                batch,
                embeddings,
                embedder,
            )
        return embedding_count

    def _upsert_embedding_batches_concurrently(
        self,
        batches: list[list[_EmbeddingInput]],
        embedder: Embedder,
    ) -> int:
        max_workers = min(embedder.max_concurrency, len(batches))
        embedding_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future[list[list[float]]], list[_EmbeddingInput]] = {
                executor.submit(
                    embedder.embed_batch,
                    [item.content for item in batch],
                ): batch
                for batch in batches
            }
            try:
                for future in as_completed(futures):
                    batch = futures[future]
                    embeddings = future.result()
                    embedding_count += self._upsert_embedding_batch(
                        batch,
                        embeddings,
                        embedder,
                    )
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        return embedding_count

    def _upsert_embedding_batch(
        self,
        batch: list[_EmbeddingInput],
        embeddings: list[list[float]],
        embedder: Embedder,
    ) -> int:
        embedding_count = 0
        for item, embedding in zip(batch, embeddings, strict=True):
            self._upsert_embedding(
                item.chunk_id,
                item.content_hash,
                embedding,
                embedder,
            )
            embedding_count += 1
        return embedding_count

    def chunks_missing_embeddings(
        self,
        file_ids: list[str],
        embedder: Embedder,
    ) -> list[CodeChunk]:
        """Return unchanged chunks missing vectors for the current embedder."""
        if not file_ids:
            return []
        matching_embedding = (
            select(ChunkEmbedding.id)
            .where(
                ChunkEmbedding.chunk_id == CodeChunk.id,
                ChunkEmbedding.project_id == self.project_id,
                ChunkEmbedding.embedding_model == embedder.model_name,
                ChunkEmbedding.embedding_dim == embedder.dimensions,
                ChunkEmbedding.content_hash == CodeChunk.content_hash,
                ChunkEmbedding.status == "active",
            )
            .exists()
        )
        return list(
            self.session.scalars(
                select(CodeChunk).where(
                    CodeChunk.project_id == self.project_id,
                    CodeChunk.file_id.in_(file_ids),
                    CodeChunk.status == "active",
                    ~matching_embedding,
                )
            )
        )

    def upsert_embeddings_for_existing_chunks(
        self,
        chunks: list[CodeChunk],
        embedder: Embedder,
    ) -> int:
        """Build embeddings for already stored chunks."""
        started_at = perf_counter()
        logger.info(
            "Missing embedding upsert started: project_id=%s, chunks=%s, "
            "embedding_model=%s, embedding_dim=%s, batch_size=%s, "
            "max_concurrency=%s",
            self.project_id,
            len(chunks),
            embedder.model_name,
            embedder.dimensions,
            embedder.batch_size,
            embedder.max_concurrency,
        )
        embedding_count = self._upsert_embedding_inputs(
            [
                _EmbeddingInput(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                )
                for chunk in chunks
            ],
            embedder,
        )
        logger.info(
            "Missing embedding upsert completed: project_id=%s, "
            "embeddings_upserted=%s, elapsed_seconds=%.3f",
            self.project_id,
            embedding_count,
            perf_counter() - started_at,
        )
        return embedding_count

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


def _batched[T](items: list[T], size: int) -> list[list[T]]:
    batch_size = max(1, size)
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
