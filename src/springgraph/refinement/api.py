"""Public API for semantic Java system refinement."""

import logging
from pathlib import Path
from time import perf_counter

from sqlalchemy.orm import Session

from springgraph.db import session_scope
from springgraph.hashing import project_id
from springgraph.refinement._embedder import create_embedder
from springgraph.refinement._extractors import extract_file_facts
from springgraph.refinement._library import (
    extract_library_file_facts,
    scan_library_files,
)
from springgraph.refinement._repository import RefinementRepository, merge_facts
from springgraph.refinement._scanner import scan_refinement_files
from springgraph.refinement._types import EdgeFact, RefinementFacts, RefinementResult
from springgraph.refinement.relational.indexer import index_project

__all__ = ["RefinementResult", "refine_project"]

logger = logging.getLogger(__name__)


def refine_project(
    project_path: str | Path,
    session: Session | None = None,
) -> RefinementResult:
    """Refine a Java workspace into graph facts, chunks, and vector rows."""
    root = Path(project_path).resolve()
    if session is None:
        with session_scope() as owned_session:
            return _refine_with_session(root, owned_session)
    return _refine_with_session(root, session)


def _refine_with_session(root: Path, session: Session) -> RefinementResult:
    started_at = perf_counter()
    project_id_value = project_id(root)
    logger.info(
        "Project refinement started: project_id=%s, root=%s",
        project_id_value,
        root,
    )
    stage_started_at = perf_counter()
    index_run_id = index_project(root, session)
    logger.info(
        "Relational indexing stage completed: project_id=%s, "
        "index_run_id=%s, elapsed_seconds=%.3f",
        project_id_value,
        index_run_id,
        perf_counter() - stage_started_at,
    )

    repository = RefinementRepository(session, project_id_value)
    repository.upsert_project(root)
    logger.info(
        "Clearing previous semantic refinement outputs: project_id=%s",
        project_id_value,
    )
    repository.clear_refinement_outputs()
    stage_started_at = perf_counter()
    source_files = scan_refinement_files(root)
    library_files = scan_library_files(root)
    files = source_files + library_files
    logger.info(
        "Semantic refinement scan completed: project_id=%s, source_files=%s, "
        "library_files=%s, total_files=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(source_files),
        len(library_files),
        len(files),
        perf_counter() - stage_started_at,
    )
    stage_started_at = perf_counter()
    file_ids = repository.upsert_files(files)
    logger.info(
        "Semantic file rows upserted: project_id=%s, files=%s, "
        "elapsed_seconds=%.3f",
        project_id_value,
        len(file_ids),
        perf_counter() - stage_started_at,
    )

    errors: list[str] = []
    per_file_facts: list[RefinementFacts] = []
    stage_started_at = perf_counter()
    for item in source_files:
        try:
            per_file_facts.append(
                extract_file_facts(
                    item.path,
                    item.relative_path,
                    item.language,
                    item.module_name,
                    item.service_name,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.relative_path}: {exc}")
            logger.warning(
                "Semantic source extraction failed: project_id=%s, file=%s, "
                "error=%s",
                project_id_value,
                item.relative_path,
                exc,
            )
    logger.info(
        "Semantic source extraction completed: project_id=%s, processed=%s, "
        "fact_batches=%s, errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(source_files),
        len(per_file_facts),
        len(errors),
        perf_counter() - stage_started_at,
    )

    library_started_at = perf_counter()
    for item in library_files:
        try:
            per_file_facts.append(
                extract_library_file_facts(item.path, item.relative_path)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.relative_path}: {exc}")
            logger.warning(
                "Library extraction failed: project_id=%s, file=%s, error=%s",
                project_id_value,
                item.relative_path,
                exc,
            )
    logger.info(
        "Library extraction completed: project_id=%s, processed=%s, "
        "fact_batches=%s, errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(library_files),
        len(per_file_facts),
        len(errors),
        perf_counter() - library_started_at,
    )

    stage_started_at = perf_counter()
    facts = _normalize_entity_table_edges(merge_facts(per_file_facts))
    logger.info(
        "Semantic facts merged: project_id=%s, symbols=%s, edges=%s, chunks=%s, "
        "errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(facts.symbols),
        len(facts.edges),
        len(facts.chunks),
        len(errors),
        perf_counter() - stage_started_at,
    )
    stage_started_at = perf_counter()
    symbols_upserted, edges_upserted, key_to_id = repository.upsert_facts(
        facts, file_ids
    )
    logger.info(
        "Semantic facts upserted: project_id=%s, symbols_upserted=%s, "
        "edges_upserted=%s, symbol_key_count=%s, elapsed_seconds=%.3f",
        project_id_value,
        symbols_upserted,
        edges_upserted,
        len(key_to_id),
        perf_counter() - stage_started_at,
    )

    embedder = create_embedder()
    logger.info(
        "Embedding stage started: project_id=%s, model=%s, dimensions=%s, "
        "chunks=%s",
        project_id_value,
        embedder.model_name,
        embedder.dimensions,
        len(facts.chunks),
    )
    stage_started_at = perf_counter()
    job = repository.create_embedding_job(index_run_id, embedder, len(facts.chunks))
    chunks_upserted, embeddings_upserted = repository.upsert_chunks_and_embeddings(
        facts.chunks,
        file_ids,
        key_to_id,
        index_run_id,
        embedder,
    )
    repository.finish_embedding_job(job, embeddings_upserted, errors)
    session.flush()
    logger.info(
        "Embedding stage completed: project_id=%s, embedding_job_id=%s, "
        "chunks_upserted=%s, embeddings_upserted=%s, elapsed_seconds=%.3f",
        project_id_value,
        job.id,
        chunks_upserted,
        embeddings_upserted,
        perf_counter() - stage_started_at,
    )
    logger.info(
        "Project refinement completed: project_id=%s, files_seen=%s, "
        "symbols_upserted=%s, edges_upserted=%s, chunks_upserted=%s, "
        "embeddings_upserted=%s, errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(files),
        symbols_upserted,
        edges_upserted,
        chunks_upserted,
        embeddings_upserted,
        len(errors),
        perf_counter() - started_at,
    )

    return RefinementResult(
        project_id=project_id_value,
        index_run_id=index_run_id,
        embedding_job_id=job.id,
        files_seen=len(files),
        symbols_upserted=symbols_upserted,
        edges_upserted=edges_upserted,
        chunks_upserted=chunks_upserted,
        embeddings_upserted=embeddings_upserted,
        errors=errors,
    )


def _normalize_entity_table_edges(facts: RefinementFacts) -> RefinementFacts:
    """Point inferred MyBatis-Plus CRUD edges at @TableName symbols when known."""
    entity_to_table = {
        symbol.name: str(symbol.metadata["table"])
        for symbol in facts.symbols
        if symbol.kind == "data_contract" and "table" in symbol.metadata
    }
    normalized_edges: list[EdgeFact] = []
    for edge in facts.edges:
        entity = edge.metadata.get("entity")
        if (
            edge.kind in {"reads_table", "writes_table"}
            and isinstance(entity, str)
            and entity in entity_to_table
        ):
            normalized_edges.append(
                EdgeFact(
                    source_key=edge.source_key,
                    target_key=f"db_table:{entity_to_table[entity]}",
                    kind=edge.kind,
                    line=edge.line,
                    confidence=max(edge.confidence, 0.86),
                    resolved_by=edge.resolved_by,
                    metadata=edge.metadata
                    | {
                        "normalized_from_entity": entity,
                        "normalized_by": "@TableName",
                    },
                )
            )
        else:
            normalized_edges.append(edge)
    return RefinementFacts(
        symbols=facts.symbols,
        edges=normalized_edges,
        chunks=facts.chunks,
    )
