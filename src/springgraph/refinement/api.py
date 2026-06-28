"""Public API for semantic Java system refinement."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from springgraph.refinement._types import (
    EdgeFact,
    RefinedFile,
    RefinementFacts,
    RefinementResult,
)
from springgraph.refinement.relational.indexer import index_project
from springgraph.refinement.relational.scanner import scan_project

__all__ = ["RefinementResult", "refine_project"]

logger = logging.getLogger(__name__)

REFINE_EXTRACT_WORKERS = max(1, min(8, os.cpu_count() or 1))


@dataclass(frozen=True)
class _ExtractionOutcome:
    """Result of one parallel file extraction."""

    relative_path: str
    facts: RefinementFacts | None = None
    error: str | None = None


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
    repository = RefinementRepository(session, project_id_value)
    repository.upsert_project(root)

    stage_started_at = perf_counter()
    relational_files = scan_project(root)
    source_files = scan_refinement_files(root)
    library_files = scan_library_files(root)
    files = source_files + library_files
    current_paths = {item.relative_path for item in relational_files}
    current_paths.update(item.relative_path for item in source_files)
    current_paths.update(item.relative_path for item in library_files)
    logger.info(
        "Project scan completed: project_id=%s, relational_files=%s, "
        "source_files=%s, library_files=%s, unique_files=%s, "
        "elapsed_seconds=%.3f",
        project_id_value,
        len(relational_files),
        len(source_files),
        len(library_files),
        len(current_paths),
        perf_counter() - stage_started_at,
    )

    stage_started_at = perf_counter()
    deleted_files = repository.delete_missing_files(current_paths)
    logger.info(
        "Missing file cleanup completed: project_id=%s, deleted_files=%s, "
        "elapsed_seconds=%.3f",
        project_id_value,
        deleted_files,
        perf_counter() - stage_started_at,
    )

    stage_started_at = perf_counter()
    file_change_plan = repository.plan_file_changes(files)
    changed_paths = {item.relative_path for item in file_change_plan.changed}
    logger.info(
        "Semantic incremental plan completed: project_id=%s, changed_files=%s, "
        "unchanged_files=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(file_change_plan.changed),
        len(file_change_plan.unchanged),
        perf_counter() - stage_started_at,
    )

    stage_started_at = perf_counter()
    index_run_id = index_project(root, session, scanned_files=relational_files)
    logger.info(
        "Relational indexing stage completed: project_id=%s, "
        "index_run_id=%s, elapsed_seconds=%.3f",
        project_id_value,
        index_run_id,
        perf_counter() - stage_started_at,
    )

    stage_started_at = perf_counter()
    file_ids = repository.file_ids_for(files)
    changed_file_ids = [
        file_ids[item.relative_path] for item in file_change_plan.changed
    ]
    repository.upsert_files(file_change_plan.changed, file_change_plan.content_hashes)
    repository.clear_file_refinement_outputs(changed_file_ids)
    logger.info(
        "Changed semantic file rows prepared: project_id=%s, changed_files=%s, "
        "elapsed_seconds=%.3f",
        project_id_value,
        len(file_change_plan.changed),
        perf_counter() - stage_started_at,
    )

    changed_source_files = [
        item for item in source_files if item.relative_path in changed_paths
    ]
    changed_library_files = [
        item for item in library_files if item.relative_path in changed_paths
    ]

    errors: list[str] = []
    stage_started_at = perf_counter()
    source_outcomes = _extract_source_files(changed_source_files)
    source_facts = []
    for outcome in source_outcomes:
        if outcome.facts is not None:
            source_facts.append(outcome.facts)
        elif outcome.error is not None:
            errors.append(outcome.error)
            logger.warning(
                "Semantic source extraction failed: project_id=%s, file=%s, "
                "error=%s",
                project_id_value,
                outcome.relative_path,
                outcome.error,
            )
    logger.info(
        "Semantic source extraction completed: project_id=%s, processed=%s, "
        "fact_batches=%s, errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(changed_source_files),
        len(source_facts),
        len(errors),
        perf_counter() - stage_started_at,
    )

    library_started_at = perf_counter()
    library_outcomes = _extract_library_files(changed_library_files)
    library_facts = []
    for outcome in library_outcomes:
        if outcome.facts is not None:
            library_facts.append(outcome.facts)
        elif outcome.error is not None:
            errors.append(outcome.error)
            logger.warning(
                "Library extraction failed: project_id=%s, file=%s, error=%s",
                project_id_value,
                outcome.relative_path,
                outcome.error,
            )
    logger.info(
        "Library extraction completed: project_id=%s, processed=%s, "
        "fact_batches=%s, errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(changed_library_files),
        len(library_facts),
        len(errors),
        perf_counter() - library_started_at,
    )

    stage_started_at = perf_counter()
    facts = _normalize_entity_table_edges(merge_facts([*source_facts, *library_facts]))
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
    if facts.symbols or facts.edges:
        symbols_upserted, edges_upserted, key_to_id = repository.upsert_facts(
            facts, file_ids
        )
    else:
        symbols_upserted, edges_upserted, key_to_id = 0, 0, {}
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
    unchanged_file_ids = [
        file_ids[item.relative_path] for item in file_change_plan.unchanged
    ]
    missing_embedding_chunks = repository.chunks_missing_embeddings(
        unchanged_file_ids,
        embedder,
    )
    logger.info(
        "Embedding stage started: project_id=%s, model=%s, dimensions=%s, "
        "changed_chunks=%s, missing_embeddings=%s",
        project_id_value,
        embedder.model_name,
        embedder.dimensions,
        len(facts.chunks),
        len(missing_embedding_chunks),
    )
    stage_started_at = perf_counter()
    job = repository.create_embedding_job(
        index_run_id,
        embedder,
        len(facts.chunks) + len(missing_embedding_chunks),
    )
    chunks_upserted, embeddings_upserted = repository.upsert_chunks_and_embeddings(
        facts.chunks,
        file_ids,
        key_to_id,
        index_run_id,
        embedder,
    )
    missing_embeddings_upserted = repository.upsert_embeddings_for_existing_chunks(
        missing_embedding_chunks,
        embedder,
    )
    embeddings_upserted += missing_embeddings_upserted
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
        files_seen=len(current_paths),
        symbols_upserted=symbols_upserted,
        edges_upserted=edges_upserted,
        chunks_upserted=chunks_upserted,
        embeddings_upserted=embeddings_upserted,
        errors=errors,
    )


def _extract_source_files(files: list[RefinedFile]) -> list[_ExtractionOutcome]:
    if not files:
        return []
    with ThreadPoolExecutor(max_workers=REFINE_EXTRACT_WORKERS) as executor:
        return list(executor.map(_extract_source_file, files))


def _extract_library_files(files: list[RefinedFile]) -> list[_ExtractionOutcome]:
    if not files:
        return []
    with ThreadPoolExecutor(max_workers=REFINE_EXTRACT_WORKERS) as executor:
        return list(executor.map(_extract_library_file, files))


def _extract_source_file(item: RefinedFile) -> _ExtractionOutcome:
    try:
        facts = extract_file_facts(
            item.path,
            item.relative_path,
            item.language,
            item.module_name,
            item.service_name,
        )
    except Exception as exc:  # noqa: BLE001
        return _ExtractionOutcome(
            relative_path=item.relative_path,
            error=f"{item.relative_path}: {exc}",
        )
    return _ExtractionOutcome(relative_path=item.relative_path, facts=facts)


def _extract_library_file(item: RefinedFile) -> _ExtractionOutcome:
    try:
        facts = extract_library_file_facts(item.path, item.relative_path)
    except Exception as exc:  # noqa: BLE001
        return _ExtractionOutcome(
            relative_path=item.relative_path,
            error=f"{item.relative_path}: {exc}",
        )
    return _ExtractionOutcome(relative_path=item.relative_path, facts=facts)


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
