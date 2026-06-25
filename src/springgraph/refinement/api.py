"""Public API for semantic Java system refinement."""

from pathlib import Path

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
    project_id_value = project_id(root)
    index_run_id = index_project(root, session)

    repository = RefinementRepository(session, project_id_value)
    repository.upsert_project(root)
    repository.clear_refinement_outputs()
    source_files = scan_refinement_files(root)
    library_files = scan_library_files(root)
    files = source_files + library_files
    file_ids = repository.upsert_files(files)

    errors: list[str] = []
    per_file_facts: list[RefinementFacts] = []
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

    for item in library_files:
        try:
            per_file_facts.append(
                extract_library_file_facts(item.path, item.relative_path)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.relative_path}: {exc}")

    facts = _normalize_entity_table_edges(merge_facts(per_file_facts))
    symbols_upserted, edges_upserted, key_to_id = repository.upsert_facts(
        facts, file_ids
    )

    embedder = create_embedder()
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
