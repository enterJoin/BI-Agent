"""Project indexing orchestration."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from sqlalchemy import Table, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from springgraph.hashing import content_hash, file_row_id, project_id
from springgraph.models import Edge, File, IndexRun, Project, Symbol, UnresolvedRef
from springgraph.refinement.relational.domain import (
    EdgeData,
    ExtractionResult,
    SymbolData,
    UnresolvedRefData,
)
from springgraph.refinement.relational.extractor.java_tree_sitter import (
    extract_java_file,
)
from springgraph.refinement.relational.resolver import resolve_project_refs
from springgraph.refinement.relational.resource_config import extract_config_file
from springgraph.refinement.relational.scanner import ScannedFile, scan_project

logger = logging.getLogger(__name__)


def index_project(
    root: Path,
    session: Session,
    scanned_files: list[ScannedFile] | None = None,
) -> int:
    """Index a Java Spring Boot project or workspace path."""
    root = root.resolve()
    project_id_value = project_id(root)
    started_at = perf_counter()
    logger.info(
        "Relational indexing started: project_id=%s, root=%s",
        project_id_value,
        root,
    )
    _upsert_project(session, project_id_value, root)
    run = IndexRun(project_id=project_id_value, status="running")
    session.add(run)
    session.flush()

    errors: list[str] = []
    scan_started_at = perf_counter()
    if scanned_files is None:
        scanned_files = scan_project(root)
    logger.info(
        "Relational scan completed: project_id=%s, files=%s, "
        "elapsed_seconds=%.3f",
        project_id_value,
        len(scanned_files),
        perf_counter() - scan_started_at,
    )
    files_indexed = 0
    extraction_started_at = perf_counter()
    skipped_unchanged = 0
    for scanned_file in scanned_files:
        try:
            with session.begin_nested():
                changed = _upsert_file(session, project_id_value, scanned_file)
                if not changed:
                    skipped_unchanged += 1
                    continue
                _delete_file_symbols(
                    session, project_id_value, scanned_file.relative_path
                )
                result = _extract_file(project_id_value, scanned_file)
                _insert_symbols(
                    session,
                    project_id_value,
                    scanned_file.relative_path,
                    result.symbols,
                )
                _insert_edges(session, project_id_value, result.edges)
                _insert_unresolved(
                    session,
                    project_id_value,
                    scanned_file.relative_path,
                    result.unresolved_refs,
                )
                if result.errors:
                    _mark_file_error(
                        session,
                        project_id_value,
                        scanned_file.relative_path,
                        "; ".join(result.errors),
                    )
                files_indexed += 1
                errors.extend(result.errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{scanned_file.relative_path}: {exc}")
            logger.warning(
                "Relational file indexing failed: project_id=%s, file=%s, "
                "error=%s",
                project_id_value,
                scanned_file.relative_path,
                exc,
            )
            with session.begin_nested():
                _upsert_file_error(session, project_id_value, scanned_file, str(exc))
    logger.info(
        "Relational file indexing completed: project_id=%s, processed=%s, "
        "indexed=%s, skipped_unchanged=%s, errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(scanned_files),
        files_indexed,
        skipped_unchanged,
        len(errors),
        perf_counter() - extraction_started_at,
    )

    if files_indexed:
        resolve_started_at = perf_counter()
        logger.info(
            "Relational reference resolution started: project_id=%s",
            project_id_value,
        )
        resolve_project_refs(session, project_id_value)
        logger.info(
            "Relational reference resolution completed: project_id=%s, "
            "elapsed_seconds=%.3f",
            project_id_value,
            perf_counter() - resolve_started_at,
        )
    else:
        logger.info(
            "Relational reference resolution skipped: project_id=%s, "
            "reason=no_changed_files",
            project_id_value,
        )
    run.status = "completed" if not errors else "completed_with_errors"
    run.finished_at = datetime.now(tz=UTC)
    run.files_seen = len(scanned_files)
    run.files_indexed = files_indexed
    run.symbols_created = _count(session, Symbol, project_id_value)
    run.edges_created = _count(session, Edge, project_id_value)
    run.unresolved_created = _count(session, UnresolvedRef, project_id_value)
    run.errors = errors
    session.flush()
    logger.info(
        "Relational indexing completed: project_id=%s, index_run_id=%s, "
        "files_seen=%s, files_indexed=%s, symbols=%s, edges=%s, "
        "unresolved_refs=%s, errors=%s, elapsed_seconds=%.3f",
        project_id_value,
        run.id,
        run.files_seen,
        run.files_indexed,
        run.symbols_created,
        run.edges_created,
        run.unresolved_created,
        len(errors),
        perf_counter() - started_at,
    )
    return run.id


def _upsert_project(session: Session, project_id_value: str, root: Path) -> None:
    stmt = insert(Project).values(
        id=project_id_value,
        root_path=str(root),
        name=root.name,
        updated_at=datetime.now(tz=UTC),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Project.id],
        set_={
            "root_path": str(root),
            "name": root.name,
            "updated_at": datetime.now(tz=UTC),
        },
    )
    session.execute(stmt)


def _upsert_file(
    session: Session, project_id_value: str, scanned_file: ScannedFile
) -> bool:
    content = _read_text(scanned_file.path)
    digest = content_hash(content)
    row_id = file_row_id(project_id_value, scanned_file.relative_path)
    existing = session.get(File, row_id)
    if existing and existing.content_hash == digest and existing.error is None:
        return False

    values = {
        "id": row_id,
        "project_id": project_id_value,
        "path": scanned_file.relative_path,
        "module_name": scanned_file.module_name,
        "service_name": scanned_file.service_name,
        "language": scanned_file.language,
        "content_hash": digest,
        "size_bytes": scanned_file.size_bytes,
        "modified_at": scanned_file.modified_at,
        "indexed_at": datetime.now(tz=UTC),
        "error": None,
    }
    stmt = insert(File).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[File.project_id, File.path],
        set_=values,
    )
    session.execute(stmt)
    return True


def _upsert_file_error(
    session: Session,
    project_id_value: str,
    scanned_file: ScannedFile,
    error: str,
) -> None:
    content = _read_text(scanned_file.path)
    values = {
        "id": file_row_id(project_id_value, scanned_file.relative_path),
        "project_id": project_id_value,
        "path": scanned_file.relative_path,
        "module_name": scanned_file.module_name,
        "service_name": scanned_file.service_name,
        "language": scanned_file.language,
        "content_hash": content_hash(content),
        "size_bytes": scanned_file.size_bytes,
        "modified_at": scanned_file.modified_at,
        "indexed_at": datetime.now(tz=UTC),
        "error": error,
    }
    stmt = insert(File).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[File.project_id, File.path],
        set_=values,
    )
    session.execute(stmt)


def _delete_file_symbols(
    session: Session, project_id_value: str, relative_path: str
) -> None:
    file_id = file_row_id(project_id_value, relative_path)
    session.execute(
        delete(Symbol).where(
            Symbol.project_id == project_id_value, Symbol.file_id == file_id
        )
    )


def _extract_file(project_id_value: str, scanned_file: ScannedFile) -> ExtractionResult:
    if scanned_file.language == "java":
        return extract_java_file(
            project_id_value,
            scanned_file.path,
            scanned_file.relative_path,
            scanned_file.module_name,
            scanned_file.service_name,
        )
    return extract_config_file(
        project_id_value,
        scanned_file.path,
        scanned_file.relative_path,
        scanned_file.module_name,
        scanned_file.service_name,
    )


def _insert_symbols(
    session: Session,
    project_id_value: str,
    relative_path: str,
    symbols: list[SymbolData],
) -> None:
    if not symbols:
        return
    file_id = file_row_id(project_id_value, relative_path)
    timestamp = datetime.now(tz=UTC)
    values_by_id: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        values_by_id[symbol.id] = {
            "id": symbol.id,
            "project_id": project_id_value,
            "file_id": file_id,
            "kind": symbol.kind,
            "name": symbol.name,
            "qualified_name": symbol.qualified_name,
            "language": symbol.language,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
            "start_column": symbol.start_column,
            "end_column": symbol.end_column,
            "signature": symbol.signature,
            "docstring": symbol.docstring,
            "annotations": symbol.annotations,
            "modifiers": symbol.modifiers,
            "metadata": symbol.metadata,
            "updated_at": timestamp,
        }
    values_list = list(values_by_id.values())
    table = cast(Table, Symbol.__table__)
    stmt = insert(table).values(values_list)
    stmt = stmt.on_conflict_do_update(
        index_elements=[table.c.id],
        set_={key: getattr(stmt.excluded, key) for key in values_list[0]},
    )
    session.execute(stmt)


def _insert_edges(
    session: Session, project_id_value: str, edges: list[EdgeData]
) -> None:
    if not edges:
        return
    values_by_key: dict[
        tuple[str, str, str, str, int | None, int | None],
        dict[str, Any],
    ] = {}
    for edge in edges:
        key = (
            project_id_value,
            edge.source_id,
            edge.target_id,
            edge.kind,
            edge.line,
            edge.column_no,
        )
        values_by_key[key] = {
            "project_id": project_id_value,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "kind": edge.kind,
            "line": edge.line,
            "column_no": edge.column_no,
            "confidence": edge.confidence,
            "resolved_by": edge.resolved_by,
            "metadata": edge.metadata,
        }
    values_list = list(values_by_key.values())
    table = cast(Table, Edge.__table__)
    stmt = insert(table).values(values_list)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=[
            table.c.project_id,
            table.c.source_id,
            table.c.target_id,
            table.c.kind,
            table.c.line,
            table.c.column_no,
        ]
    )
    session.execute(stmt)


def _insert_unresolved(
    session: Session,
    project_id_value: str,
    relative_path: str,
    unresolved_refs: list[UnresolvedRefData],
) -> None:
    if not unresolved_refs:
        return
    file_id = file_row_id(project_id_value, relative_path)
    values_by_key: dict[tuple[str, str, str, str, int, int], dict[str, Any]] = {}
    for ref in unresolved_refs:
        key = (
            project_id_value,
            ref.from_symbol_id,
            ref.reference_name,
            ref.reference_kind,
            ref.line,
            ref.column_no,
        )
        values_by_key[key] = {
            "project_id": project_id_value,
            "from_symbol_id": ref.from_symbol_id,
            "file_id": file_id,
            "reference_name": ref.reference_name,
            "reference_kind": ref.reference_kind,
            "line": ref.line,
            "column_no": ref.column_no,
            "candidates": ref.candidates,
            "metadata": ref.metadata,
        }
    values_list = list(values_by_key.values())
    table = cast(Table, UnresolvedRef.__table__)
    stmt = insert(table).values(values_list)
    stmt = stmt.on_conflict_do_update(
        index_elements=[
            table.c.project_id,
            table.c.from_symbol_id,
            table.c.reference_name,
            table.c.reference_kind,
            table.c.line,
            table.c.column_no,
        ],
        set_={key: getattr(stmt.excluded, key) for key in values_list[0]},
    )
    session.execute(stmt)


def _mark_file_error(
    session: Session, project_id_value: str, relative_path: str, error: str
) -> None:
    row = session.get(File, file_row_id(project_id_value, relative_path))
    if row:
        row.error = error


def _count(session: Session, model: type[Any], project_id_value: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.project_id == project_id_value)
        )
        or 0
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")
