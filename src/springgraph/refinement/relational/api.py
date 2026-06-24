"""Public entry point for relational Java project refinement."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from springgraph.db import session_scope
from springgraph.hashing import project_id
from springgraph.models import Edge, File, IndexRun, Symbol, UnresolvedRef
from springgraph.refinement.relational.indexer import index_project


@dataclass(frozen=True)
class RelationalRefinementResult:
    """Summary returned after refining a project into relational tables."""

    project_id: str
    index_run_id: int
    files: int
    symbols: int
    edges: int
    unresolved_refs: int
    status: str
    errors: list[str]


def refine_project(
    project_path: str | Path,
    session: Session | None = None,
) -> RelationalRefinementResult:
    """Refine a project directory into relational code graph tables."""
    root = Path(project_path).resolve()
    if session is None:
        with session_scope() as owned_session:
            return _refine_with_session(root, owned_session)
    return _refine_with_session(root, session)


def _refine_with_session(root: Path, session: Session) -> RelationalRefinementResult:
    run_id = index_project(root, session)
    session.flush()
    return _build_result(root, run_id, session)


def _build_result(
    root: Path,
    run_id: int,
    session: Session,
) -> RelationalRefinementResult:
    project_id_value = project_id(root)
    run = session.get(IndexRun, run_id)
    if run is None:
        raise RuntimeError(f"Index run was not created: {run_id}")
    return RelationalRefinementResult(
        project_id=project_id_value,
        index_run_id=run_id,
        files=_count(session, File, project_id_value),
        symbols=_count(session, Symbol, project_id_value),
        edges=_count(session, Edge, project_id_value),
        unresolved_refs=_count(session, UnresolvedRef, project_id_value),
        status=run.status,
        errors=[str(item) for item in run.errors],
    )


def _count(session: Session, model: Any, project_id_value: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.project_id == project_id_value)
        )
        or 0
    )
