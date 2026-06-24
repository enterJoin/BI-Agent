"""Read-side query helpers."""

from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from springgraph.hashing import project_id
from springgraph.models import Edge, File, Project, Symbol, UnresolvedRef


def project_id_for_path(path: Path) -> str:
    """Return the project ID for a path."""
    return project_id(path.resolve())


def status(session: Session, root: Path) -> dict[str, int | str | None]:
    """Return project status counters."""
    project_id_value = project_id_for_path(root)
    project = session.get(Project, project_id_value)
    return {
        "project": project.name if project else None,
        "files": _count(session, File, project_id_value),
        "symbols": _count(session, Symbol, project_id_value),
        "edges": _count(session, Edge, project_id_value),
        "unresolved_refs": _count(session, UnresolvedRef, project_id_value),
    }


def search_symbols(session: Session, root: Path, query: str) -> list[Symbol]:
    """Search symbols by name or qualified name."""
    project_id_value = project_id_for_path(root)
    pattern = f"%{query}%"
    return list(
        session.scalars(
            select(Symbol)
            .where(Symbol.project_id == project_id_value)
            .where(
                or_(Symbol.name.ilike(pattern), Symbol.qualified_name.ilike(pattern))
            )
            .order_by(Symbol.kind, Symbol.qualified_name)
            .limit(100)
        )
    )


def routes(session: Session, root: Path) -> list[Symbol]:
    """Return route symbols."""
    project_id_value = project_id_for_path(root)
    return list(
        session.scalars(
            select(Symbol)
            .where(Symbol.project_id == project_id_value, Symbol.kind == "route")
            .order_by(Symbol.name)
        )
    )


def files(session: Session, root: Path) -> list[File]:
    """Return indexed files."""
    project_id_value = project_id_for_path(root)
    return list(
        session.scalars(
            select(File).where(File.project_id == project_id_value).order_by(File.path)
        )
    )


def resources(session: Session, root: Path) -> list[tuple[str, str, list[str]]]:
    """Return resource symbols grouped by services."""
    project_id_value = project_id_for_path(root)
    rows = session.execute(
        select(Symbol, Edge)
        .join(Edge, Edge.target_id == Symbol.id)
        .where(
            Symbol.project_id == project_id_value,
            Symbol.kind == "resource",
            Edge.kind == "uses_resource",
        )
        .order_by(Symbol.qualified_name)
    )
    grouped: dict[tuple[str, str], set[str]] = {}
    for symbol, edge in rows:
        resource_type = str(symbol.meta.get("resource_type", "resource"))
        normalized_name = str(symbol.meta.get("normalized_name", symbol.name))
        service_name = str(edge.meta.get("service_name") or "")
        grouped.setdefault((resource_type, normalized_name), set()).add(service_name)
    return [
        (
            resource_type,
            normalized_name,
            sorted(service for service in services if service),
        )
        for (resource_type, normalized_name), services in sorted(grouped.items())
    ]


def callers(session: Session, root: Path, symbol_query: str) -> list[Symbol]:
    """Return symbols that call or reference matching symbols."""
    project_id_value = project_id_for_path(root)
    targets = _matching_symbols(session, project_id_value, symbol_query)
    target_ids = [symbol.id for symbol in targets]
    if not target_ids:
        return []
    return list(
        session.scalars(
            select(Symbol)
            .join(Edge, Edge.source_id == Symbol.id)
            .where(Edge.project_id == project_id_value, Edge.target_id.in_(target_ids))
            .order_by(Symbol.qualified_name)
        )
    )


def callees(session: Session, root: Path, symbol_query: str) -> list[Symbol]:
    """Return symbols called or referenced by matching symbols."""
    project_id_value = project_id_for_path(root)
    sources = _matching_symbols(session, project_id_value, symbol_query)
    source_ids = [symbol.id for symbol in sources]
    if not source_ids:
        return []
    return list(
        session.scalars(
            select(Symbol)
            .join(Edge, Edge.target_id == Symbol.id)
            .where(Edge.project_id == project_id_value, Edge.source_id.in_(source_ids))
            .order_by(Symbol.qualified_name)
        )
    )


def impact(session: Session, root: Path, symbol_query: str, depth: int) -> list[Symbol]:
    """Return upstream callers to a bounded depth."""
    seen: set[str] = set()
    current = callers(session, root, symbol_query)
    result: list[Symbol] = []
    for _ in range(depth):
        next_level: list[Symbol] = []
        for symbol in current:
            if symbol.id in seen:
                continue
            seen.add(symbol.id)
            result.append(symbol)
            next_level.extend(callers(session, root, symbol.qualified_name))
        current = next_level
    return result


def _matching_symbols(
    session: Session, project_id_value: str, query: str
) -> list[Symbol]:
    pattern = f"%{query}%"
    return list(
        session.scalars(
            select(Symbol)
            .where(Symbol.project_id == project_id_value)
            .where(or_(Symbol.name == query, Symbol.qualified_name.ilike(pattern)))
            .order_by(Symbol.kind, Symbol.qualified_name)
        )
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
