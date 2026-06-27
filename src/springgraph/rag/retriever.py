"""Hybrid retrieval over vector chunks and relational code graph tables."""

from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, aliased

from springgraph.db import session_scope
from springgraph.models import Edge, File, Symbol
from springgraph.rag.schemas import RagEvidence, RagPlan
from springgraph.vector_search import VectorSearchError, search_project_vectors

_GENERIC_QUERY_TERMS = {
    "controller",
    "route",
    "RequestMapping",
    "calls",
    "service",
    "method",
    "resource",
    "database",
    "redis",
    "mq",
}


def retrieve_vector_evidence(
    project_path: Path,
    project_id_value: str,
    plan: RagPlan,
) -> tuple[list[RagEvidence], list[str]]:
    """Retrieve semantic chunk evidence."""
    if not plan.use_vector_search:
        return [], []

    warnings: list[str] = []
    evidence: list[RagEvidence] = []
    for query in plan.expanded_queries[:3]:
        try:
            result = search_project_vectors(
                query=query,
                project_id_value=project_id_value,
                project_path=project_path,
                limit=max(1, plan.top_k // 2),
            )
        except (VectorSearchError, ValueError) as exc:
            warnings.append(f"Vector retrieval skipped for query {query!r}: {exc}")
            continue
        for match in result.matches:
            evidence.append(
                RagEvidence(
                    evidence_type="vector_chunk",
                    source="vector",
                    file_path=match.file_path,
                    start_line=match.start_line,
                    end_line=match.end_line,
                    symbol=match.symbol_qualified_name,
                    score=match.score,
                    content_excerpt=_truncate(match.content, 500),
                    metadata={
                        "chunk_id": match.chunk_id,
                        "chunk_type": match.chunk_type,
                        "title": match.title,
                        "module_name": match.module_name,
                        "service_name": match.service_name,
                    },
                )
            )
    return _dedupe_evidence(evidence)[: plan.top_k], warnings


def retrieve_relational_evidence(
    project_id_value: str,
    plan: RagPlan,
) -> tuple[list[RagEvidence], list[str]]:
    """Retrieve evidence from symbols, routes, resources, and graph edges."""
    if not plan.use_relational_search:
        return [], []

    with session_scope() as session:
        symbols = _search_symbols(session, project_id_value, plan)
        files = _files_by_id(session, [symbol.file_id for symbol in symbols])
        evidence = [
            _symbol_evidence(symbol, files.get(symbol.file_id), "symbol")
            for symbol in symbols
        ]
        if plan.intent == "route_lookup":
            evidence.extend(_route_evidence(session, project_id_value, plan.top_k))
        if plan.intent == "resource_lookup":
            evidence.extend(_resource_evidence(session, project_id_value, plan.top_k))
        if plan.graph_depth > 0:
            evidence.extend(_relation_evidence(session, symbols[:5], files))
    return _dedupe_evidence(evidence)[: plan.top_k * 2], []


def _search_symbols(
    session: Session,
    project_id_value: str,
    plan: RagPlan,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    seen: set[str] = set()
    specific_queries = [
        item
        for item in plan.expanded_queries
        if item not in _GENERIC_QUERY_TERMS
    ]
    fallback_queries = [
        item
        for item in plan.expanded_queries
        if item in _GENERIC_QUERY_TERMS
    ]
    for query in [*specific_queries, *fallback_queries][:8]:
        pattern = f"%{query}%"
        rows = session.scalars(
            select(Symbol)
            .where(Symbol.project_id == project_id_value)
            .where(
                or_(
                    Symbol.name.ilike(pattern),
                    Symbol.qualified_name.ilike(pattern),
                )
            )
            .order_by(Symbol.kind, Symbol.qualified_name)
            .limit(plan.top_k)
        )
        for symbol in rows:
            if symbol.id in seen:
                continue
            seen.add(symbol.id)
            symbols.append(symbol)
            if len(symbols) >= plan.top_k:
                return symbols
    return symbols


def _route_evidence(
    session: Session,
    project_id_value: str,
    limit: int,
) -> list[RagEvidence]:
    rows = list(
        session.scalars(
            select(Symbol)
            .where(Symbol.project_id == project_id_value, Symbol.kind == "route")
            .order_by(Symbol.name)
            .limit(limit)
        )
    )
    files = _files_by_id(session, [symbol.file_id for symbol in rows])
    return [
        _symbol_evidence(symbol, files.get(symbol.file_id), "route")
        for symbol in rows
    ]


def _resource_evidence(
    session: Session,
    project_id_value: str,
    limit: int,
) -> list[RagEvidence]:
    rows = list(
        session.scalars(
            select(Symbol)
            .where(Symbol.project_id == project_id_value, Symbol.kind == "resource")
            .order_by(Symbol.qualified_name)
            .limit(limit)
        )
    )
    files = _files_by_id(session, [symbol.file_id for symbol in rows])
    return [
        _symbol_evidence(symbol, files.get(symbol.file_id), "resource")
        for symbol in rows
    ]


def _relation_evidence(
    session: Session,
    symbols: list[Symbol],
    files: dict[str, File],
) -> list[RagEvidence]:
    symbol_ids = [symbol.id for symbol in symbols]
    if not symbol_ids:
        return []
    source_symbol = aliased(Symbol)
    target_symbol = aliased(Symbol)
    edges = list(
        session.execute(
            select(Edge, source_symbol, target_symbol)
            .join(source_symbol, source_symbol.id == Edge.source_id)
            .join(target_symbol, target_symbol.id == Edge.target_id)
            .where(or_(Edge.source_id.in_(symbol_ids), Edge.target_id.in_(symbol_ids)))
            .limit(50)
        )
    )
    evidence: list[RagEvidence] = []
    for edge, source, target in edges:
        file_row = files.get(source.file_id) or files.get(target.file_id)
        evidence.append(
            RagEvidence(
                evidence_type=f"edge:{edge.kind}",
                source="relational",
                file_path=file_row.path if file_row else None,
                start_line=edge.line,
                end_line=edge.line,
                symbol=f"{source.qualified_name} -> {target.qualified_name}",
                score=float(edge.confidence),
                content_excerpt=f"{source.qualified_name} {edge.kind} "
                f"{target.qualified_name}",
                metadata={"edge_id": edge.id, "resolved_by": edge.resolved_by},
            )
        )
    return evidence


def _files_by_id(session: Session, file_ids: list[str]) -> dict[str, File]:
    unique_ids = list({file_id for file_id in file_ids if file_id})
    if not unique_ids:
        return {}
    rows = session.scalars(select(File).where(File.id.in_(unique_ids)))
    return {row.id: row for row in rows}


def _symbol_evidence(
    symbol: Symbol,
    file_row: File | None,
    evidence_type: str,
) -> RagEvidence:
    return RagEvidence(
        evidence_type=evidence_type,
        source="relational",
        file_path=file_row.path if file_row else None,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        symbol=symbol.qualified_name,
        score=1.0,
        content_excerpt=symbol.signature or symbol.name,
        metadata={
            "symbol_id": symbol.id,
            "kind": symbol.kind,
            "name": symbol.name,
            "annotations": _safe_json(symbol.annotations),
            "metadata": _safe_json(symbol.meta),
        },
    )


def _dedupe_evidence(items: list[RagEvidence]) -> list[RagEvidence]:
    seen: set[tuple[str, str | None, int | None, str | None]] = set()
    result: list[RagEvidence] = []
    for item in items:
        marker = (item.evidence_type, item.file_path, item.start_line, item.symbol)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def _safe_json(value: object) -> object:
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value
    return str(value)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
