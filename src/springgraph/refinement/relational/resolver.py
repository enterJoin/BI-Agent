"""Second-pass unresolved reference resolver."""

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

from sqlalchemy import Table, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from springgraph.models import Edge, Symbol, UnresolvedRef

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000


@dataclass
class _SymbolIndex:
    methods_by_name: dict[str, list[Symbol]] = field(default_factory=dict)
    constructors_by_name: dict[str, list[Symbol]] = field(default_factory=dict)
    types_by_qualified_name: dict[str, Symbol] = field(default_factory=dict)
    types_by_name: dict[str, list[Symbol]] = field(default_factory=dict)


def resolve_project_refs(session: Session, project_id_value: str) -> int:
    """Resolve unresolved references into edges."""
    started_at = perf_counter()
    symbols = list(
        session.scalars(select(Symbol).where(Symbol.project_id == project_id_value))
    )
    refs = list(
        session.scalars(
            select(UnresolvedRef).where(UnresolvedRef.project_id == project_id_value)
        )
    )
    index = _build_symbol_index(symbols)
    logger.info(
        "Relational reference resolver loaded: project_id=%s, symbols=%s, "
        "unresolved_refs=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(symbols),
        len(refs),
        perf_counter() - started_at,
    )
    resolved_ids: list[int] = []
    edge_values_by_key: dict[
        tuple[str, str, str, str, int | None, int | None],
        dict[str, object],
    ] = {}
    for ref in refs:
        target = _resolve_target(ref, index)
        if target is None:
            continue
        key = (
            project_id_value,
            ref.from_symbol_id,
            target.id,
            ref.reference_kind,
            ref.line,
            ref.column_no,
        )
        edge_values_by_key[key] = {
            "project_id": project_id_value,
            "source_id": ref.from_symbol_id,
            "target_id": target.id,
            "kind": ref.reference_kind,
            "line": ref.line,
            "column_no": ref.column_no,
            "confidence": 0.85,
            "resolved_by": "resolver",
            "metadata": {"reference_name": ref.reference_name, **ref.meta},
        }
        resolved_ids.append(ref.id)
    _insert_resolved_edges(session, list(edge_values_by_key.values()))
    if resolved_ids:
        _delete_resolved_refs(session, resolved_ids)
    logger.info(
        "Relational reference resolver completed: project_id=%s, resolved_refs=%s, "
        "edges_upserted=%s, elapsed_seconds=%.3f",
        project_id_value,
        len(resolved_ids),
        len(edge_values_by_key),
        perf_counter() - started_at,
    )
    return len(resolved_ids)


def _build_symbol_index(symbols: list[Symbol]) -> _SymbolIndex:
    index = _SymbolIndex()
    target_kinds = {"class", "interface", "enum", "annotation"}
    for symbol in symbols:
        if symbol.kind == "method":
            index.methods_by_name.setdefault(symbol.name, []).append(symbol)
        elif symbol.kind == "constructor":
            class_name = _constructor_class_name(symbol.qualified_name)
            if class_name:
                index.constructors_by_name.setdefault(class_name, []).append(symbol)
        elif symbol.kind in target_kinds:
            index.types_by_qualified_name[symbol.qualified_name] = symbol
            index.types_by_name.setdefault(symbol.name, []).append(symbol)
    return index


def _resolve_target(ref: UnresolvedRef, index: _SymbolIndex) -> Symbol | None:
    if ref.reference_kind == "calls":
        return _resolve_call(ref.reference_name, index)
    if ref.reference_kind == "instantiates":
        return _resolve_constructor(ref.reference_name, index)
    if ref.reference_kind in {"injects", "extends", "implements", "references"}:
        return _resolve_type(ref.reference_name, index)
    return None


def _resolve_call(reference_name: str, index: _SymbolIndex) -> Symbol | None:
    if "." in reference_name:
        owner, method = reference_name.rsplit(".", maxsplit=1)
        for symbol in index.methods_by_name.get(method, []):
            if (
                symbol.qualified_name.rsplit(".", maxsplit=1)[0]
                .removesuffix(f".{method}")
                .endswith(owner)
            ):
                return symbol
    method_name = reference_name.rsplit(".", maxsplit=1)[-1]
    matches = index.methods_by_name.get(method_name, [])
    return matches[0] if len(matches) == 1 else None


def _resolve_constructor(reference_name: str, index: _SymbolIndex) -> Symbol | None:
    matches = index.constructors_by_name.get(reference_name)
    if matches and len(matches) == 1:
        return matches[0]
    return _resolve_type(reference_name, index)


def _resolve_type(reference_name: str, index: _SymbolIndex) -> Symbol | None:
    direct = index.types_by_qualified_name.get(reference_name)
    if direct is not None:
        return direct
    matches = index.types_by_name.get(reference_name.rsplit(".", maxsplit=1)[-1], [])
    return matches[0] if len(matches) == 1 else None


def _constructor_class_name(qualified_name: str) -> str | None:
    if not qualified_name.endswith(".<init>"):
        return None
    owner = qualified_name.rsplit(".", maxsplit=1)[0]
    return owner.rsplit(".", maxsplit=1)[-1]


def _insert_resolved_edges(
    session: Session,
    values_list: list[dict[str, object]],
) -> None:
    if not values_list:
        return
    table = cast(Table, Edge.__table__)
    for batch in _batches(values_list, BATCH_SIZE):
        stmt = insert(table).values(batch)
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


def _delete_resolved_refs(session: Session, resolved_ids: list[int]) -> None:
    for batch in _batches(resolved_ids, BATCH_SIZE):
        session.execute(delete(UnresolvedRef).where(UnresolvedRef.id.in_(batch)))


def _batches[T](items: list[T], batch_size: int) -> list[list[T]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]
