"""Second-pass unresolved reference resolver."""

from typing import cast

from sqlalchemy import Table, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from springgraph.models import Edge, Symbol, UnresolvedRef


def resolve_project_refs(session: Session, project_id_value: str) -> int:
    """Resolve unresolved references into edges."""
    symbols = list(
        session.scalars(select(Symbol).where(Symbol.project_id == project_id_value))
    )
    refs = list(
        session.scalars(
            select(UnresolvedRef).where(UnresolvedRef.project_id == project_id_value)
        )
    )
    resolved_ids: list[int] = []
    resolved_count = 0
    for ref in refs:
        target = _resolve_target(ref, symbols)
        if target is None:
            continue
        table = cast(Table, Edge.__table__)
        stmt = insert(table).values(
            project_id=project_id_value,
            source_id=ref.from_symbol_id,
            target_id=target.id,
            kind=ref.reference_kind,
            line=ref.line,
            column_no=ref.column_no,
            confidence=0.85,
            resolved_by="resolver",
            metadata={"reference_name": ref.reference_name, **ref.meta},
        )
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
        resolved_ids.append(ref.id)
        resolved_count += 1
    if resolved_ids:
        session.execute(delete(UnresolvedRef).where(UnresolvedRef.id.in_(resolved_ids)))
    return resolved_count


def _resolve_target(ref: UnresolvedRef, symbols: list[Symbol]) -> Symbol | None:
    if ref.reference_kind == "calls":
        return _resolve_call(ref.reference_name, symbols)
    if ref.reference_kind == "instantiates":
        return _resolve_constructor(ref.reference_name, symbols)
    if ref.reference_kind in {"injects", "extends", "implements", "references"}:
        return _resolve_type(ref.reference_name, symbols)
    return None


def _resolve_call(reference_name: str, symbols: list[Symbol]) -> Symbol | None:
    if "." in reference_name:
        owner, method = reference_name.rsplit(".", maxsplit=1)
        for symbol in symbols:
            if (
                symbol.kind == "method"
                and symbol.name == method
                and symbol.qualified_name.rsplit(".", maxsplit=1)[0].endswith(owner)
            ):
                return symbol
    method_name = reference_name.rsplit(".", maxsplit=1)[-1]
    matches = [
        symbol
        for symbol in symbols
        if symbol.kind == "method" and symbol.name == method_name
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_constructor(reference_name: str, symbols: list[Symbol]) -> Symbol | None:
    for symbol in symbols:
        if symbol.kind == "constructor" and symbol.qualified_name.endswith(
            f".{reference_name}.<init>"
        ):
            return symbol
    return _resolve_type(reference_name, symbols)


def _resolve_type(reference_name: str, symbols: list[Symbol]) -> Symbol | None:
    target_kinds = {"class", "interface", "enum", "annotation"}
    for symbol in symbols:
        if symbol.kind in target_kinds and symbol.qualified_name == reference_name:
            return symbol
    matches = [
        symbol
        for symbol in symbols
        if symbol.kind in target_kinds
        and symbol.name == reference_name.rsplit(".", maxsplit=1)[-1]
    ]
    return matches[0] if len(matches) == 1 else None
