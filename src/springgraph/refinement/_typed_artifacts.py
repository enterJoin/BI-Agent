"""Generate typed artifact vector chunks from relational code evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from springgraph.models import Edge, File, Symbol
from springgraph.refinement._types import ChunkFact

TYPED_ARTIFACT_TEMPLATE_VERSION = "typed-artifact-balanced-v1"

_DIRECT_KIND_TO_ARTIFACT_TYPE = {
    "bean": "bean",
    "cache_key": "cache_key",
    "class": "class",
    "config": "config",
    "constructor": "constructor",
    "data_contract": "data_contract",
    "db_column": "db_column",
    "db_table": "db_table",
    "enum": "enum",
    "interface": "interface",
    "mapper": "mapper",
    "mq_exchange": "mq_exchange",
    "mq_queue": "mq_queue",
    "mq_tag": "mq_tag",
    "mq_topic": "mq_topic",
    "oauth_provider": "oauth_provider",
    "permission_rule": "permission_rule",
    "remote_service": "remote_service",
    "resource": "resource",
    "route": "api_route",
    "service": "service",
    "sql_statement": "sql_statement",
}

_SUMMARY_KINDS = {
    "annotation_usage",
    "field",
    "method",
    "parameter",
}
_IGNORED_STRUCTURAL_KINDS = {"file", "import", "package"}
_STRUCTURAL_EDGE_KINDS = {"annotated_by", "annotates", "contains"}
_METHOD_ARTIFACT_EDGE_KINDS = {"maps_route"}


@dataclass(frozen=True)
class _SymbolRow:
    id: str
    kind: str
    name: str
    qualified_name: str
    language: str
    start_line: int
    end_line: int
    signature: str | None
    annotations: list[object]
    metadata: dict[str, object]
    file_path: str
    module_name: str | None
    service_name: str | None


@dataclass(frozen=True)
class _EdgeRow:
    source_id: str
    target_id: str
    kind: str
    line: int | None


def build_typed_artifact_chunks(
    session: Session,
    project_id: str,
    *,
    file_ids: set[str] | None = None,
    current_paths: set[str] | None = None,
) -> list[ChunkFact]:
    """Build balanced typed artifact chunks from stored relational symbols."""
    symbols = _load_symbols(session, project_id, file_ids, current_paths)
    if not symbols:
        return []
    symbol_by_id = {
        symbol_id: (kind, name, qualified_name)
        for symbol_id, kind, name, qualified_name in session.execute(
            select(Symbol.id, Symbol.kind, Symbol.name, Symbol.qualified_name).where(
                Symbol.project_id == project_id
            )
        )
    }
    edge_rows = _load_edges(session, project_id)
    method_artifact_symbol_ids = {
        edge.source_id
        for edge in edge_rows
        if edge.kind in _METHOD_ARTIFACT_EDGE_KINDS
    } | {
        edge.target_id
        for edge in edge_rows
        if edge.kind in _METHOD_ARTIFACT_EDGE_KINDS
    }
    outgoing: dict[str, list[_EdgeRow]] = defaultdict(list)
    incoming: dict[str, list[_EdgeRow]] = defaultdict(list)
    for edge in edge_rows:
        outgoing[edge.source_id].append(edge)
        incoming[edge.target_id].append(edge)

    chunks: list[ChunkFact] = []
    for symbol in symbols:
        artifact_type = _artifact_type(symbol, method_artifact_symbol_ids)
        if artifact_type is None:
            continue
        chunks.append(
            _symbol_chunk(
                symbol,
                artifact_type,
                outgoing=outgoing.get(symbol.id, []),
                incoming=incoming.get(symbol.id, []),
                symbol_by_id=symbol_by_id,
            )
        )
    chunks.extend(_summary_chunks(symbols))
    return chunks


def _load_symbols(
    session: Session,
    project_id: str,
    file_ids: set[str] | None,
    current_paths: set[str] | None,
) -> list[_SymbolRow]:
    statement = (
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
    )
    if file_ids is not None:
        statement = statement.where(Symbol.file_id.in_(file_ids))
    if current_paths is not None:
        statement = statement.where(File.path.in_(current_paths))
    rows = session.execute(statement.order_by(File.path, Symbol.start_line)).all()
    return [
        _SymbolRow(
            id=symbol.id,
            kind=symbol.kind,
            name=symbol.name,
            qualified_name=symbol.qualified_name,
            language=symbol.language,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            signature=symbol.signature,
            annotations=list(symbol.annotations or []),
            metadata=dict(symbol.meta or {}),
            file_path=file_row.path,
            module_name=file_row.module_name,
            service_name=file_row.service_name,
        )
        for symbol, file_row in rows
    ]


def _load_edges(
    session: Session,
    project_id: str,
) -> list[_EdgeRow]:
    rows = session.execute(
        select(Edge.source_id, Edge.target_id, Edge.kind, Edge.line)
        .where(Edge.project_id == project_id)
    ).all()
    return [
        _EdgeRow(
            source_id=str(source_id),
            target_id=str(target_id),
            kind=str(kind),
            line=line,
        )
        for source_id, target_id, kind, line in rows
    ]


def _artifact_type(
    symbol: _SymbolRow,
    method_artifact_symbol_ids: set[str],
) -> str | None:
    if symbol.kind == "annotation_usage":
        annotation = _metadata_text(symbol.metadata, "annotation")
        if annotation and annotation.lower() in {"xxljob", "scheduled"}:
            return "job_entrypoint"
        return None
    if symbol.kind == "method":
        return (
            "method"
            if _is_important_method(symbol, method_artifact_symbol_ids)
            else None
        )
    if symbol.kind in _SUMMARY_KINDS:
        return None
    if symbol.kind in _IGNORED_STRUCTURAL_KINDS:
        return None
    return _DIRECT_KIND_TO_ARTIFACT_TYPE.get(symbol.kind, symbol.kind)


def _is_important_method(
    symbol: _SymbolRow,
    method_artifact_symbol_ids: set[str],
) -> bool:
    if symbol.id in method_artifact_symbol_ids:
        return True
    if symbol.annotations:
        return True
    return False


def _symbol_chunk(
    symbol: _SymbolRow,
    artifact_type: str,
    *,
    outgoing: list[_EdgeRow],
    incoming: list[_EdgeRow],
    symbol_by_id: dict[str, tuple[str, str, str]],
) -> ChunkFact:
    title = f"{artifact_type}: {symbol.name}"
    metadata: dict[str, object] = {
        "artifact_type": artifact_type,
        "source": "relational_index",
        "symbol_id": symbol.id,
        "symbol_kind": symbol.kind,
        "qualified_name": symbol.qualified_name,
        "module_name": symbol.module_name,
        "service_name": symbol.service_name,
        "parser_version": TYPED_ARTIFACT_TEMPLATE_VERSION,
        "template_version": TYPED_ARTIFACT_TEMPLATE_VERSION,
    }
    lines = [
        f"Artifact type: {artifact_type}",
        f"Symbol kind: {symbol.kind}",
        f"Name: {symbol.name}",
        f"Qualified name: {symbol.qualified_name}",
        f"Source file: {symbol.file_path}:{symbol.start_line}-{symbol.end_line}",
    ]
    if symbol.module_name:
        lines.append(f"Module: {symbol.module_name}")
    if symbol.service_name:
        lines.append(f"Service: {symbol.service_name}")
    if symbol.signature:
        lines.append(f"Signature: {symbol.signature}")
    if symbol.annotations:
        lines.append(f"Annotations: {_compact_json(symbol.annotations)}")
    if symbol.metadata:
        lines.append(f"Metadata: {_compact_json(symbol.metadata)}")
    related_lines = _related_edge_lines(outgoing, incoming, symbol_by_id)
    if related_lines:
        lines.append("Related evidence:")
        lines.extend(related_lines)
    return ChunkFact(
        key=f"typed_artifact:{artifact_type}:{symbol.id}",
        file_path=symbol.file_path,
        symbol_key=None,
        chunk_type=f"artifact_{artifact_type}",
        title=title,
        content="\n".join(lines),
        language=symbol.language,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        metadata=metadata,
    )


def _related_edge_lines(
    outgoing: list[_EdgeRow],
    incoming: list[_EdgeRow],
    symbol_by_id: dict[str, tuple[str, str, str]],
) -> list[str]:
    lines: list[str] = []
    for edge in outgoing[:16]:
        target = symbol_by_id.get(edge.target_id)
        if target is None:
            continue
        target_kind, target_name, target_qualified = target
        lines.append(
            f"- outgoing {edge.kind}: {target_kind} {target_name} "
            f"({target_qualified})"
        )
    for edge in incoming[:8]:
        source = symbol_by_id.get(edge.source_id)
        if source is None:
            continue
        source_kind, source_name, source_qualified = source
        lines.append(
            f"- incoming {edge.kind}: {source_kind} {source_name} "
            f"({source_qualified})"
        )
    return lines


def _summary_chunks(symbols: list[_SymbolRow]) -> list[ChunkFact]:
    grouped: dict[tuple[str, str], list[_SymbolRow]] = defaultdict(list)
    for symbol in symbols:
        if symbol.kind in _SUMMARY_KINDS:
            grouped[(symbol.kind, symbol.file_path)].append(symbol)

    chunks: list[ChunkFact] = []
    for (kind, file_path), items in sorted(grouped.items()):
        if not items:
            continue
        module_name = items[0].module_name
        service_name = items[0].service_name
        names = _dedupe([item.name for item in items])[:240]
        qualified = _dedupe([item.qualified_name for item in items])[:80]
        artifact_type = f"{kind}_summary"
        content_lines = [
            f"Artifact type: {artifact_type}",
            f"Symbol kind summarized: {kind}",
            f"Source file: {file_path}",
        ]
        if module_name:
            content_lines.append(f"Module: {module_name}")
        if service_name:
            content_lines.append(f"Service: {service_name}")
        content_lines.append(f"Symbols: {', '.join(names)}")
        content_lines.append(f"Qualified symbols: {', '.join(qualified)}")
        chunks.append(
            ChunkFact(
                key=f"typed_artifact:{artifact_type}:{file_path}",
                file_path=file_path,
                symbol_key=None,
                chunk_type=f"artifact_{artifact_type}",
                title=f"{artifact_type}: {file_path}",
                content="\n".join(content_lines),
                language=items[0].language,
                start_line=None,
                end_line=None,
                metadata={
                    "artifact_type": artifact_type,
                    "source": "relational_index",
                    "symbol_kind": kind,
                    "symbol_count": len(items),
                    "module_name": module_name,
                    "service_name": service_name,
                    "parser_version": TYPED_ARTIFACT_TEMPLATE_VERSION,
                    "template_version": TYPED_ARTIFACT_TEMPLATE_VERSION,
                },
            )
        )
    return chunks


def _metadata_text(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
