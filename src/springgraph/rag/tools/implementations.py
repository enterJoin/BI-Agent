"""Composable Agentic RAG tool implementations."""

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import Text, or_, select

from springgraph.db import session_scope
from springgraph.models import Edge, File, Symbol
from springgraph.rag.config.loader import load_intent_configs
from springgraph.rag.config.models import ToolConfig
from springgraph.rag.intent import TABLE_RETRIEVAL_INTENTS, infer_query_intent
from springgraph.rag.library_hints import load_query_hints
from springgraph.rag.schemas import RagEvidence
from springgraph.rag.source_reader import check_source_path, read_source_snippets
from springgraph.rag.target_trace import (
    persistence_edge_kinds,
    source_priority,
    table_target_kinds,
    target_candidates,
)
from springgraph.rag.tools.schemas import RagTool, ToolInput, ToolResult

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$.:/-]*")
_MODULE_GENERIC_TERMS = {
    "artifact",
    "basemapper",
    "config",
    "dao",
    "data_contract",
    "database",
    "db_table",
    "entity",
    "jpa",
    "mapper",
    "mybatis",
    "repository",
    "sql",
    "table",
    "xml",
}
_GENERIC_QUERY_EXPANSIONS = {
    "\u63a5\u53e3": [
        "api",
        "endpoint",
        "route",
        "controller",
        "mapping",
        "RestController",
        "RequestMapping",
        "GetMapping",
        "PostMapping",
        "PutMapping",
        "DeleteMapping",
    ],
    "\u8def\u7531": ["route", "endpoint", "mapping", "controller"],
    "http": ["api", "endpoint", "route", "controller", "mapping"],
    "api": ["endpoint", "route", "controller", "mapping"],
    "endpoint": ["api", "route", "controller", "mapping"],
}
@dataclass(frozen=True)
class ArtifactSearchTool:
    """Batch search code artifacts from relational indexes."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        query = _expand_query_with_library(tool_input.query, tool_input)
        filters = tool_input.filters
        kinds = _string_list(filters.get("kinds"))
        module = _optional_string(filters.get("module"))
        path_contains = _optional_string(filters.get("path_contains"))
        terms = _query_terms(query)
        is_http_query = _looks_like_http_api_query(query)
        limit = _limit(tool_input.top_k, multiplier=20 if is_http_query else 4)

        with session_scope() as session:
            resolved_module = _resolve_module(
                session=session,
                project_id=tool_input.project_id,
                module=module,
                query=query,
                project_path=tool_input.project_path,
            )
            statement = (
                select(Symbol, File)
                .join(File, File.id == Symbol.file_id)
                .where(Symbol.project_id == tool_input.project_id)
            )
            if kinds:
                statement = statement.where(Symbol.kind.in_(kinds))
            if resolved_module:
                pattern = f"%{resolved_module}%"
                statement = statement.where(
                    or_(
                        File.module_name.ilike(pattern),
                        File.service_name.ilike(pattern),
                        File.path.ilike(pattern),
                    )
                )
            if path_contains:
                statement = statement.where(File.path.ilike(f"%{path_contains}%"))
            if terms:
                term_filters = []
                for term in terms[:10]:
                    pattern = f"%{term}%"
                    term_filters.extend(
                        [
                            Symbol.name.ilike(pattern),
                            Symbol.qualified_name.ilike(pattern),
                            Symbol.annotations.cast(Text).ilike(pattern),
                            Symbol.meta.cast(Text).ilike(pattern),
                            File.path.ilike(pattern),
                        ]
                    )
                statement = statement.where(or_(*term_filters))

            rows = session.execute(
                statement.order_by(File.path, Symbol.kind, Symbol.qualified_name).limit(
                    limit
                )
            ).all()
            if is_http_query:
                rows = sorted(
                    rows,
                    key=lambda row: _http_artifact_rank(row[0], row[1]),
                )

        evidence = [
            _symbol_evidence(symbol, file_row, "artifact")
            for symbol, file_row in rows
        ]
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Artifact search query={query!r}, filters={filters}, "
                f"resolved_module={resolved_module!r} returned {len(evidence)} "
                "evidence items."
            ),
            evidence=_dedupe(evidence),
        )


@dataclass(frozen=True)
class RelationSearchTool:
    """Batch search relations around matched symbols."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        seed_ids = _evidence_symbol_ids(tool_input.evidence)
        base_evidence: list[RagEvidence] = []
        if not seed_ids:
            artifact_result = ArtifactSearchTool(self.config).invoke(tool_input)
            seed_ids = _evidence_symbol_ids(artifact_result.evidence)
            base_evidence = artifact_result.evidence
        depth = max(1, min(tool_input.graph_depth, 4))
        edge_kinds = _string_list(tool_input.filters.get("edge_kinds"))

        evidence: list[RagEvidence] = [*base_evidence]
        frontier = set(seed_ids)
        seen_edges: set[int] = set()
        with session_scope() as session:
            for _ in range(depth):
                if not frontier:
                    break
                rows = session.execute(
                    _relation_statement(
                        frontier,
                        tool_input.project_id,
                        edge_kinds,
                    )
                ).all()
                next_frontier: set[str] = set()
                for edge, source, target, file_row in rows:
                    if edge.id in seen_edges:
                        continue
                    seen_edges.add(edge.id)
                    evidence.append(_edge_evidence(edge, source, target, file_row))
                    next_frontier.add(source.id)
                    next_frontier.add(target.id)
                frontier = next_frontier - frontier

        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Relation search depth={depth}, filters={tool_input.filters} "
                f"returned {len(evidence)} evidence items."
            ),
            evidence=_dedupe(evidence),
        )


@dataclass(frozen=True)
class AggregateQueryTool:
    """Aggregate evidence into task-specific structured facts."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        group_by = _optional_string(tool_input.filters.get("group_by"))
        explicit_intent = _optional_string(tool_input.filters.get("intent"))
        intent = infer_query_intent(
            query=tool_input.query,
            explicit_intent=explicit_intent,
            group_by=group_by,
            intent_configs=load_intent_configs(),
        )
        if intent in TABLE_RETRIEVAL_INTENTS:
            return _aggregate_tables(self.config, tool_input)
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                "Aggregate query had no supported group_by; "
                "returned input evidence."
            ),
            evidence=tool_input.evidence,
        )


@dataclass(frozen=True)
class TargetTraceTool:
    """Resolve one explicit target and trace its indexed relations."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        filters = tool_input.filters
        target_terms = _target_candidates(tool_input.query, filters)
        if not target_terms:
            return ToolResult(
                tool_name=self.config.name,
                summary=(
                    "Target trace found no explicit target in the query or filters."
                ),
                evidence=tool_input.evidence,
                warnings=["target_trace skipped: no explicit target"],
            )

        intent = _optional_string(filters.get("intent"))
        target_kind = _optional_string(filters.get("target_kind"))
        direction = _trace_direction(filters)
        edge_kinds = _trace_edge_kinds(intent, filters)
        limit = _limit(tool_input.top_k, multiplier=6)

        with session_scope() as session:
            target_rows = _resolve_target_symbols(
                session=session,
                project_id=tool_input.project_id,
                target_terms=target_terms,
                target_kind=target_kind,
                limit=limit,
            )
            relation_rows = _trace_target_relations(
                session=session,
                project_id=tool_input.project_id,
                target_rows=target_rows,
                direction=direction,
                edge_kinds=edge_kinds,
                limit=limit,
            )

        evidence = _target_match_evidence(target_rows)
        evidence.extend(_target_relation_evidence(relation_rows))
        evidence = _prioritize_trace_evidence(_dedupe(evidence))
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Target trace targets={target_terms}, target_kind={target_kind!r}, "
                f"direction={direction!r}, edge_kinds={edge_kinds} returned "
                f"{len(evidence)} evidence items."
            ),
            evidence=evidence,
        )


@dataclass(frozen=True)
class SourceReadTool:
    """Read source snippets for selected evidence."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        if not tool_input.source_available:
            return ToolResult(
                tool_name=self.config.name,
                summary="Source reading skipped because source is unavailable.",
                warnings=["source_read skipped: source_available=false"],
            )
        allowed, reason = check_source_path(tool_input.project_path)
        if not allowed:
            return ToolResult(
                tool_name=self.config.name,
                summary=f"Source reading skipped: {reason}.",
                warnings=[f"source_read skipped: {reason}"],
            )
        snippets, warnings = read_source_snippets(
            tool_input.project_path,
            tool_input.evidence,
            max_files=tool_input.max_source_files,
            max_lines=tool_input.max_source_lines,
            line_padding=tool_input.source_line_padding,
        )
        return ToolResult(
            tool_name=self.config.name,
            summary=f"Source read returned {len(snippets)} snippets.",
            source_snippets=snippets,
            warnings=warnings,
        )


def create_tool(config: ToolConfig) -> RagTool:
    """Create one tool from registration metadata."""
    if config.name == "artifact_search":
        return ArtifactSearchTool(config)
    if config.name == "relation_search":
        return RelationSearchTool(config)
    if config.name == "aggregate_query":
        return AggregateQueryTool(config)
    if config.name == "target_trace":
        return TargetTraceTool(config)
    if config.name == "source_read":
        return SourceReadTool(config)
    raise ValueError(f"Unsupported RAG tool implementation: {config.name}")


def _aggregate_tables(config: ToolConfig, tool_input: ToolInput) -> ToolResult:
    module = _optional_string(tool_input.filters.get("module"))
    path_contains = _optional_string(tool_input.filters.get("path_contains"))
    query = _expand_query_with_library(tool_input.query, tool_input)
    inferred_module = _infer_module(tool_input.evidence)
    limit = _limit(tool_input.top_k, multiplier=8)

    with session_scope() as session:
        resolved_module = _resolve_module(
            session=session,
            project_id=tool_input.project_id,
            module=module or inferred_module,
            query=query,
            project_path=tool_input.project_path,
        )
        statement = (
            select(Symbol, File)
            .join(File, File.id == Symbol.file_id)
            .where(Symbol.project_id == tool_input.project_id)
            .where(Symbol.kind.in_(["db_table", "data_contract"]))
        )
        if resolved_module:
            pattern = f"%{resolved_module}%"
            statement = statement.where(
                or_(
                    File.module_name.ilike(pattern),
                    File.service_name.ilike(pattern),
                    File.path.ilike(pattern),
                    Symbol.qualified_name.ilike(pattern),
                )
            )
        elif path_contains:
            statement = statement.where(File.path.ilike(f"%{path_contains}%"))
        rows = session.execute(
            statement.order_by(File.path, Symbol.kind, Symbol.name).limit(limit)
        ).all()

    evidence: list[RagEvidence] = []
    seen_tables: set[str] = set()
    for symbol, file_row in rows:
        table_name = _metadata_string(symbol.meta, "table") or symbol.name
        marker = f"{table_name}|{file_row.path}"
        if marker in seen_tables:
            continue
        seen_tables.add(marker)
        evidence.append(
            RagEvidence(
                evidence_type="table_usage",
                source="aggregate",
                file_path=file_row.path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol=symbol.qualified_name,
                score=1.0,
                content_excerpt=(
                    f"table={table_name}; artifact={symbol.name}; "
                    f"kind={symbol.kind}; module={file_row.module_name}; "
                    f"service={file_row.service_name}"
                ),
                metadata={
                    "table": table_name,
                    "artifact_kind": symbol.kind,
                    "module_name": file_row.module_name,
                    "service_name": file_row.service_name,
                },
            )
        )
    return ToolResult(
        tool_name=config.name,
        summary=(
            f"Aggregate table query query={query!r}, module={module!r}, "
            f"resolved_module={resolved_module!r} returned "
            f"{len(evidence)} table evidence items."
        ),
        evidence=evidence,
    )


def _target_candidates(query: str, filters: dict[str, Any]) -> list[str]:
    values: list[str] = []
    explicit_target = _optional_string(filters.get("target"))
    if explicit_target:
        values.append(explicit_target)
    values.extend(_string_list(filters.get("targets")))
    return target_candidates(query, values)


def _trace_direction(filters: dict[str, Any]) -> str:
    direction = _optional_string(filters.get("direction"))
    if direction in {"incoming", "outgoing", "both"}:
        return direction
    return "incoming"


def _trace_edge_kinds(intent: str | None, filters: dict[str, Any]) -> list[str]:
    edge_kinds = _string_list(filters.get("edge_kinds"))
    if edge_kinds:
        return edge_kinds
    if intent == "persistence_location":
        return persistence_edge_kinds()
    return []


def _resolve_target_symbols(
    session: Any,
    project_id: str,
    target_terms: list[str],
    target_kind: str | None,
    limit: int,
) -> list[tuple[Symbol, File]]:
    if not target_terms:
        return []
    filters = []
    for term in target_terms[:8]:
        pattern = f"%{term}%"
        filters.extend(
            [
                Symbol.name.ilike(pattern),
                Symbol.qualified_name.ilike(pattern),
                Symbol.meta.cast(Text).ilike(pattern),
                File.path.ilike(pattern),
            ]
        )
    statement = (
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
        .where(or_(*filters))
    )
    if target_kind:
        statement = statement.where(Symbol.kind == target_kind)
    rows = session.execute(statement.limit(limit)).all()
    return _rank_target_rows(rows, target_terms)


def _rank_target_rows(
    rows: list[tuple[Symbol, File]],
    target_terms: list[str],
) -> list[tuple[Symbol, File]]:
    def rank(row: tuple[Symbol, File]) -> tuple[int, str, str]:
        symbol, file_row = row
        haystack = _target_haystack(symbol, file_row)
        exact = any(_target_exact_match(symbol, term) for term in target_terms)
        table_match = symbol.kind in table_target_kinds() and exact
        partial = any(term.lower() in haystack for term in target_terms)
        if table_match:
            priority = 0
        elif exact:
            priority = 1
        elif partial:
            priority = 2
        else:
            priority = 5
        return (priority, file_row.path, symbol.qualified_name)

    ranked = sorted(rows, key=rank)
    return ranked


def _target_haystack(symbol: Symbol, file_row: File) -> str:
    return " ".join(
        [
            symbol.name,
            symbol.qualified_name,
            str(_json_safe(symbol.meta)),
            file_row.path,
        ]
    ).lower()


def _target_exact_match(symbol: Symbol, term: str) -> bool:
    lowered = term.lower()
    metadata_table = _metadata_string(symbol.meta, "table")
    values = [symbol.name, symbol.qualified_name]
    if metadata_table is not None:
        values.append(metadata_table)
    return any(value.lower() == lowered for value in values)


def _trace_target_relations(
    session: Any,
    project_id: str,
    target_rows: list[tuple[Symbol, File]],
    direction: str,
    edge_kinds: list[str],
    limit: int,
) -> list[tuple[Edge, Symbol, Symbol, File]]:
    target_ids = _trace_target_ids(target_rows)
    if not target_ids:
        return []
    from sqlalchemy.orm import aliased

    source = aliased(Symbol)
    target = aliased(Symbol)
    file_alias = aliased(File)
    statement = (
        select(Edge, source, target, file_alias)
        .join(source, source.id == Edge.source_id)
        .join(target, target.id == Edge.target_id)
        .join(file_alias, file_alias.id == source.file_id)
        .where(Edge.project_id == project_id)
        .limit(limit)
    )
    relation_filters = []
    if direction in {"incoming", "both"}:
        relation_filters.append(Edge.target_id.in_(target_ids))
    if direction in {"outgoing", "both"}:
        relation_filters.append(Edge.source_id.in_(target_ids))
    if relation_filters:
        statement = statement.where(or_(*relation_filters))
    if edge_kinds:
        statement = statement.where(Edge.kind.in_(edge_kinds))
    rows = session.execute(statement).all()
    return _rank_relation_rows(rows)


def _trace_target_ids(target_rows: list[tuple[Symbol, File]]) -> list[str]:
    result: list[str] = []
    for symbol, _ in target_rows:
        if symbol.kind in {"db_column"}:
            continue
        if symbol.id not in result:
            result.append(symbol.id)
    return result


def _rank_relation_rows(
    rows: list[tuple[Edge, Symbol, Symbol, File]],
) -> list[tuple[Edge, Symbol, Symbol, File]]:
    return sorted(
        rows,
        key=lambda row: (
            source_priority(f"edge:{row[0].kind}"),
            row[3].path,
            row[0].line or 0,
        ),
    )


def _target_match_evidence(rows: list[tuple[Symbol, File]]) -> list[RagEvidence]:
    evidence: list[RagEvidence] = []
    for symbol, file_row in rows:
        table_name = _metadata_string(symbol.meta, "table")
        details = [
            f"target={symbol.qualified_name}",
            f"kind={symbol.kind}",
        ]
        if table_name:
            details.append(f"table={table_name}")
        evidence.append(
            RagEvidence(
                evidence_type="target_match",
                source="target_trace",
                file_path=file_row.path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol=symbol.qualified_name,
                score=1.0,
                content_excerpt="; ".join(details),
                metadata={
                    "symbol_id": symbol.id,
                    "kind": symbol.kind,
                    "name": symbol.name,
                    "table": table_name,
                    "module_name": file_row.module_name,
                    "service_name": file_row.service_name,
                    "metadata": _json_safe(symbol.meta),
                },
            )
        )
    return evidence


def _target_relation_evidence(
    rows: list[tuple[Edge, Symbol, Symbol, File]],
) -> list[RagEvidence]:
    evidence: list[RagEvidence] = []
    for edge, source, target, file_row in rows:
        table_name = _metadata_string(target.meta, "table")
        evidence.append(
            RagEvidence(
                evidence_type=f"target_relation:{edge.kind}",
                source="target_trace",
                file_path=file_row.path,
                start_line=edge.line,
                end_line=edge.line,
                symbol=f"{source.qualified_name} -> {target.qualified_name}",
                score=float(edge.confidence),
                content_excerpt=(
                    f"{source.qualified_name} {edge.kind} "
                    f"{target.qualified_name}; source_kind={source.kind}; "
                    f"target_kind={target.kind}"
                ),
                metadata={
                    "edge_id": edge.id,
                    "edge_kind": edge.kind,
                    "source_symbol_id": source.id,
                    "target_symbol_id": target.id,
                    "source_symbol": source.qualified_name,
                    "target_symbol": target.qualified_name,
                    "source_kind": source.kind,
                    "target_kind": target.kind,
                    "table": table_name,
                    "resolved_by": edge.resolved_by,
                    "metadata": _json_safe(edge.meta),
                },
            )
        )
    return evidence


def _prioritize_trace_evidence(items: list[RagEvidence]) -> list[RagEvidence]:
    return sorted(
        items,
        key=lambda item: (
            source_priority(item.evidence_type),
            item.file_path or "",
            item.start_line or 0,
            item.symbol or "",
        ),
    )


def _relation_statement(
    frontier: set[str],
    project_id: str,
    edge_kinds: list[str],
) -> Any:
    from sqlalchemy.orm import aliased

    source = aliased(Symbol)
    target = aliased(Symbol)
    file_alias = aliased(File)
    statement = (
        select(Edge, source, target, file_alias)
        .join(source, source.id == Edge.source_id)
        .join(target, target.id == Edge.target_id)
        .join(file_alias, file_alias.id == source.file_id)
        .where(Edge.project_id == project_id)
        .where(or_(Edge.source_id.in_(frontier), Edge.target_id.in_(frontier)))
        .limit(80)
    )
    if edge_kinds:
        statement = statement.where(Edge.kind.in_(edge_kinds))
    return statement


def _symbol_evidence(
    symbol: Symbol,
    file_row: File,
    evidence_type: str,
) -> RagEvidence:
    return RagEvidence(
        evidence_type=evidence_type,
        source="relational",
        file_path=file_row.path,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        symbol=symbol.qualified_name,
        score=1.0,
        content_excerpt=symbol.signature or symbol.name,
        metadata={
            "symbol_id": symbol.id,
            "kind": symbol.kind,
            "name": symbol.name,
            "module_name": file_row.module_name,
            "service_name": file_row.service_name,
            "metadata": _json_safe(symbol.meta),
        },
    )


def _edge_evidence(
    edge: Edge,
    source: Symbol,
    target: Symbol,
    file_row: File,
) -> RagEvidence:
    return RagEvidence(
        evidence_type=f"edge:{edge.kind}",
        source="relational",
        file_path=file_row.path,
        start_line=edge.line,
        end_line=edge.line,
        symbol=f"{source.qualified_name} -> {target.qualified_name}",
        score=float(edge.confidence),
        content_excerpt=f"{source.qualified_name} {edge.kind} {target.qualified_name}",
        metadata={"edge_id": edge.id, "resolved_by": edge.resolved_by},
    )


def _evidence_symbol_ids(evidence: list[RagEvidence]) -> list[str]:
    result: list[str] = []
    for item in evidence:
        symbol_id = item.metadata.get("symbol_id")
        if isinstance(symbol_id, str):
            result.append(symbol_id)
    return result


def _expand_query_with_library(query: str, tool_input: ToolInput) -> str:
    hints = load_query_hints(tool_input.project_path)
    terms = [query]
    for keyword, values in hints.items():
        if keyword in query:
            terms.extend(values)
    lowered = query.lower()
    for keyword, values in _GENERIC_QUERY_EXPANSIONS.items():
        if keyword in query or keyword in lowered:
            terms.extend(values)
    return " ".join(_dedupe_terms(terms))


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in _TOKEN_RE.findall(query):
        cleaned = match.strip(".,;:()[]{}<>\"'")
        if cleaned:
            terms.append(cleaned)
    for raw in query.split():
        cleaned = raw.strip(".,;:()[]{}<>\"'")
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return _dedupe_terms(terms)


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        normalized = term.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _dedupe(items: list[RagEvidence]) -> list[RagEvidence]:
    seen: set[tuple[str, str | None, int | None, str | None]] = set()
    result: list[RagEvidence] = []
    for item in items:
        key = (item.evidence_type, item.file_path, item.start_line, item.symbol)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _infer_module(evidence: list[RagEvidence]) -> str | None:
    for item in evidence:
        module = item.metadata.get("module_name")
        if isinstance(module, str) and module:
            return module
        if item.file_path and "/" in item.file_path:
            return item.file_path.split("/", 1)[0]
    return None


def _resolve_module(
    session: Any,
    project_id: str,
    module: str | None,
    query: str,
    project_path: Any,
) -> str | None:
    resolution_text = _expand_resolution_text(
        " ".join(value for value in (module, query) if value),
        project_path,
    )
    terms = [
        term
        for term in _query_terms(resolution_text)
        if term.lower() not in _MODULE_GENERIC_TERMS
    ]
    candidates = _module_candidates(session, project_id)
    scored = [
        (_module_score(candidate, terms), candidate)
        for candidate in candidates
    ]
    scored = [(score, candidate) for score, candidate in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _module_candidates(session: Any, project_id: str) -> list[str]:
    rows = session.execute(
        select(File.module_name, File.service_name, File.path)
        .where(File.project_id == project_id)
        .limit(20000)
    ).all()
    candidates: list[str] = []
    for module_name, service_name, file_path in rows:
        for value in (service_name, module_name, _path_root(file_path)):
            if isinstance(value, str) and value and value not in candidates:
                candidates.append(value)
    return candidates


def _path_root(file_path: object) -> str | None:
    if not isinstance(file_path, str) or not file_path:
        return None
    normalized = file_path.replace("\\", "/")
    return normalized.split("/", 1)[0]


def _module_score(candidate: str, terms: list[str]) -> int:
    lowered_candidate = candidate.lower()
    score = 0
    for term in _prioritized_terms(terms)[:16]:
        lowered_term = term.lower()
        if len(lowered_term) < 3 and lowered_term.isascii():
            continue
        if lowered_candidate == lowered_term:
            score += 100
        elif lowered_candidate.startswith(lowered_term):
            score += 60
        elif f"-{lowered_term}-" in f"-{lowered_candidate}-":
            score += 50
        elif lowered_term in lowered_candidate:
            score += 30
    if score and ("-service" in lowered_candidate or "_service" in lowered_candidate):
        score += 5
    return score


def _expand_resolution_text(text: str, project_path: Any) -> str:
    if not text:
        return text
    terms = [text]
    for keyword, values in load_query_hints(project_path).items():
        if keyword in text:
            terms.extend(values)
    return " ".join(_dedupe_terms(terms))


def _prioritized_terms(terms: list[str]) -> list[str]:
    def score(term: str) -> tuple[int, int]:
        lowered = term.lower()
        if "-service" in lowered or "_service" in lowered:
            return (0, -len(term))
        if any(char in lowered for char in ("-", "_", "/", ".")):
            return (1, -len(term))
        if lowered.isascii() and len(lowered) >= 3:
            return (2, -len(term))
        return (3, -len(term))

    return sorted(_dedupe_terms(terms), key=score)


def _looks_like_table_question(query: str) -> bool:
    intent = infer_query_intent(
        query=query,
        explicit_intent=None,
        group_by=None,
        intent_configs=load_intent_configs(),
    )
    return intent in TABLE_RETRIEVAL_INTENTS


def _looks_like_http_api_query(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in query
        for term in ("\u63a5\u53e3", "\u8def\u7531")
    ) or any(
        term in lowered
        for term in (
            "http",
            "api",
            "endpoint",
            "route",
            "controller",
            "mapping",
        )
    )


def _http_artifact_rank(symbol: Symbol, file_row: File) -> tuple[int, str, int, str]:
    path = file_row.path.lower()
    qualified = symbol.qualified_name.lower()
    if symbol.kind == "route":
        primary = 0
    elif "/controller/" in path or qualified.endswith("controller"):
        primary = 1
    elif "/web/" in path:
        primary = 2
    elif "/feign/" in path or "feign" in qualified:
        primary = 3
    elif "application" in path or "/config/" in path:
        primary = 8
    else:
        primary = 5
    return (primary, path, symbol.start_line, symbol.qualified_name)


def _limit(top_k: int, multiplier: int) -> int:
    return max(1, min(top_k * multiplier, 200))


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _metadata_string(metadata: object, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return cast(Any, value)
