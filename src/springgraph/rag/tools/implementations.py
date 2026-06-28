"""Composable Agentic RAG tool implementations."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Text, or_, select
from sqlalchemy.sql.elements import ColumnElement

from springgraph.db import session_scope
from springgraph.models import Edge, File, Symbol
from springgraph.rag.config.loader import (
    load_aggregation_specs,
    load_execution_trace_config,
    load_intent_configs,
    load_scope_fallback_config,
    load_target_trace_config,
)
from springgraph.rag.config.models import AggregationSpecConfig, ToolConfig
from springgraph.rag.intent import TABLE_RETRIEVAL_INTENTS, infer_query_intent
from springgraph.rag.library_hints import load_query_hints
from springgraph.rag.schemas import RagEvidence, SourceSnippet
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

_CONTROL_FLOW_RE = re.compile(r"\b(if|else\s+if|else|for|while|switch|catch)\b")
_RETURN_CONTINUE_RE = re.compile(r"\b(return|continue|break)\b")
_JAVA_CALL_RE = re.compile(
    r"\b(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_JAVA_METHOD_REF_RE = re.compile(
    r"\b(?:this|[A-Za-z_][A-Za-z0-9_]*)::"
    r"(?P<method>[A-Za-z_][A-Za-z0-9_]*)"
)
_JAVA_LOCAL_CALL_RE = re.compile(r"\b(?P<method>[a-z][A-Za-z0-9_]*)\s*\(")
_SQL_TABLE_RE = re.compile(
    r"\b(?:from|join|into|update)\s+`?(?P<table>[A-Za-z_][\w.]*)`?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _MethodSource:
    symbol: Symbol
    file_row: File
    start_line: int
    end_line: int
    content: str


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
            fallback_evidence = []
            if not rows and resolved_module:
                fallback_evidence = _module_scope_fallback(
                    session=session,
                    project_id=tool_input.project_id,
                    module=resolved_module,
                    query=tool_input.query,
                    scope="artifacts",
                    include_related_tables=False,
                    table_limit=0,
                )

        evidence = [
            _symbol_evidence(symbol, file_row, "artifact")
            for symbol, file_row in rows
        ]
        evidence.extend(fallback_evidence)
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
        spec = _aggregation_spec_for(group_by)
        if spec is not None:
            if spec.group_by == "table":
                return _aggregate_tables(self.config, tool_input)
            return _aggregate_by_spec(self.config, tool_input, spec)
        if group_by:
            artifact_spec = _aggregation_spec_for("artifact")
            if artifact_spec is not None:
                result = _aggregate_by_spec(self.config, tool_input, artifact_spec)
                return ToolResult(
                    tool_name=result.tool_name,
                    summary=(
                        f"Unsupported aggregate group_by={group_by!r}; "
                        f"fell back to group_by='artifact'. {result.summary}"
                    ),
                    evidence=result.evidence,
                    source_snippets=result.source_snippets,
                    warnings=[
                        *result.warnings,
                        f"Unsupported aggregate group_by: {group_by}",
                    ],
                )
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
class ExecutionTraceTool:
    """Trace detailed method execution from source and graph evidence."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        if not tool_input.source_available:
            return ToolResult(
                tool_name=self.config.name,
                summary="Execution trace skipped because source is unavailable.",
                warnings=["execution_trace skipped: source_available=false"],
            )
        allowed, reason = check_source_path(tool_input.project_path)
        if not allowed:
            return ToolResult(
                tool_name=self.config.name,
                summary=f"Execution trace skipped: {reason}.",
                warnings=[f"execution_trace skipped: {reason}"],
            )

        target_terms = _execution_target_candidates(
            tool_input.query,
            tool_input.filters,
        )
        limit = _limit(tool_input.top_k, multiplier=4)
        with session_scope() as session:
            target_rows = _resolve_execution_targets(
                session=session,
                project_id=tool_input.project_id,
                target_terms=target_terms,
                limit=limit,
            )
            method_sources = _read_method_sources(
                root=tool_input.project_path,
                rows=target_rows,
                max_methods=max(1, min(tool_input.top_k, 6)),
                max_lines=max(tool_input.max_source_lines, 180),
            )
            downstream_rows = _resolve_downstream_methods(
                session=session,
                project_id=tool_input.project_id,
                method_sources=method_sources,
                limit=limit,
            )
            downstream_sources = _read_method_sources(
                root=tool_input.project_path,
                rows=downstream_rows,
                max_methods=max(1, min(tool_input.top_k, 8)),
                max_lines=max(tool_input.max_source_lines, 220),
            )
            second_downstream_rows = _resolve_downstream_methods(
                session=session,
                project_id=tool_input.project_id,
                method_sources=downstream_sources,
                limit=limit,
            )
            second_downstream_sources = _read_method_sources(
                root=tool_input.project_path,
                rows=second_downstream_rows,
                max_methods=max(1, min(tool_input.top_k, 8)),
                max_lines=max(tool_input.max_source_lines, 180),
            )
            sql_rows = _resolve_sql_statement_rows(
                session=session,
                project_id=tool_input.project_id,
                method_refs=_called_method_refs(
                    [
                        *method_sources,
                        *downstream_sources,
                        *second_downstream_sources,
                    ]
                ),
                limit=limit,
            )
            sql_sources = _read_method_sources(
                root=tool_input.project_path,
                rows=sql_rows,
                max_methods=max(1, min(tool_input.top_k, 10)),
                max_lines=max(tool_input.max_source_lines, 160),
            )
            table_rows = _trace_method_table_relations(
                session=session,
                project_id=tool_input.project_id,
                symbol_ids=[
                    item.symbol.id
                    for item in [
                        *method_sources,
                        *downstream_sources,
                        *second_downstream_sources,
                        *sql_sources,
                    ]
                ],
                limit=limit,
            )

        all_sources = _dedupe_method_sources(
            [
                *method_sources,
                *downstream_sources,
                *second_downstream_sources,
                *sql_sources,
            ]
        )
        evidence = _execution_trace_evidence(all_sources)
        evidence.extend(_execution_table_evidence(table_rows))
        source_snippets = [
            SourceSnippet(
                file_path=item.file_row.path,
                start_line=item.start_line,
                end_line=item.end_line,
                content=item.content,
            )
            for item in all_sources[: max(2, tool_input.max_source_files * 2)]
        ]
        warnings: list[str] = []
        if not all_sources:
            warnings.append("execution_trace found no readable method body")
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Execution trace targets={target_terms}, methods_read="
                f"{len(all_sources)}, table_relations={len(table_rows)}."
            ),
            evidence=_dedupe(evidence),
            source_snippets=source_snippets,
            warnings=warnings,
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
    if config.name == "execution_trace":
        return ExecutionTraceTool(config)
    if config.name == "source_read":
        return SourceReadTool(config)
    raise ValueError(f"Unsupported RAG tool implementation: {config.name}")


def _resolve_execution_targets(
    session: Any,
    project_id: str,
    target_terms: list[str],
    limit: int,
) -> list[tuple[Symbol, File]]:
    if not target_terms:
        return []
    annotation_target_rows = _resolve_annotation_execution_targets(
        session=session,
        project_id=project_id,
        target_terms=target_terms,
        limit=limit,
    )
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
    method_rows = session.execute(
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
        .where(Symbol.kind.in_(["method", "constructor", "route"]))
        .where(or_(*filters))
        .limit(limit)
    ).all()
    return _merge_ranked_execution_targets(
        annotation_target_rows,
        _rank_execution_target_rows(method_rows, target_terms),
    )[:limit]


def _resolve_annotation_execution_targets(
    session: Any,
    project_id: str,
    target_terms: list[str],
    limit: int,
) -> list[tuple[Symbol, File]]:
    filters = []
    for term in target_terms[:8]:
        pattern = f"%{term}%"
        filters.extend(
            [
                Symbol.name.ilike(pattern),
                Symbol.qualified_name.ilike(pattern),
                Symbol.meta.cast(Text).ilike(pattern),
            ]
        )
    annotation_rows = session.execute(
        select(Symbol)
        .where(Symbol.project_id == project_id)
        .where(Symbol.kind == "annotation_usage")
        .where(or_(*filters))
        .limit(limit)
    ).scalars()
    target_ids = [
        target_id
        for symbol in annotation_rows
        if isinstance(symbol.meta, dict)
        if isinstance(target_id := symbol.meta.get("target_symbol_id"), str)
    ]
    if not target_ids:
        return []
    rows = session.execute(
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
        .where(Symbol.id.in_(target_ids))
        .limit(limit)
    ).all()
    order = {target_id: index for index, target_id in enumerate(target_ids)}
    return sorted(rows, key=lambda row: order.get(row[0].id, len(order)))


def _merge_ranked_execution_targets(
    primary_rows: list[tuple[Symbol, File]],
    fallback_rows: list[tuple[Symbol, File]],
) -> list[tuple[Symbol, File]]:
    rows: list[tuple[Symbol, File]] = []
    seen: set[str] = set()
    for symbol, file_row in [*primary_rows, *fallback_rows]:
        if symbol.id in seen:
            continue
        seen.add(symbol.id)
        rows.append((symbol, file_row))
    return rows


def _execution_target_candidates(
    query: str,
    filters: dict[str, Any],
) -> list[str]:
    values = _target_candidates(query, filters)
    explicit_target = _optional_string(filters.get("target"))
    if explicit_target:
        values.append(explicit_target)
    values.extend(_string_list(filters.get("targets")))
    generic = {term.lower() for term in load_target_trace_config().generic_terms}
    for term in _query_terms(query):
        lowered = term.lower()
        if lowered in generic:
            continue
        if _is_execution_suffix_noise(term):
            continue
        if len(term) < 3:
            continue
        if _looks_like_java_identifier(term):
            values.append(term)
    return _dedupe_terms(values)


def _is_execution_suffix_noise(term: str) -> bool:
    lowered = term.lower()
    return any(
        lowered == suffix.lower()
        for suffix in load_execution_trace_config().entrypoint_suffixes
    )


def _looks_like_java_identifier(term: str) -> bool:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", term):
        return False
    return (
        any(char.isupper() for char in term[1:])
        or term.endswith(("Handler", "Job", "Listener", "Task", "Service"))
    )


def _rank_execution_target_rows(
    rows: list[tuple[Symbol, File]],
    target_terms: list[str],
) -> list[tuple[Symbol, File]]:
    lowered_terms = [term.lower() for term in target_terms]

    def rank(row: tuple[Symbol, File]) -> tuple[int, str, int, str]:
        symbol, file_row = row
        exact = symbol.name.lower() in lowered_terms
        qualified = symbol.qualified_name.lower() in lowered_terms
        path = file_row.path.lower()
        if qualified:
            priority = 0
        elif exact and "/service/" in path.replace("\\", "/"):
            priority = 1
        elif exact:
            priority = 2
        else:
            priority = 4
        return (priority, file_row.path, symbol.start_line, symbol.qualified_name)

    return sorted(rows, key=rank)


def _read_method_sources(
    root: Path,
    rows: list[tuple[Symbol, File]],
    max_methods: int,
    max_lines: int,
) -> list[_MethodSource]:
    sources: list[_MethodSource] = []
    seen: set[tuple[str, int]] = set()
    project_root = root.resolve()
    for symbol, file_row in rows:
        if len(sources) >= max_methods:
            break
        if symbol.start_line is None:
            continue
        marker = (file_row.path, symbol.start_line)
        if marker in seen:
            continue
        seen.add(marker)
        source_path = (project_root / file_row.path).resolve()
        if not _is_relative_to(source_path, project_root):
            continue
        if not source_path.exists() or not source_path.is_file():
            continue
        lines = _read_text(source_path).splitlines()
        start_line, end_line = _method_source_window(
            lines=lines,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            max_lines=max_lines,
        )
        if start_line > end_line:
            continue
        sources.append(
            _MethodSource(
                symbol=symbol,
                file_row=file_row,
                start_line=start_line,
                end_line=end_line,
                content="\n".join(lines[start_line - 1 : end_line]),
            )
        )
    return sources


def _method_source_window(
    lines: list[str],
    start_line: int,
    end_line: int | None,
    max_lines: int,
) -> tuple[int, int]:
    total_lines = len(lines)
    if total_lines == 0:
        return (1, 0)
    start = max(1, start_line)
    inferred_end = end_line if end_line and end_line > start_line else None
    if inferred_end is None:
        inferred_end = _infer_block_end_line(lines, start)
    end = min(total_lines, max(start, inferred_end))
    if end - start + 1 > max_lines:
        end = start + max_lines - 1
    return (start, min(total_lines, end))


def _infer_block_end_line(lines: list[str], start_line: int) -> int:
    brace_depth = 0
    seen_open = False
    for index in range(start_line - 1, len(lines)):
        line = _strip_line_comment(lines[index])
        brace_depth += line.count("{")
        if "{" in line:
            seen_open = True
        brace_depth -= line.count("}")
        if seen_open and brace_depth <= 0:
            return index + 1
    return min(len(lines), start_line + 80)


def _resolve_downstream_methods(
    session: Any,
    project_id: str,
    method_sources: list[_MethodSource],
    limit: int,
) -> list[tuple[Symbol, File]]:
    method_names = _called_method_names(method_sources)
    if not method_names:
        return []
    method_refs = _called_method_refs(method_sources)
    rows = session.execute(
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
        .where(Symbol.kind.in_(["method", "constructor"]))
        .where(Symbol.name.in_(method_names[:80]))
        .limit(limit)
    ).all()
    return _rank_downstream_rows(rows, method_sources, method_names, method_refs)


def _resolve_sql_statement_rows(
    session: Any,
    project_id: str,
    method_refs: list[tuple[str, str]],
    limit: int,
) -> list[tuple[Symbol, File]]:
    if not method_refs:
        return []
    exact_filters = []
    fallback_names: list[str] = []
    for receiver, method in method_refs[:120]:
        fallback_names.append(method)
        mapper_name = _receiver_mapper_name(receiver)
        if mapper_name is None:
            continue
        exact_filters.append(Symbol.qualified_name.ilike(f"%{mapper_name}#{method}"))
    filters = list(exact_filters)
    if fallback_names and not exact_filters:
        filters.append(Symbol.name.in_(fallback_names))
    rows = session.execute(
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
        .where(Symbol.kind == "sql_statement")
        .where(or_(*filters))
        .limit(limit)
    ).all()
    exact_names = {
        f"{_receiver_mapper_name(receiver)}#{method}": index
        for index, (receiver, method) in enumerate(method_refs)
        if _receiver_mapper_name(receiver) is not None
    }
    name_order = {
        method: index
        for index, (_, method) in enumerate(method_refs)
        if method not in exact_names
    }
    return sorted(
        rows,
        key=lambda row: (
            _sql_statement_priority(row[0], exact_names),
            name_order.get(row[0].name, len(name_order)),
            row[1].path,
            row[0].start_line,
        ),
    )


def _called_method_refs(method_sources: list[_MethodSource]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for item in method_sources:
        for match in _JAVA_CALL_RE.finditer(item.content):
            method = match.group("method")
            if _is_generic_call(method):
                continue
            refs.append((match.group("receiver"), method))
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        result.append(ref)
    return result


def _receiver_mapper_name(receiver: str) -> str | None:
    if not receiver.lower().endswith("mapper"):
        return None
    return receiver[:1].upper() + receiver[1:]


def _receiver_class_name(receiver: str) -> str | None:
    if not receiver:
        return None
    return receiver[:1].upper() + receiver[1:]


def _sql_statement_priority(
    symbol: Symbol,
    exact_names: dict[str, int],
) -> int:
    for marker in exact_names:
        if symbol.qualified_name.endswith(marker):
            return 0
    return 5


def _called_method_names(method_sources: list[_MethodSource]) -> list[str]:
    ignored = {
        "catch",
        "for",
        "if",
        "new",
        "return",
        "switch",
        "throw",
        "while",
    }
    names: list[str] = []
    for item in method_sources:
        content = item.content
        names.extend(
            match.group("method")
            for match in _JAVA_CALL_RE.finditer(content)
            if not _is_generic_call(match.group("method"))
        )
        names.extend(
            match.group("method") for match in _JAVA_METHOD_REF_RE.finditer(content)
        )
        for match in _JAVA_LOCAL_CALL_RE.finditer(content):
            method = match.group("method")
            if method not in ignored and not _is_generic_call(method):
                names.append(method)
    return _dedupe_terms(names)


def _is_generic_call(method: str) -> bool:
    return method in {
        "add",
        "collect",
        "contains",
        "equals",
        "filter",
        "forEach",
        "format",
        "get",
        "getOrDefault",
        "getValue",
        "identity",
        "info",
        "isEmpty",
        "isNotEmpty",
        "map",
        "newArrayList",
        "of",
        "parseLong",
        "put",
        "set",
        "stream",
        "toJsonString",
        "toList",
        "toMap",
    }


def _rank_downstream_rows(
    rows: list[tuple[Symbol, File]],
    method_sources: list[_MethodSource],
    method_names: list[str],
    method_refs: list[tuple[str, str]],
) -> list[tuple[Symbol, File]]:
    source_paths = {item.file_row.path for item in method_sources}
    source_classes = {
        item.symbol.qualified_name.rsplit(".", maxsplit=1)[0]
        for item in method_sources
    }
    name_order = {name: index for index, name in enumerate(method_names)}
    receiver_targets = {
        (class_name, method): index
        for index, (receiver, method) in enumerate(method_refs)
        if (class_name := _receiver_class_name(receiver)) is not None
    }
    source_ids = {item.symbol.id for item in method_sources}

    def rank(row: tuple[Symbol, File]) -> tuple[int, int, str, int]:
        symbol, file_row = row
        receiver_priority = _receiver_match_priority(symbol, receiver_targets)
        if receiver_priority is not None:
            priority = receiver_priority
        elif symbol.id in source_ids:
            priority = 9
        elif file_row.path in source_paths:
            priority = 2
        elif symbol.qualified_name.rsplit(".", maxsplit=1)[0] in source_classes:
            priority = 3
        else:
            priority = 6
        return (
            priority,
            name_order.get(symbol.name, len(name_order)),
            file_row.path,
            symbol.start_line,
        )

    return sorted(rows, key=rank)


def _receiver_match_priority(
    symbol: Symbol,
    receiver_targets: dict[tuple[str, str], int],
) -> int | None:
    for (class_name, method), _ in receiver_targets.items():
        if symbol.name != method:
            continue
        if symbol.qualified_name.rsplit(".", maxsplit=1)[0].endswith(
            f".{class_name}"
        ):
            return 0
        if class_name in symbol.qualified_name:
            return 1
    return None


def _trace_method_table_relations(
    session: Any,
    project_id: str,
    symbol_ids: list[str],
    limit: int,
) -> list[tuple[Edge, Symbol, Symbol, File]]:
    if not symbol_ids:
        return []
    from sqlalchemy.orm import aliased

    source = aliased(Symbol)
    target = aliased(Symbol)
    file_alias = aliased(File)
    rows = session.execute(
        select(Edge, source, target, file_alias)
        .join(source, source.id == Edge.source_id)
        .join(target, target.id == Edge.target_id)
        .join(file_alias, file_alias.id == source.file_id)
        .where(Edge.project_id == project_id)
        .where(Edge.source_id.in_(symbol_ids))
        .where(Edge.kind.in_(["reads_table", "writes_table", "defines_contract"]))
        .limit(limit)
    ).all()
    return _rank_relation_rows(rows)


def _execution_trace_evidence(
    method_sources: list[_MethodSource],
) -> list[RagEvidence]:
    evidence: list[RagEvidence] = []
    for item in method_sources:
        details = _execution_details(item.content)
        evidence.append(
            RagEvidence(
                evidence_type="execution_method",
                source="execution_trace",
                file_path=item.file_row.path,
                start_line=item.start_line,
                end_line=item.end_line,
                symbol=item.symbol.qualified_name,
                score=1.0,
                content_excerpt=details,
                metadata={
                    "symbol_id": item.symbol.id,
                    "kind": item.symbol.kind,
                    "method": item.symbol.name,
                    "line_count": item.end_line - item.start_line + 1,
                },
            )
        )
    return evidence


def _execution_details(content: str) -> str:
    lines = content.splitlines()
    details: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if _CONTROL_FLOW_RE.search(line) or _RETURN_CONTINUE_RE.search(line):
            details.append(f"control[{line_number}]: {line}")
        elif "kafkaService.send" in line or ".send(" in line:
            details.append(f"message[{line_number}]: {line}")
        else:
            call_match = _JAVA_CALL_RE.search(line)
            if call_match:
                details.append(
                    f"call[{line_number}]: "
                    f"{call_match.group('receiver')}.{call_match.group('method')}"
                )
        for table in _tables_from_source_line(line):
            details.append(f"table_hint[{line_number}]: {table}")
        if len(details) >= 220:
            break
    return "\n".join(details) if details else content[:1200]


def _tables_from_source_line(line: str) -> list[str]:
    return [
        match.group("table").split(".", maxsplit=1)[-1]
        for match in _SQL_TABLE_RE.finditer(line)
    ]


def _execution_table_evidence(
    rows: list[tuple[Edge, Symbol, Symbol, File]],
) -> list[RagEvidence]:
    evidence: list[RagEvidence] = []
    for edge, source, target, file_row in rows:
        evidence.append(
            RagEvidence(
                evidence_type=f"execution_table:{edge.kind}",
                source="execution_trace",
                file_path=file_row.path,
                start_line=edge.line,
                end_line=edge.line,
                symbol=f"{source.qualified_name} -> {target.qualified_name}",
                score=float(edge.confidence),
                content_excerpt=(
                    f"{source.qualified_name} {edge.kind} "
                    f"{target.qualified_name}"
                ),
                metadata={
                    "edge_id": edge.id,
                    "edge_kind": edge.kind,
                    "source_symbol_id": source.id,
                    "target_symbol_id": target.id,
                    "table": _metadata_string(target.meta, "table") or target.name,
                },
            )
        )
    return evidence


def _dedupe_method_sources(items: list[_MethodSource]) -> list[_MethodSource]:
    seen: set[tuple[str, int, str]] = set()
    result: list[_MethodSource] = []
    for item in items:
        key = (item.file_row.path, item.start_line, item.symbol.qualified_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _strip_line_comment(line: str) -> str:
    return line.split("//", maxsplit=1)[0]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _aggregate_by_spec(
    config: ToolConfig,
    tool_input: ToolInput,
    spec: AggregationSpecConfig,
) -> ToolResult:
    module = _optional_string(tool_input.filters.get("module"))
    path_contains = _optional_string(tool_input.filters.get("path_contains"))
    query = _expand_query_with_library(tool_input.query, tool_input)
    inferred_module = _infer_module(tool_input.evidence)
    output_limit = _limit(tool_input.top_k, multiplier=8)
    candidate_limit = max(_limit(tool_input.top_k, multiplier=25), spec.candidate_limit)

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
            .where(Symbol.kind.in_(spec.symbol_kinds))
        )
        annotation_filter = _annotation_filter(spec)
        if annotation_filter is not None:
            statement = statement.where(annotation_filter)
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
        row_results = session.execute(
            statement.order_by(File.path, Symbol.kind, Symbol.name).limit(
                candidate_limit
            )
        ).all()
        rows = [(symbol, file_row) for symbol, file_row in row_results]
        rows = _rank_aggregation_rows(rows, query, spec)[:output_limit]
        fallback_evidence = []
        if spec.group_by == "table" and not rows and resolved_module:
            fallback_evidence = _module_scope_fallback(
                session=session,
                project_id=tool_input.project_id,
                module=resolved_module,
                query=tool_input.query,
                scope="tables",
                include_related_tables=True,
                table_limit=tool_input.top_k,
            )

    evidence: list[RagEvidence] = []
    seen_tables: set[str] = set()
    for symbol, file_row in rows:
        label_value = _aggregation_label_value(symbol, file_row, spec)
        marker = f"{spec.group_by}|{label_value}|{file_row.path}|{symbol.start_line}"
        if marker in seen_tables:
            continue
        seen_tables.add(marker)
        evidence.append(
            RagEvidence(
                evidence_type=spec.evidence_type,
                source="aggregate",
                file_path=file_row.path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol=symbol.qualified_name,
                score=1.0,
                content_excerpt=(
                    f"{spec.label}={label_value}; artifact={symbol.name}; "
                    f"kind={symbol.kind}; module={file_row.module_name}; "
                    f"service={file_row.service_name}"
                ),
                metadata={
                    "group_by": spec.group_by,
                    spec.label: label_value,
                    "symbol_id": symbol.id,
                    "artifact_kind": symbol.kind,
                    "name": symbol.name,
                    "module_name": file_row.module_name,
                    "service_name": file_row.service_name,
                    "metadata": _json_safe(symbol.meta),
                },
            )
        )
    evidence.extend(fallback_evidence)
    return ToolResult(
        tool_name=config.name,
        summary=(
            f"Aggregate {spec.group_by} query query={query!r}, "
            f"module={module!r}, "
            f"resolved_module={resolved_module!r} returned "
            f"{len(evidence)} evidence items."
        ),
        evidence=evidence,
    )


def _aggregate_tables(config: ToolConfig, tool_input: ToolInput) -> ToolResult:
    spec = _aggregation_spec_for("table")
    if spec is None:
        raise RuntimeError("Missing table aggregation spec.")
    return _aggregate_by_spec(config, tool_input, spec)


def _rank_table_rows(
    rows: list[tuple[Symbol, File]],
    query: str,
) -> list[tuple[Symbol, File]]:
    spec = _aggregation_spec_for("table")
    if spec is None:
        return rows
    return _rank_aggregation_rows(rows, query, spec)


def _rank_aggregation_rows(
    rows: list[tuple[Symbol, File]],
    query: str,
    spec: AggregationSpecConfig,
) -> list[tuple[Symbol, File]]:
    terms = [term.lower() for term in _prioritized_terms(_query_terms(query))[:40]]

    def rank(row: tuple[Symbol, File]) -> tuple[int, int, str, str]:
        symbol, file_row = row
        label_value = _aggregation_label_value(symbol, file_row, spec).lower()
        haystack = _aggregation_haystack(symbol, file_row, spec).lower()
        primary_match = any(
            _matches_primary_aggregation_name(label_value, term) for term in terms
        )
        if terms and primary_match:
            priority = 0
        elif terms and any(term in label_value for term in terms):
            priority = 1
        elif terms and any(term in haystack for term in terms):
            priority = 2
        else:
            priority = 5
        score = sum(
            _aggregation_term_score(term, haystack, label_value)
            for term in terms
        )
        return (priority, -score, file_row.path, symbol.name)

    return sorted(rows, key=rank)


def _matches_primary_aggregation_name(name: str, term: str) -> bool:
    return name == term or name.endswith(f"_{term}") or name.endswith(f"-{term}")


def _aggregation_spec_for(group_by: str | None) -> AggregationSpecConfig | None:
    if group_by is None:
        return None
    normalized = group_by.strip().lower().replace("-", "_")
    for spec in load_aggregation_specs():
        aliases = (spec.group_by, *spec.aliases)
        if normalized in {alias.strip().lower().replace("-", "_") for alias in aliases}:
            return spec
    return None


def _annotation_filter(spec: AggregationSpecConfig) -> ColumnElement[bool] | None:
    if not spec.annotation_names:
        return None
    filters = []
    for annotation_name in spec.annotation_names:
        filters.extend(
            [
                Symbol.name.ilike(f"@{annotation_name}%"),
                Symbol.qualified_name.ilike(f"%:{annotation_name}:%"),
                Symbol.meta.cast(Text).ilike(
                    f'%"annotation": "{annotation_name}"%'
                ),
            ]
        )
    if set(spec.symbol_kinds) == {"annotation_usage"}:
        return or_(*filters)
    return or_(Symbol.kind != "annotation_usage", *filters)


def _aggregation_label_value(
    symbol: Symbol,
    file_row: File,
    spec: AggregationSpecConfig,
) -> str:
    for path in spec.metadata_label_paths:
        value = _metadata_path_string(symbol.meta, path)
        if value:
            return value
    if spec.group_by == "module":
        return (
            file_row.module_name
            or file_row.service_name
            or _path_root(file_row.path)
            or symbol.name
        )
    if spec.group_by == "api":
        route_path = _metadata_path_string(symbol.meta, "path")
        http_method = _metadata_path_string(symbol.meta, "http_method")
        if route_path and http_method:
            return f"{http_method} {route_path}"
        if route_path:
            return route_path
    return _clean_symbol_label(symbol.name, spec)


def _metadata_path_string(metadata: object, path: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value: object = metadata
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, str) and value:
        return value
    return None


def _clean_symbol_label(name: str, spec: AggregationSpecConfig) -> str:
    value = name.strip()
    if spec.group_by == "job" and value.startswith("@XxlJob"):
        return value.removeprefix("@XxlJob").strip() or value
    if value.startswith("@"):
        parts = value.split(maxsplit=1)
        if len(parts) == 2:
            return parts[1].strip()
    return value


def _aggregation_haystack(
    symbol: Symbol,
    file_row: File,
    spec: AggregationSpecConfig,
) -> str:
    metadata = symbol.meta if isinstance(symbol.meta, dict) else {}
    values = [
        spec.group_by,
        spec.label,
        _aggregation_label_value(symbol, file_row, spec),
        symbol.name,
        symbol.qualified_name,
        symbol.kind,
        symbol.signature or "",
        file_row.path,
        file_row.module_name or "",
        file_row.service_name or "",
        json.dumps(_json_safe(metadata), ensure_ascii=False, sort_keys=True),
    ]
    return " ".join(value for value in values if value)


def _aggregation_term_score(term: str, haystack: str, label_value: str) -> int:
    if len(term) < 2:
        return 0
    if term == label_value:
        return 100
    if label_value.endswith(f"_{term}") or label_value.endswith(f"-{term}"):
        return 80
    if term in label_value:
        return 50 + min(len(term), 20)
    if term in haystack:
        return 20 + min(len(term), 20)
    return 0


def _module_scope_fallback(
    session: Any,
    project_id: str,
    module: str,
    query: str,
    scope: str,
    include_related_tables: bool,
    table_limit: int,
) -> list[RagEvidence]:
    config = load_scope_fallback_config()
    if not config.enabled:
        return []
    if not _module_has_files(session, project_id, module):
        return []

    evidence = [
        _module_scope_empty_evidence(
            module=module,
            query=query,
            scope=scope,
        )
    ]
    related_rows = _module_related_service_rows(
        session=session,
        project_id=project_id,
        module=module,
        kinds=config.related_service_symbol_kinds,
        limit=config.related_service_limit,
    )
    evidence.extend(_related_service_evidence(related_rows, module))
    if include_related_tables:
        related_services = _related_service_names(related_rows)
        evidence.extend(
            _related_service_table_evidence(
                session=session,
                project_id=project_id,
                services=related_services,
                limit_per_service=min(
                    table_limit,
                    config.related_table_limit_per_service,
                ),
            )
        )
    return _dedupe(evidence)


def _module_has_files(session: Any, project_id: str, module: str) -> bool:
    pattern = f"%{module}%"
    row = session.execute(
        select(File.id)
        .where(File.project_id == project_id)
        .where(
            or_(
                File.module_name.ilike(pattern),
                File.service_name.ilike(pattern),
                File.path.ilike(pattern),
            )
        )
        .limit(1)
    ).first()
    return row is not None


def _module_scope_empty_evidence(
    module: str,
    query: str,
    scope: str,
) -> RagEvidence:
    return RagEvidence(
        evidence_type="module_scope_empty",
        source="module_scope",
        symbol=f"module:{module}",
        score=1.0,
        content_excerpt=(
            f"module={module}; scope={scope}; local_evidence=0; "
            f"module_exists=true; query={query}"
        ),
        metadata={
            "module_name": module,
            "scope": scope,
            "local_evidence_count": 0,
            "module_exists": True,
        },
    )


def _module_related_service_rows(
    session: Any,
    project_id: str,
    module: str,
    kinds: list[str],
    limit: int,
) -> list[tuple[Symbol, File]]:
    if not kinds or limit <= 0:
        return []
    pattern = f"%{module}%"
    rows = session.execute(
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
        .where(Symbol.kind.in_(kinds))
        .where(
            or_(
                File.module_name.ilike(pattern),
                File.service_name.ilike(pattern),
                File.path.ilike(pattern),
            )
        )
        .order_by(File.path, Symbol.name)
        .limit(limit)
    ).all()
    return cast(list[tuple[Symbol, File]], rows)


def _related_service_evidence(
    rows: list[tuple[Symbol, File]],
    module: str,
) -> list[RagEvidence]:
    evidence: list[RagEvidence] = []
    for symbol, file_row in rows:
        service = _related_service_name(symbol)
        evidence.append(
            RagEvidence(
                evidence_type="module_related_service",
                source="module_scope",
                file_path=file_row.path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol=symbol.qualified_name,
                score=1.0,
                content_excerpt=(
                    f"module={module}; related_service={service}; "
                    f"artifact={symbol.name}; kind={symbol.kind}"
                ),
                metadata={
                    "module_name": module,
                    "related_service": service,
                    "artifact_kind": symbol.kind,
                    "service_name": file_row.service_name,
                },
            )
        )
    return evidence


def _related_service_names(rows: list[tuple[Symbol, File]]) -> list[str]:
    return _dedupe_terms(
        [
            service
            for symbol, _ in rows
            if (service := _related_service_name(symbol))
        ]
    )


def _related_service_name(symbol: Symbol) -> str:
    service = _metadata_string(symbol.meta, "service")
    if service:
        return service
    if symbol.qualified_name.startswith(f"{symbol.kind}:"):
        return symbol.qualified_name.split(":", 1)[1]
    return symbol.name


def _related_service_table_evidence(
    session: Any,
    project_id: str,
    services: list[str],
    limit_per_service: int,
) -> list[RagEvidence]:
    table_kinds = list(table_target_kinds())
    if not table_kinds or limit_per_service <= 0:
        return []

    evidence: list[RagEvidence] = []
    seen: set[str] = set()
    for service in services:
        rows = _service_table_rows(
            session=session,
            project_id=project_id,
            service=service,
            table_kinds=table_kinds,
            limit=limit_per_service,
        )
        for symbol, file_row in rows:
            table_name = _metadata_string(symbol.meta, "table") or symbol.name
            marker = f"{service}|{table_name}|{file_row.path}"
            if marker in seen:
                continue
            seen.add(marker)
            evidence.append(
                RagEvidence(
                    evidence_type="related_service_table",
                    source="module_scope",
                    file_path=file_row.path,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    symbol=symbol.qualified_name,
                    score=0.8,
                    content_excerpt=(
                        f"related_service={service}; table={table_name}; "
                        f"artifact={symbol.name}; kind={symbol.kind}"
                    ),
                    metadata={
                        "related_service": service,
                        "table": table_name,
                        "artifact_kind": symbol.kind,
                        "module_name": file_row.module_name,
                        "service_name": file_row.service_name,
                    },
                )
            )
    return evidence


def _service_table_rows(
    session: Any,
    project_id: str,
    service: str,
    table_kinds: list[str],
    limit: int,
) -> list[tuple[Symbol, File]]:
    pattern = f"%{service}%"
    rows = session.execute(
        select(Symbol, File)
        .join(File, File.id == Symbol.file_id)
        .where(Symbol.project_id == project_id)
        .where(Symbol.kind.in_(table_kinds))
        .where(
            or_(
                File.module_name.ilike(pattern),
                File.service_name.ilike(pattern),
                File.path.ilike(pattern),
                Symbol.qualified_name.ilike(pattern),
            )
        )
        .order_by(File.path, Symbol.kind, Symbol.name)
        .limit(limit)
    ).all()
    return cast(list[tuple[Symbol, File]], rows)


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
            terms.extend(_ascii_subterms(cleaned))
    terms.extend(_cjk_ngrams(query))
    for raw in query.split():
        cleaned = raw.strip(".,;:()[]{}<>\"'")
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return _dedupe_terms(terms)


def _ascii_subterms(token: str) -> list[str]:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", token)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    return [
        item
        for item in spaced.split()
        if len(item) >= 3 and not item.isdigit()
    ]


def _cjk_ngrams(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for char in text:
        if _is_cjk(char):
            current.append(char)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))

    terms: list[str] = []
    for run in runs:
        if len(run) <= 1:
            continue
        for size in range(2, min(4, len(run)) + 1):
            terms.extend(
                run[index : index + size]
                for index in range(0, len(run) - size + 1)
            )
    return terms


def _is_cjk(char: str) -> bool:
    return (
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )


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
