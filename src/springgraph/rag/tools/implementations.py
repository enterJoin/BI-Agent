"""Concrete Agentic RAG tool implementations."""

import re
from dataclasses import dataclass

from springgraph.rag import retriever, source_reader
from springgraph.rag.config.models import ToolConfig
from springgraph.rag.library_hints import load_query_hints
from springgraph.rag.memory.store import get_thread
from springgraph.rag.schemas import RagEvidence, RagPlan
from springgraph.rag.tools.schemas import RagTool, ToolInput, ToolResult

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$.:/-]*")


@dataclass(frozen=True)
class VectorSearchTool:
    """Semantic retrieval tool."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        query = _expand_query_with_library(tool_input.query, tool_input)
        plan = _plan(
            query=query,
            top_k=tool_input.top_k,
            graph_depth=tool_input.graph_depth,
            use_vector_search=True,
            use_relational_search=False,
        )
        evidence, warnings = retriever.retrieve_vector_evidence(
            tool_input.project_path,
            tool_input.project_id,
            plan,
        )
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Vector search query={query!r} returned "
                f"{len(evidence)} evidence items."
            ),
            evidence=evidence,
            warnings=warnings,
        )


@dataclass(frozen=True)
class RelationalSearchTool:
    """Structured graph retrieval tool."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        query = _expand_query_with_library(tool_input.query, tool_input)
        plan = _plan(
            query=query,
            top_k=tool_input.top_k,
            graph_depth=tool_input.graph_depth,
            use_vector_search=False,
            use_relational_search=True,
        )
        evidence, warnings = retriever.retrieve_relational_evidence(
            tool_input.project_id,
            plan,
        )
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Relational search query={query!r} returned "
                f"{len(evidence)} evidence items."
            ),
            evidence=evidence,
            warnings=warnings,
        )


@dataclass(frozen=True)
class CallGraphSearchTool:
    """Call graph retrieval tool."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        query = _expand_query_with_library(tool_input.query, tool_input)
        plan = _plan(
            query=query,
            top_k=tool_input.top_k,
            graph_depth=tool_input.graph_depth,
            use_vector_search=False,
            use_relational_search=True,
            relation_expansion=True,
        )
        evidence, warnings = retriever.retrieve_relational_evidence(
            tool_input.project_id,
            plan,
        )
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Call graph search query={query!r} returned "
                f"{len(evidence)} evidence items."
            ),
            evidence=evidence,
            warnings=warnings,
        )


@dataclass(frozen=True)
class DbMappingSearchTool:
    """Persistence and database mapping retrieval tool."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        query = _expand_query_with_library(tool_input.query, tool_input)
        query = (
            f"{query} mapper repository entity table sql insert update save"
        )
        plan = _plan(
            query=query,
            top_k=tool_input.top_k,
            graph_depth=tool_input.graph_depth,
            use_vector_search=True,
            use_relational_search=True,
        )
        relational, relational_warnings = retriever.retrieve_relational_evidence(
            tool_input.project_id,
            plan,
        )
        vector, vector_warnings = retriever.retrieve_vector_evidence(
            tool_input.project_path,
            tool_input.project_id,
            plan,
        )
        evidence = _dedupe([*relational, *vector])
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"DB mapping search query={query!r} returned "
                f"{len(evidence)} evidence items."
            ),
            evidence=evidence,
            warnings=[*relational_warnings, *vector_warnings],
        )


@dataclass(frozen=True)
class ConfigSearchTool:
    """Configuration and enum retrieval tool."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        query = _expand_query_with_library(tool_input.query, tool_input)
        query = f"{query} config properties yaml yml enum constant"
        plan = _plan(
            query=query,
            top_k=tool_input.top_k,
            graph_depth=0,
            use_vector_search=True,
            use_relational_search=True,
        )
        relational, relational_warnings = retriever.retrieve_relational_evidence(
            tool_input.project_id,
            plan,
        )
        vector, vector_warnings = retriever.retrieve_vector_evidence(
            tool_input.project_path,
            tool_input.project_id,
            plan,
        )
        evidence = _dedupe([*relational, *vector])
        return ToolResult(
            tool_name=self.config.name,
            summary=(
                f"Config search query={query!r} returned "
                f"{len(evidence)} evidence items."
            ),
            evidence=evidence,
            warnings=[*relational_warnings, *vector_warnings],
        )


@dataclass(frozen=True)
class SourceReaderTool:
    """Source reader tool with a hard path gate."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        if not tool_input.source_available:
            return ToolResult(
                tool_name=self.config.name,
                summary="Source reading skipped because source is unavailable.",
                warnings=["source_reader skipped: source_available=false"],
            )
        snippets, warnings = source_reader.read_source_snippets(
            tool_input.project_path,
            tool_input.evidence,
            max_files=tool_input.max_source_files,
            max_lines=tool_input.max_source_lines,
            line_padding=tool_input.source_line_padding,
        )
        return ToolResult(
            tool_name=self.config.name,
            summary=f"Source reader returned {len(snippets)} snippets.",
            source_snippets=snippets,
            warnings=warnings,
        )


@dataclass(frozen=True)
class MemorySearchTool:
    """Short-term thread memory lookup tool."""

    config: ToolConfig

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        if tool_input.thread_id is None:
            return ToolResult(
                tool_name=self.config.name,
                summary="No thread_id was available for memory lookup.",
            )
        memory = get_thread(tool_input.thread_id)
        excerpts = [
            message["content"]
            for message in memory.messages[-4:]
            if message.get("content")
        ]
        if memory.last_question is not None and memory.last_question not in excerpts:
            excerpts.append(memory.last_question)
        evidence = [
            RagEvidence(
                evidence_type="memory",
                source="memory",
                score=1.0,
                content_excerpt=excerpt[:500],
            )
            for excerpt in excerpts
        ]
        return ToolResult(
            tool_name=self.config.name,
            summary=f"Memory search returned {len(evidence)} memory items.",
            evidence=evidence,
        )


def create_tool(config: ToolConfig) -> RagTool:
    """Create one tool from registration metadata."""
    if config.name == "vector_search":
        return VectorSearchTool(config)
    if config.name == "relational_search":
        return RelationalSearchTool(config)
    if config.name == "call_graph_search":
        return CallGraphSearchTool(config)
    if config.name == "db_mapping_search":
        return DbMappingSearchTool(config)
    if config.name == "config_search":
        return ConfigSearchTool(config)
    if config.name == "source_reader":
        return SourceReaderTool(config)
    if config.name == "memory_search":
        return MemorySearchTool(config)
    raise ValueError(f"Unsupported RAG tool implementation: {config.name}")


def _plan(
    query: str,
    top_k: int,
    graph_depth: int,
    use_vector_search: bool,
    use_relational_search: bool,
    relation_expansion: bool = True,
) -> RagPlan:
    queries = _query_terms(query)
    return RagPlan(
        intent="agentic_tool_query",
        original_question=query,
        rewritten_query=query,
        expanded_queries=[*queries, query],
        use_vector_search=use_vector_search,
        use_relational_search=use_relational_search,
        relation_expansion=relation_expansion,
        need_source_reading=True,
        top_k=top_k,
        graph_depth=graph_depth,
    )


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


def _expand_query_with_library(query: str, tool_input: ToolInput) -> str:
    hints = load_query_hints(tool_input.project_path)
    terms = [query]
    for keyword, values in hints.items():
        if keyword in query:
            terms.extend(values)
    return " ".join(_dedupe_terms(terms))


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
