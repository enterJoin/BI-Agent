"""Typed configuration models for Agentic RAG."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval limits used by tools."""

    default_top_k: int
    max_top_k: int
    default_graph_depth: int
    max_graph_depth: int


@dataclass(frozen=True)
class SourceReadingConfig:
    """Source reading behavior."""

    enabled_by_default: bool
    max_files: int
    max_lines_per_file: int
    line_padding: int


@dataclass(frozen=True)
class ContextConfig:
    """Context assembly limits."""

    max_evidence_items: int
    max_source_snippets: int


@dataclass(frozen=True)
class MemoryConfig:
    """Memory options."""

    short_term_enabled: bool
    long_term_enabled: bool
    max_history_messages: int = 6
    max_history_chars: int = 4000


@dataclass(frozen=True)
class SafetyConfig:
    """Hard safety options."""

    require_evidence_citation: bool
    allow_shell_execution: bool
    allow_code_modification: bool


@dataclass(frozen=True)
class AgenticRagConfig:
    """Runtime configuration for Agentic RAG."""

    retrieval: RetrievalConfig
    source_reading: SourceReadingConfig
    context: ContextConfig
    memory: MemoryConfig
    safety: SafetyConfig


@dataclass(frozen=True)
class ToolConfig:
    """External tool registration metadata."""

    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(frozen=True)
class AggregationSpecConfig:
    """aggregate_query grouping dimension loaded from configuration."""

    group_by: str
    aliases: tuple[str, ...]
    symbol_kinds: tuple[str, ...]
    evidence_type: str
    label: str
    metadata_label_paths: tuple[str, ...] = ()
    annotation_names: tuple[str, ...] = ()
    path_hints: tuple[str, ...] = ()
    candidate_limit: int = 200


@dataclass(frozen=True)
class IntentConfig:
    """Query intent routing metadata."""

    name: str
    description: str
    default_tool: str
    default_filters: dict[str, object] = field(default_factory=dict)
    chinese_terms: list[str] = field(default_factory=list)
    english_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TargetTraceConfig:
    """Target tracing heuristics and relation defaults."""

    min_target_length: int = 3
    default_source_priority: int = 40
    generic_terms: list[str] = field(default_factory=list)
    class_suffixes: list[str] = field(default_factory=list)
    symbolic_chars: list[str] = field(default_factory=list)
    persistence_edge_kinds: list[str] = field(default_factory=list)
    table_target_kinds: list[str] = field(default_factory=list)
    source_priorities: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTraceConfig:
    """Execution trace trigger and target heuristics."""

    trigger_terms: list[str] = field(default_factory=list)
    entrypoint_suffixes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScopeFallbackConfig:
    """Fallback evidence options when a resolved module has no local hits."""

    enabled: bool = True
    related_service_symbol_kinds: list[str] = field(default_factory=list)
    related_service_limit: int = 3
    related_table_limit_per_service: int = 20


@dataclass(frozen=True)
class TaskPlanningConfig:
    """Task planning intent and response guidance."""

    intent_name: str = "task_planning"
    planning_terms: list[str] = field(default_factory=list)
    action_terms: list[str] = field(default_factory=list)
    scope_terms: list[str] = field(default_factory=list)
    default_steps: list[dict[str, object]] = field(default_factory=list)
    output_sections: list[str] = field(default_factory=list)
    schema_decision_rules: dict[str, list[str]] = field(default_factory=dict)
    source_evidence_priorities: dict[str, int] = field(default_factory=dict)
    default_source_evidence_priority: int = 50
