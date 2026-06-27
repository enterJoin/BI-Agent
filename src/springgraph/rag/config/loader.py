"""Load Agentic RAG configuration from YAML files."""

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml

from springgraph.rag.config.models import (
    AgenticRagConfig,
    ContextConfig,
    IntentConfig,
    MemoryConfig,
    RetrievalConfig,
    SafetyConfig,
    ScopeFallbackConfig,
    SourceReadingConfig,
    TargetTraceConfig,
    TaskPlanningConfig,
    ToolConfig,
)


@lru_cache(maxsize=1)
def load_agentic_rag_config() -> AgenticRagConfig:
    """Load controlled agent settings."""
    data = _load_yaml(_config_path("agentic_rag.yml"))
    agent = _mapping(data.get("agent"))
    return AgenticRagConfig(
        retrieval=RetrievalConfig(**_mapping(agent.get("retrieval"))),
        source_reading=SourceReadingConfig(
            **_mapping(agent.get("source_reading"))
        ),
        context=ContextConfig(**_mapping(agent.get("context"))),
        memory=MemoryConfig(**_mapping(agent.get("memory"))),
        safety=SafetyConfig(**_mapping(agent.get("safety"))),
    )


@lru_cache(maxsize=1)
def load_tool_configs() -> list[ToolConfig]:
    """Load enabled tool registrations."""
    data = _load_yaml(_config_path("tools.yml"))
    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list):
        return []
    tools: list[ToolConfig] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        tools.append(
            ToolConfig(
                name=str(item["name"]),
                description=str(item["description"]),
                capabilities=[str(value) for value in item.get("capabilities", [])],
                requires=[str(value) for value in item.get("requires", [])],
                enabled=bool(item.get("enabled", True)),
            )
        )
    return [tool for tool in tools if tool.enabled]


@lru_cache(maxsize=1)
def load_intent_configs() -> dict[str, IntentConfig]:
    """Load query intent routing metadata."""
    data = _load_yaml(_config_path("intents.yml"))
    raw_intents = data.get("intents")
    if not isinstance(raw_intents, dict):
        return {}
    intents: dict[str, IntentConfig] = {}
    for raw_name, raw_item in raw_intents.items():
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_name).strip()
        if not name:
            continue
        intents[name] = IntentConfig(
            name=name,
            description=str(raw_item.get("description", "")),
            default_tool=str(raw_item.get("default_tool", "")),
            default_filters=_mapping(raw_item.get("default_filters")),
            chinese_terms=_string_list(raw_item.get("chinese_terms")),
            english_terms=_string_list(raw_item.get("english_terms")),
        )
    return intents


@lru_cache(maxsize=1)
def load_target_trace_config() -> TargetTraceConfig:
    """Load target trace heuristics and defaults."""
    data = _load_yaml(_config_path("target_trace.yml"))
    raw_config = _mapping(data.get("target_trace"))
    return TargetTraceConfig(
        min_target_length=_int_value(raw_config.get("min_target_length"), 3),
        default_source_priority=_int_value(
            raw_config.get("default_source_priority"),
            40,
        ),
        generic_terms=_string_list(raw_config.get("generic_terms")),
        class_suffixes=_string_list(raw_config.get("class_suffixes")),
        symbolic_chars=_string_list(raw_config.get("symbolic_chars")),
        persistence_edge_kinds=_string_list(
            raw_config.get("persistence_edge_kinds")
        ),
        table_target_kinds=_string_list(raw_config.get("table_target_kinds")),
        source_priorities=_int_mapping(raw_config.get("source_priorities")),
    )


@lru_cache(maxsize=1)
def load_scope_fallback_config() -> ScopeFallbackConfig:
    """Load fallback options for resolved module scopes with no local hits."""
    data = _load_yaml(_config_path("scope_fallback.yml"))
    raw_config = _mapping(data.get("scope_fallback"))
    return ScopeFallbackConfig(
        enabled=bool(raw_config.get("enabled", True)),
        related_service_symbol_kinds=_string_list(
            raw_config.get("related_service_symbol_kinds")
        ),
        related_service_limit=_int_value(
            raw_config.get("related_service_limit"),
            3,
        ),
        related_table_limit_per_service=_int_value(
            raw_config.get("related_table_limit_per_service"),
            20,
        ),
    )


@lru_cache(maxsize=1)
def load_task_planning_config() -> TaskPlanningConfig:
    """Load task planning trigger terms, default steps, and answer guidance."""
    data = _load_yaml(_config_path("task_planning.yml"))
    raw_config = _mapping(data.get("task_planning"))
    return TaskPlanningConfig(
        intent_name=str(raw_config.get("intent_name", "task_planning")),
        planning_terms=_string_list(raw_config.get("planning_terms")),
        action_terms=_string_list(raw_config.get("action_terms")),
        scope_terms=_string_list(raw_config.get("scope_terms")),
        default_steps=_dict_list(raw_config.get("default_steps")),
        output_sections=_string_list(raw_config.get("output_sections")),
        schema_decision_rules=_string_list_mapping(
            raw_config.get("schema_decision_rules")
        ),
        source_evidence_priorities=_int_mapping(
            raw_config.get("source_evidence_priorities")
        ),
        default_source_evidence_priority=_int_value(
            raw_config.get("default_source_evidence_priority"),
            50,
        ),
    )


def _config_path(name: str) -> Path:
    return Path(__file__).resolve().parents[4] / "config" / "rag" / name


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"RAG config must be a mapping: {path}")
    return cast(dict[str, Any], data)


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, Any], value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        try:
            result[str(raw_key)] = int(raw_value)
        except (TypeError, ValueError):
            continue
    return result


def _string_list_mapping(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(raw_key): _string_list(raw_value)
        for raw_key, raw_value in value.items()
    }


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]


def _int_value(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
