"""Load Agentic RAG configuration from YAML files."""

from pathlib import Path
from typing import Any, cast

import yaml

from springgraph.rag.config.models import (
    AgenticRagConfig,
    ContextConfig,
    MemoryConfig,
    RetrievalConfig,
    SafetyConfig,
    SourceReadingConfig,
    ToolConfig,
)


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
