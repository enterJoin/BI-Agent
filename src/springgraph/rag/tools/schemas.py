"""Tool contracts for Agentic RAG."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from springgraph.rag.config.models import ToolConfig
from springgraph.rag.schemas import RagEvidence, SourceSnippet


@dataclass(frozen=True)
class ToolInput:
    """Common input passed to a RAG tool."""

    query: str
    filters: dict[str, Any]
    project_id: str
    project_path: Path
    top_k: int
    graph_depth: int
    source_available: bool
    max_source_files: int
    max_source_lines: int
    source_line_padding: int
    evidence: list[RagEvidence] = field(default_factory=list)
    thread_id: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """Standard result returned by every RAG tool."""

    tool_name: str
    summary: str
    evidence: list[RagEvidence] = field(default_factory=list)
    source_snippets: list[SourceSnippet] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RagTool(Protocol):
    """Executable tool protocol."""

    @property
    def config(self) -> ToolConfig:
        """Return tool metadata."""

    def invoke(self, tool_input: ToolInput) -> ToolResult:
        """Run the tool."""
