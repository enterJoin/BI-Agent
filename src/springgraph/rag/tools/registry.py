"""Tool registry backed by external tool configuration."""

from dataclasses import dataclass

from springgraph.rag.config.loader import load_tool_configs
from springgraph.rag.config.models import ToolConfig
from springgraph.rag.tools.implementations import create_tool
from springgraph.rag.tools.schemas import RagTool


@dataclass(frozen=True)
class ToolRegistry:
    """Runtime registry of available Agentic RAG tools."""

    tools: dict[str, RagTool]

    def get(self, name: str) -> RagTool | None:
        """Return a tool by name."""
        return self.tools.get(name)

    def require(self, name: str) -> RagTool:
        """Return a tool or raise a clear error."""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown RAG tool requested: {name}")
        return tool

    def names(self) -> list[str]:
        """Return tool names in registration order."""
        return list(self.tools)

    def configs(self) -> list[ToolConfig]:
        """Return registered tool metadata."""
        return [tool.config for tool in self.tools.values()]


def load_tool_registry() -> ToolRegistry:
    """Build the registry from tools.yml."""
    tools: dict[str, RagTool] = {}
    for config in load_tool_configs():
        tools[config.name] = create_tool(config)
    return ToolRegistry(tools=tools)
