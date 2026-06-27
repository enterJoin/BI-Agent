"""Candidate tool ranking helpers."""

from math import sqrt

from springgraph.rag.config.models import AgenticRagConfig, ToolConfig
from springgraph.refinement._embedder import create_embedder


def propose_candidate_tools(
    question: str,
    tools: list[ToolConfig],
    config: AgenticRagConfig,
) -> list[ToolConfig]:
    """Rank tool candidates with embedding similarity when available."""
    if not tools:
        return []
    try:
        embedder = create_embedder()
        question_embedding = embedder.embed(question)
        scored = [
            (
                _cosine_similarity(
                    question_embedding,
                    embedder.embed(_tool_text(tool)),
                ),
                tool,
            )
            for tool in tools
        ]
    except Exception:  # noqa: BLE001
        return tools[: config.retrieval.top_candidate_tools]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        tool for _, tool in scored[: config.retrieval.top_candidate_tools]
    ]


def _tool_text(tool: ToolConfig) -> str:
    return " ".join([tool.name, tool.description, *tool.capabilities])


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(item * item for item in left))
    right_norm = sqrt(sum(item * item for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(
        left_item * right_item
        for left_item, right_item in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
