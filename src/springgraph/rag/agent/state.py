"""State for the controlled Agentic RAG graph."""

from typing import Any, TypedDict

from springgraph.rag.config.models import AgenticRagConfig, ToolConfig
from springgraph.rag.schemas import RagEvidence, SourceSnippet
from springgraph.rag.tools.schemas import ToolResult


class QuestionUnderstanding(TypedDict, total=False):
    """Structured question understanding returned by LLM."""

    task_goal: str
    intent: str
    sub_questions: list[str]
    business_terms: list[str]
    technical_terms: list[str]
    entities: list[str]
    expected_evidence: list[str]


class PlanStep(TypedDict, total=False):
    """One planned retrieval step."""

    tool_name: str
    query: str
    filters: dict[str, Any]
    reason: str


class RetrievalPlan(TypedDict, total=False):
    """A structured retrieval plan generated once per question."""

    task_goal: str
    steps: list[PlanStep]


class AgenticRagState(TypedDict, total=False):
    """LangGraph state for Agentic RAG."""

    thread_id: str
    load_memory: bool
    user_id: str | None
    project_path: str
    project_id: str
    question: str
    title: str | None
    top_k: int
    graph_depth: int
    read_source: bool
    runtime_config: AgenticRagConfig
    tool_configs: list[ToolConfig]
    source_available: bool
    source_reading_skipped_reason: str | None
    question_understanding: QuestionUnderstanding
    retrieval_plan: RetrievalPlan
    tool_results: list[ToolResult]
    used_tools: list[str]
    observations: list[str]
    evidence: list[RagEvidence]
    source_snippets: list[SourceSnippet]
    conversation_history: list[dict[str, str]]
    answer: str
    warnings: list[str]
