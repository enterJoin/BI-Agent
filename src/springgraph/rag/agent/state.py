"""State for the controlled Agentic RAG graph."""

from typing import TypedDict

from springgraph.rag.config.models import AgenticRagConfig, ToolConfig
from springgraph.rag.schemas import RagEvidence, SourceSnippet
from springgraph.rag.tools.schemas import ToolResult


class QuestionUnderstanding(TypedDict, total=False):
    """Structured question understanding returned by LLM."""

    task_goal: str
    sub_questions: list[str]
    business_terms: list[str]
    technical_terms: list[str]
    entities: list[str]
    expected_evidence: list[str]


class AgentAction(TypedDict, total=False):
    """Next action chosen by the agent."""

    action: str
    tool_name: str
    query: str
    reason: str


class EvidenceJudgeResult(TypedDict, total=False):
    """Evidence sufficiency decision."""

    evidence_sufficient: bool
    missing_information: list[str]
    suggested_next_tools: list[str]


class AgenticRagState(TypedDict, total=False):
    """LangGraph state for Agentic RAG."""

    thread_id: str
    user_id: str | None
    project_path: str
    project_id: str
    question: str
    top_k: int
    graph_depth: int
    read_source: bool
    runtime_config: AgenticRagConfig
    tool_configs: list[ToolConfig]
    candidate_tools: list[str]
    source_available: bool
    source_reading_skipped_reason: str | None
    question_understanding: QuestionUnderstanding
    next_action: AgentAction
    judge_result: EvidenceJudgeResult
    tool_results: list[ToolResult]
    used_tools: list[str]
    observations: list[str]
    evidence: list[RagEvidence]
    source_snippets: list[SourceSnippet]
    answer: str
    warnings: list[str]
    iteration: int
    tool_call_count: int
    no_new_evidence_rounds: int
    last_source_read_evidence_count: int
    should_continue: bool
