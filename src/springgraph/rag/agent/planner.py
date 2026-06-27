"""LLM planning helpers for controlled Agentic RAG."""

import json
from typing import Any, cast

from springgraph.rag.agent.state import (
    AgentAction,
    EvidenceJudgeResult,
    QuestionUnderstanding,
)
from springgraph.rag.config.models import ToolConfig
from springgraph.rag.llm import invoke_agent_model
from springgraph.rag.prompts.loader import load_prompt
from springgraph.rag.schemas import RagEvidence, SourceSnippet


def understand_question(question: str) -> QuestionUnderstanding:
    """Ask the LLM to understand the question without fixed intent enums."""
    prompt = "\n\n".join(
        [
            load_prompt("question_understanding.md"),
            f"User question:\n{question}",
        ]
    )
    payload = _invoke_json(prompt)
    return {
        "task_goal": _string(payload.get("task_goal"), question),
        "sub_questions": _string_list(payload.get("sub_questions")),
        "business_terms": _string_list(payload.get("business_terms")),
        "technical_terms": _string_list(payload.get("technical_terms")),
        "entities": _string_list(payload.get("entities")),
        "expected_evidence": _string_list(payload.get("expected_evidence")),
    }


def decide_next_action(
    question: str,
    understanding: QuestionUnderstanding,
    candidate_tools: list[ToolConfig],
    evidence: list[RagEvidence],
    source_available: bool,
    observations: list[str],
    judge_result: EvidenceJudgeResult | None,
) -> AgentAction:
    """Ask the LLM to choose the next tool or finish."""
    prompt = "\n\n".join(
        [
            load_prompt("tool_decision.md"),
            f"Question:\n{question}",
            f"Question understanding:\n{json.dumps(understanding, ensure_ascii=False)}",
            f"Available tools:\n{_tools_json(candidate_tools)}",
            f"Source available: {source_available}",
            f"Current evidence summary:\n{_evidence_summary(evidence)}",
            f"Observations:\n{json.dumps(observations[-8:], ensure_ascii=False)}",
            f"Judge result:\n{json.dumps(judge_result or {}, ensure_ascii=False)}",
        ]
    )
    payload = _invoke_json(prompt)
    action = _string(payload.get("action"), "final_answer")
    if action != "call_tool":
        return {"action": "final_answer", "reason": _string(payload.get("reason"), "")}
    return {
        "action": "call_tool",
        "tool_name": _string(payload.get("tool_name"), ""),
        "query": _string(payload.get("query"), question),
        "reason": _string(payload.get("reason"), ""),
    }


def judge_evidence(
    question: str,
    understanding: QuestionUnderstanding,
    evidence: list[RagEvidence],
    source_snippets: list[SourceSnippet],
    observations: list[str],
) -> EvidenceJudgeResult:
    """Ask the LLM whether current evidence is sufficient."""
    prompt = "\n\n".join(
        [
            load_prompt("evidence_judge.md"),
            f"Question:\n{question}",
            f"Question understanding:\n{json.dumps(understanding, ensure_ascii=False)}",
            f"Evidence summary:\n{_evidence_summary(evidence)}",
            f"Source snippets:\n{_snippet_summary(source_snippets)}",
            f"Observations:\n{json.dumps(observations[-8:], ensure_ascii=False)}",
        ]
    )
    payload = _invoke_json(prompt)
    return {
        "evidence_sufficient": bool(payload.get("evidence_sufficient", False)),
        "missing_information": _string_list(payload.get("missing_information")),
        "suggested_next_tools": _string_list(payload.get("suggested_next_tools")),
    }


def build_final_prompt(
    question: str,
    understanding: QuestionUnderstanding,
    evidence: list[RagEvidence],
    source_snippets: list[SourceSnippet],
    observations: list[str],
    source_reading_skipped_reason: str | None,
) -> str:
    """Build the final answer prompt."""
    return "\n\n".join(
        [
            load_prompt("final_answer.md"),
            f"Question:\n{question}",
            f"Question understanding:\n{json.dumps(understanding, ensure_ascii=False)}",
            f"Tool observations:\n{json.dumps(observations, ensure_ascii=False)}",
            f"Evidence:\n{_evidence_summary(evidence)}",
            f"Source snippets:\n{_snippet_summary(source_snippets)}",
            f"Source reading skipped reason: {source_reading_skipped_reason}",
        ]
    )


def _invoke_json(prompt: str) -> dict[str, Any]:
    raw = invoke_agent_model(prompt)
    text = _strip_json_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON response must be an object.")
    return cast(dict[str, Any], payload)


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _tools_json(tools: list[ToolConfig]) -> str:
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "capabilities": tool.capabilities,
            "requires": tool.requires,
        }
        for tool in tools
    ]
    return json.dumps(payload, ensure_ascii=False)


def _evidence_summary(evidence: list[RagEvidence]) -> str:
    lines: list[str] = []
    for index, item in enumerate(evidence[:30], start=1):
        location = item.file_path or "unknown"
        if item.start_line is not None:
            end_line = item.end_line or item.start_line
            location = f"{location}:{item.start_line}-{end_line}"
        lines.append(
            f"{index}. [{item.source}/{item.evidence_type}] "
            f"{item.symbol or item.content_excerpt or ''} at {location}"
        )
    return "\n".join(lines) if lines else "No evidence yet."


def _snippet_summary(snippets: list[SourceSnippet]) -> str:
    lines: list[str] = []
    for index, snippet in enumerate(snippets[:8], start=1):
        lines.append(
            f"{index}. {snippet.file_path}:{snippet.start_line}-{snippet.end_line}\n"
            f"{snippet.content[:1200]}"
        )
    return "\n\n".join(lines) if lines else "No source snippets."


def _string(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
