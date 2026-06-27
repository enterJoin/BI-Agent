"""Task planning intent helpers."""

from copy import deepcopy
from typing import Any, cast

from springgraph.rag.config.loader import load_task_planning_config


def looks_like_task_planning(query: str) -> bool:
    """Return whether a query asks for a development task plan."""
    config = load_task_planning_config()
    lowered = query.lower()
    planning_match = _contains_any(query, lowered, config.planning_terms)
    action_match = _contains_any(query, lowered, config.action_terms)
    scope_match = _contains_any(query, lowered, config.scope_terms)
    return planning_match or (action_match and scope_match)


def task_planning_intent_name() -> str:
    """Return configured task planning intent name."""
    return load_task_planning_config().intent_name


def task_planning_default_steps(question: str) -> list[dict[str, Any]]:
    """Return configured default retrieval steps for task planning."""
    steps: list[dict[str, Any]] = []
    for raw_step in load_task_planning_config().default_steps:
        step = cast(dict[str, Any], deepcopy(raw_step))
        query = step.get("query")
        if isinstance(query, str):
            step["query"] = query.format(question=question)
        filters = step.get("filters")
        if not isinstance(filters, dict):
            step["filters"] = {}
        steps.append(step)
    return steps


def task_planning_tool_order() -> list[str]:
    """Return configured task planning tool order."""
    result: list[str] = []
    for step in load_task_planning_config().default_steps:
        tool_name = step.get("tool_name")
        if isinstance(tool_name, str) and tool_name and tool_name not in result:
            result.append(tool_name)
    return result


def task_planning_source_priority(evidence_type: str) -> int:
    """Return source-reading priority for task planning evidence."""
    config = load_task_planning_config()
    return config.source_evidence_priorities.get(
        evidence_type,
        config.default_source_evidence_priority,
    )


def task_planning_prompt_guidance() -> str:
    """Return final-answer prompt guidance assembled from config."""
    config = load_task_planning_config()
    sections = "\n".join(f"- {section}" for section in config.output_sections)
    add_column_rules = "\n".join(
        f"- {rule}"
        for rule in config.schema_decision_rules.get("prefer_add_column_when", [])
    )
    new_table_rules = "\n".join(
        f"- {rule}"
        for rule in config.schema_decision_rules.get("prefer_new_table_when", [])
    )
    return "\n".join(
        [
            "Task planning answer sections:",
            sections or "- 任务理解\n- 推荐方案\n- 测试建议",
            "",
            "Prefer adding a column when:",
            add_column_rules
            or "- evidence supports a one-to-one existing entity attribute",
            "",
            "Prefer creating a new table when:",
            new_table_rules or "- evidence supports independent or one-to-many data",
        ]
    )


def _contains_any(text: str, lowered_text: str, terms: list[str]) -> bool:
    for term in terms:
        normalized = term.strip()
        if not normalized:
            continue
        if _is_ascii(normalized):
            if normalized.lower() in lowered_text:
                return True
        elif normalized in text:
            return True
    return False


def _is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True
