"""Query intent inference for Agentic RAG retrieval."""

from springgraph.rag.config.models import IntentConfig
from springgraph.rag.task_planning import (
    looks_like_task_planning,
    task_planning_intent_name,
)

TABLE_RETRIEVAL_INTENTS = frozenset({"persistence_location", "table_usage"})


def infer_query_intent(
    query: str,
    explicit_intent: str | None,
    group_by: str | None,
    intent_configs: dict[str, IntentConfig],
) -> str | None:
    """Infer a query intent from planner output and configured terms."""
    normalized_intent = _normalize(explicit_intent)
    if normalized_intent in intent_configs:
        return normalized_intent
    if _normalize(group_by) == "table_name":
        return "table_usage"
    if task_planning_intent_name() in intent_configs and looks_like_task_planning(
        query
    ):
        return task_planning_intent_name()

    lowered_query = query.lower()
    for intent_name, config in intent_configs.items():
        if _contains_any(query, config.chinese_terms):
            return intent_name
        if _contains_any(lowered_query, _lower_terms(config.english_terms)):
            return intent_name
    return None


def intent_default_filters(
    intent: str | None,
    intent_configs: dict[str, IntentConfig],
) -> dict[str, object]:
    """Return default filters for an intent."""
    if intent is None:
        return {}
    config = intent_configs.get(intent)
    if config is None:
        return {}
    return dict(config.default_filters)


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term and term in text for term in terms)


def _lower_terms(terms: list[str]) -> list[str]:
    return [term.lower() for term in terms]


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None
