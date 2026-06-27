"""Shared target tracing heuristics."""

import re

from springgraph.rag.config.loader import load_target_trace_config

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$.:/-]*")


def explicit_trace_target(query: str) -> str | None:
    """Return the first explicit trace target from a query."""
    for candidate in trace_target_tokens(query):
        return candidate
    return None


def trace_target_tokens(query: str) -> list[str]:
    """Return target-like tokens found in query text."""
    candidates: list[str] = []
    for match in TOKEN_RE.findall(query):
        candidate = match.strip(".,;:()[]{}<>\"'")
        if looks_like_trace_target(candidate) and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def looks_like_trace_target(candidate: str) -> bool:
    """Return whether a token is specific enough to trace in code indexes."""
    config = load_target_trace_config()
    lowered = candidate.lower()
    if len(lowered) < config.min_target_length:
        return False
    if lowered in _lowered_set(config.generic_terms):
        return False
    if any(char in lowered for char in config.symbolic_chars):
        return True
    if candidate.endswith(tuple(config.class_suffixes)):
        return True
    return any(char.isupper() for char in candidate[1:]) and candidate[0].isupper()


def target_candidates(query: str, explicit_values: list[str]) -> list[str]:
    """Merge configured targets and query tokens into ordered target candidates."""
    values = [value for value in explicit_values if value.strip()]
    values.extend(trace_target_tokens(query))
    return _dedupe(values)


def persistence_edge_kinds() -> list[str]:
    """Return default relation kinds for persistence target tracing."""
    return list(load_target_trace_config().persistence_edge_kinds)


def table_target_kinds() -> set[str]:
    """Return symbol kinds that represent database table targets."""
    return set(load_target_trace_config().table_target_kinds)


def source_priority(evidence_type: str) -> int:
    """Return source reading priority for one evidence type."""
    config = load_target_trace_config()
    return config.source_priorities.get(
        evidence_type,
        config.default_source_priority,
    )


def _lowered_set(values: list[str]) -> set[str]:
    return {value.lower() for value in values}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
