"""Load project-specific RAG query hints from the library directory."""

import re
from pathlib import Path
from typing import Any

import yaml

LIBRARY_DIR_NAME = "library"
_MARKDOWN_HINT_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(.+?)\s*(?:=>|->|:|：|≈)\s*(.+?)\s*$"
)
_SUPPORTED_SUFFIXES = {".yml", ".yaml", ".md", ".markdown"}


def load_query_hints(project_path: Path | None) -> dict[str, list[str]]:
    """Load query expansion hints from a project's optional library directory."""
    if project_path is None:
        return {}
    library_root = project_path.expanduser().resolve() / LIBRARY_DIR_NAME
    if not library_root.exists() or not library_root.is_dir():
        return {}

    hints: dict[str, list[str]] = {}
    for path in sorted(library_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() in {".yml", ".yaml"}:
            _merge_hints(hints, _load_yaml_hints(path))
        else:
            _merge_hints(hints, _load_markdown_hints(path))
    return hints


def _load_yaml_hints(path: Path) -> dict[str, list[str]]:
    try:
        payload = yaml.safe_load(_read_text(path))
    except yaml.YAMLError:
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_hints = payload.get("query_hints", payload)
    if not isinstance(raw_hints, dict):
        return {}
    return _coerce_hints(raw_hints)


def _load_markdown_hints(path: Path) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {}
    for line in _read_text(path).splitlines():
        match = _MARKDOWN_HINT_RE.match(line)
        if match is None:
            continue
        keyword = _clean_token(match.group(1))
        values = _split_values(match.group(2))
        if keyword and values:
            hints[keyword] = values
    return hints


def _coerce_hints(raw_hints: dict[Any, Any]) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {}
    for raw_key, raw_value in raw_hints.items():
        key = _clean_token(str(raw_key))
        if not key:
            continue
        values = _coerce_values(raw_value)
        if values:
            hints[key] = values
    return hints


def _coerce_values(raw_value: object) -> list[str]:
    if isinstance(raw_value, list):
        return [
            cleaned
            for item in raw_value
            if (cleaned := _clean_token(str(item)))
        ]
    if isinstance(raw_value, str):
        return _split_values(raw_value)
    return []


def _split_values(raw_value: str) -> list[str]:
    values = [
        cleaned
        for item in re.split(r"[,，、/\s]+", raw_value)
        if (cleaned := _clean_token(item))
    ]
    return _preserve_negative_phrases(values)


def _preserve_negative_phrases(values: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value.lower() == "not" and index + 1 < len(values):
            result.append(f"not {values[index + 1]}")
            index += 2
            continue
        result.append(value)
        index += 1
    return result


def _merge_hints(
    target: dict[str, list[str]],
    source: dict[str, list[str]],
) -> None:
    for key, values in source.items():
        existing = target.setdefault(key, [])
        for value in values:
            if value not in existing:
                existing.append(value)


def _clean_token(value: str) -> str:
    return value.strip().strip("`'\"[]【】()（）")


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
