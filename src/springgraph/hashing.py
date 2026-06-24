"""Stable ID and content hashing helpers."""

from hashlib import sha256
from pathlib import Path


def digest(value: str, length: int = 32) -> str:
    """Return a shortened SHA-256 digest."""
    return sha256(value.encode("utf-8")).hexdigest()[:length]


def project_id(root: Path) -> str:
    """Return a stable project ID for a root directory."""
    return f"project:{digest(str(root.resolve()))}"


def file_row_id(project_id_value: str, relative_path: str) -> str:
    """Return the files table row ID."""
    return f"file-row:{digest(project_id_value + ':' + relative_path)}"


def file_symbol_id(project_id_value: str, relative_path: str) -> str:
    """Return the symbol ID for a file."""
    return f"file:{digest(project_id_value + ':' + relative_path)}"


def symbol_id(
    project_id_value: str,
    file_path: str,
    kind: str,
    qualified_name: str,
    start_line: int,
) -> str:
    """Return a stable symbol ID."""
    raw = f"{project_id_value}:{file_path}:{kind}:{qualified_name}:{start_line}"
    return f"{kind}:{digest(raw)}"


def route_symbol_id(
    project_id_value: str,
    http_method: str,
    route_path: str,
    handler_qualified_name: str,
) -> str:
    """Return a stable route symbol ID."""
    raw = f"{project_id_value}:{http_method}:{route_path}:{handler_qualified_name}"
    return f"route:{digest(raw)}"


def resource_symbol_id(
    project_id_value: str,
    resource_type: str,
    normalized_resource_name: str,
) -> str:
    """Return a stable resource symbol ID."""
    raw = f"{project_id_value}:{resource_type}:{normalized_resource_name}"
    return f"resource:{digest(raw)}"


def config_symbol_id(
    project_id_value: str,
    relative_path: str,
    config_key: str,
) -> str:
    """Return a stable config symbol ID."""
    raw = f"{project_id_value}:{relative_path}:{config_key}"
    return f"config:{digest(raw)}"


def content_hash(content: str) -> str:
    """Return the full SHA-256 hash for file content."""
    return sha256(content.encode("utf-8")).hexdigest()
