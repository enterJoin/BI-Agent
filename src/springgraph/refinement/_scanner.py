"""File discovery for semantic refinement."""

from datetime import UTC, datetime
from pathlib import Path

from springgraph.refinement._types import RefinedFile
from springgraph.refinement.relational.scanner import module_name_for, service_name_for

CONFIG_NAMES = {
    "application.yml",
    "application.yaml",
    "application.properties",
    "bootstrap.yml",
    "bootstrap.yaml",
    "bootstrap.properties",
}

IGNORED_DIRS = {
    ".git",
    ".idea",
    "target",
    "build",
    "node_modules",
    ".mvn",
}


def scan_refinement_files(root: Path) -> list[RefinedFile]:
    """Return files that can produce refinement facts or chunks."""
    resolved_root = root.resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(
            f"Project path does not exist or is not a directory: {root}"
        )

    files: list[RefinedFile] = []
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file() or _is_ignored(path):
            continue
        relative_path = path.relative_to(resolved_root).as_posix()
        language = _language_for(path, relative_path)
        if language is None:
            continue
        module_name = module_name_for(resolved_root, path)
        service_name = service_name_for(resolved_root, path, module_name)
        stat = path.stat()
        files.append(
            RefinedFile(
                path=path,
                relative_path=relative_path,
                language=language,
                module_name=module_name,
                service_name=service_name,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            )
        )
    return files


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def _language_for(path: Path, relative_path: str) -> str | None:
    normalized = relative_path.replace("\\", "/")
    name = path.name
    if name in CONFIG_NAMES:
        return "config"
    if normalized.endswith(".java") and "/src/" in normalized:
        return "java"
    if normalized.endswith(".xml") and (
        "/mapper/" in normalized or "/mappers/" in normalized
    ):
        return "mybatis_xml"
    if normalized.endswith(".sql"):
        return "sql"
    if normalized.endswith((".html", ".htm")) and "/templates/" in normalized:
        return "html"
    return None
