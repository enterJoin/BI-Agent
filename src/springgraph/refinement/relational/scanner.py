"""Project file scanner."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

JAVA_PATTERNS = ("src/main/java", "src/test/java")
CONFIG_NAMES = {
    "application.yml",
    "application.yaml",
    "application.properties",
    "bootstrap.yml",
    "bootstrap.yaml",
    "bootstrap.properties",
}
BUILD_FILES = {"pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"}


@dataclass(frozen=True)
class ScannedFile:
    """A source or config file discovered in a project."""

    path: Path
    relative_path: str
    language: str
    module_name: str | None
    service_name: str | None
    size_bytes: int
    modified_at: datetime


def scan_project(root: Path) -> list[ScannedFile]:
    """Scan a Spring Boot project or workspace root."""
    resolved_root = root.resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(
            f"Project path does not exist or is not a directory: {root}"
        )

    files: list[ScannedFile] = []
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = _relative_path(resolved_root, path)
        language = _language_for(relative_path, path.name)
        if language is None:
            continue
        module_name = module_name_for(resolved_root, path)
        service_name = service_name_for(resolved_root, path, module_name)
        stat = path.stat()
        files.append(
            ScannedFile(
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


def module_name_for(root: Path, path: Path) -> str | None:
    """Infer the Maven/Gradle module name for a file."""
    root = root.resolve()
    path = path.resolve()
    current = path.parent
    while current != root and current != current.parent:
        if any((current / build_file).exists() for build_file in BUILD_FILES):
            return current.name
        current = current.parent
    if any((root / build_file).exists() for build_file in BUILD_FILES):
        return root.name
    parts = path.relative_to(root).parts
    if len(parts) > 2 and parts[1] == "src":
        return parts[0]
    return root.name


def service_name_for(root: Path, path: Path, fallback: str | None) -> str | None:
    """Infer service name from nearby Spring config, falling back to module name."""
    module_root = _module_root(root.resolve(), path.resolve())
    for config_name in CONFIG_NAMES:
        config_path = module_root / "src" / "main" / "resources" / config_name
        if config_path.exists():
            service_name = _read_service_name(config_path)
            if service_name:
                return service_name
    return fallback


def _language_for(relative_path: str, name: str) -> str | None:
    normalized = relative_path.replace("\\", "/")
    if normalized.endswith(".java") and any(
        pattern in normalized for pattern in JAVA_PATTERNS
    ):
        return "java"
    if name in CONFIG_NAMES:
        return "config"
    return None


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _module_root(root: Path, path: Path) -> Path:
    current = path.parent
    while current != root and current != current.parent:
        if any((current / build_file).exists() for build_file in BUILD_FILES):
            return current
        current = current.parent
    return root


def _read_service_name(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="gbk", errors="ignore").splitlines()
    if path.suffix == ".properties":
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("spring.application.name="):
                return stripped.split("=", maxsplit=1)[1].strip()
    keys: list[str] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key = line.strip().split(":", maxsplit=1)[0].strip()
        value = line.strip().split(":", maxsplit=1)[1].strip() if ":" in line else ""
        level = indent // 2
        keys = keys[:level] + [key]
        if keys == ["spring", "application", "name"] and value:
            return value.strip("'\"")
    return None
