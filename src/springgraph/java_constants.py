"""Helpers for resolving Java string constants from project source."""

import re
from functools import lru_cache
from pathlib import Path

PACKAGE_RE = re.compile(r"^\s*package\s+(?P<name>[\w.]+)\s*;", re.MULTILINE)
CLASS_RE = re.compile(r"\b(?:class|interface|enum)\s+(?P<name>[A-Za-z_]\w*)")
STRING_CONSTANT_RE = re.compile(
    r"\b(?:public|protected|private)?\s*static\s+final\s+String\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+);"
)
STRING_LITERAL_RE = re.compile(r'"(?P<value>(?:\\.|[^"\\])*)"')
SKIPPED_DIRS = {
    ".git",
    ".idea",
    ".gradle",
    ".mvn",
    "build",
    "node_modules",
    "target",
}


def constants_for_file(path: Path, relative_path: str) -> dict[str, str]:
    """Return project Java string constants addressable from one file."""
    root = project_root_for(path, relative_path)
    if root is None:
        return {}
    return constants_for_root(str(root))


def project_root_for(path: Path, relative_path: str) -> Path | None:
    """Infer project root from an absolute path and stored relative path."""
    resolved = path.expanduser().resolve()
    relative = Path(relative_path)
    try:
        root = resolved
        for _ in relative.parts:
            root = root.parent
        if root.exists():
            return root
    except (OSError, RuntimeError):
        return None
    return None


@lru_cache(maxsize=16)
def constants_for_root(root: str) -> dict[str, str]:
    """Scan Java files under root and return string constant aliases."""
    root_path = Path(root)
    constants: dict[str, str] = {}
    simple_values: dict[str, set[str]] = {}
    for path in _java_files(root_path):
        source = _read_text(path)
        package = _first_match(PACKAGE_RE, source, "name")
        class_name = _first_match(CLASS_RE, source, "name")
        if not class_name:
            continue
        class_prefix = f"{package}.{class_name}" if package else class_name
        for match in STRING_CONSTANT_RE.finditer(source):
            name = match.group("name")
            value = _string_value(match.group("value"))
            if value is None:
                continue
            constants[f"{class_prefix}.{name}"] = value
            constants[f"{class_name}.{name}"] = value
            simple_values.setdefault(name, set()).add(value)
    for name, values in simple_values.items():
        if len(values) == 1:
            constants[name] = next(iter(values))
    return constants


def resolve_java_string_expression(
    expression: str,
    constants: dict[str, str],
) -> list[str]:
    """Resolve a Java expression to possible string values."""
    value = clean_java_expression(expression)
    if not value:
        return []
    if value in constants:
        return [constants[value]]
    literal = _string_value(value)
    if literal is not None:
        return [literal]
    concatenated = _concat_string_value(value, constants)
    if concatenated is not None:
        return [concatenated]
    return [value]


def clean_java_expression(value: str) -> str:
    """Trim Java expression delimiters used around topic values."""
    return value.strip().strip('"')


def _java_files(root: Path) -> list[Path]:
    try:
        return [
            path
            for path in root.rglob("*.java")
            if not any(part in SKIPPED_DIRS for part in path.parts)
        ]
    except OSError:
        return []


def _string_value(expression: str) -> str | None:
    text = expression.strip()
    match = STRING_LITERAL_RE.fullmatch(text)
    if match is None:
        return None
    return bytes(match.group("value"), "utf-8").decode("unicode_escape")


def _concat_string_value(
    expression: str,
    constants: dict[str, str],
) -> str | None:
    parts = [part.strip() for part in expression.split("+")]
    if len(parts) < 2:
        return None
    values: list[str] = []
    for part in parts:
        if part in constants:
            values.append(constants[part])
            continue
        literal = _string_value(part)
        if literal is None:
            return None
        values.append(literal)
    return "".join(values)


def _first_match(pattern: re.Pattern[str], source: str, group: str) -> str | None:
    match = pattern.search(source)
    return match.group(group) if match else None


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
