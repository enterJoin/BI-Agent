"""Source code reading with project path safety checks."""

from pathlib import Path

from springgraph.rag.schemas import RagEvidence, SourceSnippet

MAX_SOURCE_FILES = 8
MAX_SOURCE_LINES = 80
LINE_PADDING = 3


def check_source_path(project_path: Path | None) -> tuple[bool, str | None]:
    """Return whether source reading is possible and why it is not."""
    if project_path is None:
        return False, "project_path_not_provided"
    if not project_path.exists():
        return False, "project_path_not_found"
    if not project_path.is_dir():
        return False, "project_path_not_directory"
    return True, None


def read_source_snippets(
    project_path: Path,
    evidence: list[RagEvidence],
    max_files: int = MAX_SOURCE_FILES,
    max_lines: int = MAX_SOURCE_LINES,
    line_padding: int = LINE_PADDING,
) -> tuple[list[SourceSnippet], list[str]]:
    """Read source snippets referenced by evidence items."""
    root = project_path.resolve()
    snippets: list[SourceSnippet] = []
    warnings: list[str] = []
    seen_files: set[str] = set()
    for item in evidence:
        if item.file_path is None or item.file_path in seen_files:
            continue
        if len(seen_files) >= max_files:
            break
        seen_files.add(item.file_path)
        source_path = (root / item.file_path).resolve()
        if not _is_relative_to(source_path, root):
            warnings.append(
                f"Source path outside project root skipped: {item.file_path}"
            )
            continue
        if not source_path.exists() or not source_path.is_file():
            warnings.append(f"Source file not found: {item.file_path}")
            continue
        content = _read_text(source_path)
        lines = content.splitlines()
        if not lines:
            continue
        start_line, end_line = _line_window(
            item.start_line,
            item.end_line,
            len(lines),
            max_lines,
            line_padding,
        )
        snippet_text = "\n".join(lines[start_line - 1 : end_line])
        snippets.append(
            SourceSnippet(
                file_path=item.file_path,
                start_line=start_line,
                end_line=end_line,
                content=snippet_text,
            )
        )
    return snippets, warnings


def _line_window(
    start_line: int | None,
    end_line: int | None,
    total_lines: int,
    max_lines: int = MAX_SOURCE_LINES,
    line_padding: int = LINE_PADDING,
) -> tuple[int, int]:
    start = max(1, (start_line or 1) - line_padding)
    end = min(total_lines, (end_line or start + max_lines) + line_padding)
    if end - start + 1 > max_lines:
        end = min(total_lines, start + max_lines - 1)
    return start, end


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
