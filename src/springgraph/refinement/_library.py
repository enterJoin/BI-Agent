"""Markdown library ingestion for project knowledge chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from springgraph.hashing import digest
from springgraph.refinement._types import ChunkFact, RefinedFile, RefinementFacts

LIBRARY_DIR_NAME = "library"
MAX_LIBRARY_FILE_BYTES = 2 * 1024 * 1024
TARGET_CHUNK_TOKENS = 600
MAX_CHUNK_TOKENS = 800
OVERLAP_TOKENS = 120

HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_./:-]+")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s+")
IGNORED_DIRS = {
    ".git",
    ".idea",
    "target",
    "build",
    "node_modules",
    ".mvn",
}


@dataclass(frozen=True)
class MarkdownSection:
    """One heading-scoped section extracted from a Markdown file."""

    heading_path: tuple[str, ...]
    heading_title: str
    heading_level: int
    start_line: int
    end_line: int
    body_lines: list[str]


@dataclass(frozen=True)
class MarkdownChunkDraft:
    """A chunk draft built from a Markdown section."""

    body: str
    start_line: int
    end_line: int
    chunk_index: int
    chunk_count: int


def scan_library_files(root: Path) -> list[RefinedFile]:
    """Return Markdown files under the optional library directory."""
    resolved_root = root.resolve()
    library_root = resolved_root / LIBRARY_DIR_NAME
    if not library_root.exists() or not library_root.is_dir():
        return []

    files: list[RefinedFile] = []
    for path in sorted(library_root.rglob("*.md")):
        if not path.is_file() or _is_ignored(path):
            continue
        stat = path.stat()
        if stat.st_size > MAX_LIBRARY_FILE_BYTES:
            continue
        relative_path = path.relative_to(resolved_root).as_posix()
        files.append(
            RefinedFile(
                path=path,
                relative_path=relative_path,
                language="markdown",
                module_name=LIBRARY_DIR_NAME,
                service_name=None,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            )
        )
    return files


def extract_library_file_facts(path: Path, relative_path: str) -> RefinementFacts:
    """Extract Markdown knowledge chunks for a single library file."""
    source = _read_text(path)
    sections = _parse_sections(source, path.stem)
    chunks: list[ChunkFact] = []
    for section in sections:
        chunks.extend(_section_to_chunks(relative_path, section))
    return RefinementFacts(chunks=chunks)


def _parse_sections(source: str, fallback_title: str) -> list[MarkdownSection]:
    lines = source.splitlines()
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    in_fence = False
    section_start = 1
    section_heading_path: tuple[str, ...] | None = None
    section_heading_title = fallback_title
    section_heading_level = 0
    body_lines: list[str] = []
    seen_heading = False

    def flush(end_line: int) -> None:
        nonlocal section_start, section_heading_path, section_heading_title
        nonlocal section_heading_level, body_lines
        if section_heading_path is None:
            return
        if not body_lines and not seen_heading:
            return
        sections.append(
            MarkdownSection(
                heading_path=section_heading_path,
                heading_title=section_heading_title,
                heading_level=section_heading_level,
                start_line=section_start,
                end_line=end_line,
                body_lines=body_lines.copy(),
            )
        )
        body_lines = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if FENCE_RE.match(stripped):
            in_fence = not in_fence
            body_lines.append(line)
            continue
        if not in_fence:
            match = HEADING_RE.match(line)
            if match is not None:
                flush(line_number - 1)
                level = len(match.group("marks"))
                title = match.group("title").strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                section_heading_path = tuple(item[1] for item in heading_stack)
                section_heading_title = title
                section_heading_level = level
                section_start = line_number
                body_lines = []
                seen_heading = True
                continue
        body_lines.append(line)

    if section_heading_path is None:
        if not lines:
            return []
        return [
            MarkdownSection(
                heading_path=(fallback_title,),
                heading_title=fallback_title,
                heading_level=0,
                start_line=1,
                end_line=max(1, len(lines)),
                body_lines=lines.copy(),
            )
        ]

    if body_lines or seen_heading:
        sections.append(
            MarkdownSection(
                heading_path=section_heading_path,
                heading_title=section_heading_title,
                heading_level=section_heading_level,
                start_line=section_start,
                end_line=max(section_start, len(lines)),
                body_lines=body_lines.copy(),
            )
        )
    return sections


def _section_to_chunks(
    relative_path: str, section: MarkdownSection
) -> list[ChunkFact]:
    body = "\n".join(section.body_lines).strip()
    if not body:
        return []

    body_blocks = _split_blocks(section.body_lines)
    if not body_blocks:
        body_blocks = [body]

    body_drafts = _pack_blocks(body_blocks, section.start_line)
    parent_doc_id = _parent_doc_id(relative_path, section)
    parent_title = " > ".join(section.heading_path)
    parent_content = _build_section_text(section.heading_path, body)
    child_count = len(body_drafts)
    chunks: list[ChunkFact] = [
        _parent_chunk(
            relative_path=relative_path,
            section=section,
            parent_doc_id=parent_doc_id,
            parent_title=parent_title,
            parent_content=parent_content,
            child_count=child_count,
        )
    ]
    for index, draft in enumerate(body_drafts):
        before = (
            _tail_tokens(body_drafts[index - 1].body, OVERLAP_TOKENS)
            if index > 0
            else ""
        )
        after = (
            _head_tokens(body_drafts[index + 1].body, OVERLAP_TOKENS)
            if index + 1 < child_count
            else ""
        )
        content_parts = [
            "Chunk type: project_knowledge",
            f"Source: {relative_path}",
            f"Parent doc: {parent_doc_id}",
            f"Heading: {parent_title}",
        ]
        if before:
            content_parts.extend(["Context before:", before])
        content_parts.extend(["Content:", draft.body])
        if after:
            content_parts.extend(["Context after:", after])
        metadata = {
            "source_type": "library",
            "library_path": relative_path,
            "parent_doc_id": parent_doc_id,
            "parent_title": parent_title,
            "parent_content": parent_content,
            "heading_path": list(section.heading_path),
            "heading_level": section.heading_level,
            "chunk_index": index,
            "chunk_count": child_count,
            "chunk_role": "child",
            "overlap_tokens": OVERLAP_TOKENS,
            "parser_version": "library-markdown-v1",
            "template_version": "library-markdown-v1",
        }
        raw_key = (
            f"library:{relative_path}:{parent_doc_id}:{index}:{draft.start_line}:"
            f"{draft.end_line}:{draft.body}"
        )
        chunks.append(
            ChunkFact(
                key=f"chunk:{digest(raw_key, 32)}",
                file_path=relative_path,
                symbol_key=None,
                chunk_type="project_knowledge",
                title=(
                    parent_title
                    if child_count == 1
                    else f"{parent_title} ({index + 1}/{child_count})"
                ),
                content="\n".join(content_parts).strip(),
                language="markdown",
                start_line=draft.start_line,
                end_line=draft.end_line,
                metadata=metadata,
            )
        )
    return chunks


def _parent_chunk(
    relative_path: str,
    section: MarkdownSection,
    parent_doc_id: str,
    parent_title: str,
    parent_content: str,
    child_count: int,
) -> ChunkFact:
    content = "\n".join(
        [
            "Chunk type: project_knowledge_parent",
            f"Source: {relative_path}",
            f"Parent doc: {parent_doc_id}",
            f"Heading: {parent_title}",
            "Full context:",
            parent_content,
        ]
    ).strip()
    metadata = {
        "source_type": "library",
        "library_path": relative_path,
        "parent_doc_id": parent_doc_id,
        "parent_title": parent_title,
        "parent_content": parent_content,
        "heading_path": list(section.heading_path),
        "heading_level": section.heading_level,
        "chunk_index": 0,
        "chunk_count": child_count,
        "chunk_role": "parent",
        "child_count": child_count,
        "overlap_tokens": OVERLAP_TOKENS,
        "parser_version": "library-markdown-v1",
        "template_version": "library-markdown-v1",
    }
    raw_key = (
        f"library-parent-chunk:{relative_path}:{parent_doc_id}:"
        f"{section.start_line}:{section.end_line}:{parent_content}"
    )
    return ChunkFact(
        key=f"chunk:{digest(raw_key, 32)}",
        file_path=relative_path,
        symbol_key=None,
        chunk_type="project_knowledge_parent",
        title=parent_title,
        content=content,
        language="markdown",
        start_line=section.start_line,
        end_line=section.end_line,
        metadata=metadata,
    )


def _pack_blocks(
    blocks: list[str],
    section_start_line: int,
) -> list[MarkdownChunkDraft]:
    drafts: list[MarkdownChunkDraft] = []
    current_blocks: list[str] = []
    current_tokens = 0
    current_start = section_start_line
    line_cursor = section_start_line

    for _block_index, block in enumerate(blocks):
        block_tokens = _count_tokens(block)
        if block_tokens > MAX_CHUNK_TOKENS:
            if current_blocks:
                body = "\n\n".join(current_blocks).strip()
                drafts.append(
                    MarkdownChunkDraft(
                        body=body,
                        start_line=current_start,
                        end_line=max(current_start, line_cursor - 1),
                        chunk_index=len(drafts),
                        chunk_count=0,
                    )
                )
                current_blocks = []
                current_tokens = 0
                current_start = line_cursor
            for text in _split_large_block(block):
                drafts.append(
                    MarkdownChunkDraft(
                        body=text,
                        start_line=line_cursor,
                        end_line=line_cursor,
                        chunk_index=len(drafts),
                        chunk_count=0,
                    )
                )
                current_start = line_cursor
                current_tokens = 0
            line_cursor += _line_count(block) + 1
            continue

        if current_blocks and current_tokens + block_tokens > MAX_CHUNK_TOKENS:
            body = "\n\n".join(current_blocks).strip()
            drafts.append(
                MarkdownChunkDraft(
                    body=body,
                    start_line=current_start,
                    end_line=max(current_start, line_cursor - 1),
                    chunk_index=len(drafts),
                    chunk_count=0,
                )
            )
            current_blocks = [block]
            current_tokens = block_tokens
            current_start = line_cursor
        else:
            if not current_blocks:
                current_start = line_cursor
            current_blocks.append(block)
            current_tokens += block_tokens

        if current_tokens >= TARGET_CHUNK_TOKENS:
            body = "\n\n".join(current_blocks).strip()
            drafts.append(
                MarkdownChunkDraft(
                    body=body,
                    start_line=current_start,
                    end_line=max(current_start, line_cursor + _line_count(block) - 1),
                    chunk_index=len(drafts),
                    chunk_count=0,
                )
            )
            current_blocks = []
            current_tokens = 0
            current_start = line_cursor + _line_count(block)

        line_cursor += _line_count(block) + 1

    if current_blocks:
        body = "\n\n".join(current_blocks).strip()
        drafts.append(
            MarkdownChunkDraft(
                body=body,
                start_line=current_start,
                end_line=max(current_start, line_cursor - 1),
                chunk_index=len(drafts),
                chunk_count=0,
            )
        )

    if not drafts:
        return []

    return [
        MarkdownChunkDraft(
            body=draft.body,
            start_line=draft.start_line,
            end_line=draft.end_line,
            chunk_index=draft.chunk_index,
            chunk_count=len(drafts),
        )
        for draft in drafts
    ]


def _split_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if FENCE_RE.match(stripped):
            current.append(line)
            in_fence = not in_fence
            if not in_fence:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        if in_fence:
            current.append(line)
            continue
        if not stripped:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _split_large_block(block: str) -> list[str]:
    if _count_tokens(block) <= MAX_CHUNK_TOKENS:
        return [block]
    sentences = [
        item.strip()
        for item in SENTENCE_SPLIT_RE.split(block)
        if item.strip()
    ]
    if not sentences:
        sentences = [line.strip() for line in block.splitlines() if line.strip()]
    if not sentences:
        return [block]
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)
        if current and current_tokens + sentence_tokens > MAX_CHUNK_TOKENS:
            pieces.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
            continue
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        pieces.append(" ".join(current).strip())
    return pieces or [block]


def _build_section_text(heading_path: tuple[str, ...], body: str) -> str:
    return "\n".join(
        [
            f"Heading: {' > '.join(heading_path)}",
            "Content:",
            body,
        ]
    ).strip()


def _parent_doc_id(relative_path: str, section: MarkdownSection) -> str:
    raw = (
        f"library-parent:{relative_path}:{section.heading_level}:"
        f"{'>'.join(section.heading_path)}:{section.start_line}:{section.end_line}"
    )
    return f"library-parent:{digest(raw, 32)}"


def _count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _head_tokens(text: str, token_count: int) -> str:
    spans = list(TOKEN_RE.finditer(text))
    if not spans:
        return ""
    if token_count >= len(spans):
        return text.strip()
    return text[: spans[token_count - 1].end()].strip()


def _tail_tokens(text: str, token_count: int) -> str:
    spans = list(TOKEN_RE.finditer(text))
    if not spans:
        return ""
    if token_count >= len(spans):
        return text.strip()
    return text[spans[-token_count].start() :].strip()


def _line_count(text: str) -> int:
    return max(1, text.count("\n") + 1)


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
