"""Domain objects used by extractors and indexers."""

from dataclasses import dataclass, field
from typing import Literal

SymbolKind = Literal[
    "file",
    "package",
    "import",
    "class",
    "interface",
    "enum",
    "annotation",
    "method",
    "constructor",
    "field",
    "parameter",
    "route",
    "bean",
    "config",
    "resource",
]

EdgeKind = Literal[
    "contains",
    "imports",
    "calls",
    "instantiates",
    "extends",
    "implements",
    "annotates",
    "injects",
    "maps_route",
    "references",
    "uses_resource",
    "publishes",
    "consumes",
]


@dataclass(frozen=True)
class SymbolData:
    """Extracted symbol."""

    id: str
    kind: SymbolKind
    name: str
    qualified_name: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    signature: str | None = None
    docstring: str | None = None
    annotations: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeData:
    """Extracted graph edge."""

    source_id: str
    target_id: str
    kind: EdgeKind
    line: int | None = None
    column_no: int | None = None
    confidence: float = 1.0
    resolved_by: str = "extractor"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class UnresolvedRefData:
    """Reference that needs a second-pass resolver."""

    from_symbol_id: str
    file_path: str
    reference_name: str
    reference_kind: EdgeKind
    line: int
    column_no: int
    candidates: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Symbols, edges, and unresolved refs extracted from a file."""

    symbols: list[SymbolData]
    edges: list[EdgeData]
    unresolved_refs: list[UnresolvedRefData]
    errors: list[str] = field(default_factory=list)
