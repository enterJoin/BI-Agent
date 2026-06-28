"""Java source extractor.

The first implementation keeps the public contract independent from the database
and focuses on common Spring Boot source. tree-sitter dependencies are part of
the project contract; this module keeps parsing helpers small so a fuller AST
walk can replace the line-oriented fallbacks without changing callers.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from springgraph.hashing import file_symbol_id, route_symbol_id, symbol_id
from springgraph.refinement.relational.domain import (
    EdgeData,
    ExtractionResult,
    SymbolData,
    SymbolKind,
    UnresolvedRefData,
)
from springgraph.refinement.relational.java_names import (
    class_fqn,
    member_fqn,
    package_name,
)
from springgraph.refinement.relational.spring_annotations import (
    http_method_for_annotation,
    is_component_annotation,
    is_route_annotation,
)

ANNOTATION_RE = re.compile(r"@(?P<name>[A-Za-z_][\w.]*)(?:\((?P<args>.*)\))?")
CLASS_RE = re.compile(
    r"\b(?P<kind>class|interface|enum|@interface)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?P<tail>[^{]*)"
)
FIELD_RE = re.compile(
    r"^\s*(?:private|protected|public)?\s*(?:static\s+)?(?:final\s+)?"
    r"(?P<type>[A-Z][\w<>?, ]*)\s+(?P<name>[a-zA-Z_]\w*)"
    r"\s*(?:=\s*(?P<value>[^;]+))?;"
)
METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private)?\s*(?:static\s+)?"
    r"(?P<return>[A-Za-z_][\w<>?, .]*)\s+"
    r"(?P<name>[a-zA-Z_]\w*)\s*\((?P<params>[^)]*)\)\s*(?:throws [^{]+)?\{?"
)
CTOR_RE = re.compile(
    r"^\s*(?:public|protected|private)?\s*(?P<name>[A-Z]\w*)\s*"
    r"\((?P<params>[^)]*)\)\s*(?:throws [^{]+)?\{?"
)
CALL_RE = re.compile(r"\b(?P<receiver>[a-zA-Z_]\w*)\.(?P<method>[a-zA-Z_]\w*)\s*\(")
NEW_RE = re.compile(r"\bnew\s+(?P<type>[A-Z]\w*)\s*\(")
IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?(?P<name>[\w.*]+)\s*;")


def extract_java_file(
    project_id_value: str,
    path: Path,
    relative_path: str,
    module_name: str | None = None,
    service_name: str | None = None,
) -> ExtractionResult:
    """Extract Java symbols and first-pass edges from a source file."""
    source = _read_text(path)
    lines = source.splitlines()
    symbols: list[SymbolData] = []
    edges: list[EdgeData] = []
    unresolved: list[UnresolvedRefData] = []

    file_id = file_symbol_id(project_id_value, relative_path)
    file_symbol = SymbolData(
        id=file_id,
        kind="file",
        name=path.name,
        qualified_name=relative_path,
        file_path=relative_path,
        language="java",
        start_line=1,
        end_line=max(1, len(lines)),
        start_column=0,
        end_column=0,
        metadata={"module_name": module_name, "service_name": service_name},
    )
    symbols.append(file_symbol)

    if _is_protected_or_binary(source):
        return ExtractionResult(
            symbols=symbols,
            edges=edges,
            unresolved_refs=unresolved,
            errors=[
                "File appears encrypted, protected, or binary; Java source "
                "cannot be parsed."
            ],
        )

    package = package_name(source)

    if package:
        package_id = symbol_id(project_id_value, relative_path, "package", package, 1)
        symbols.append(
            SymbolData(
                id=package_id,
                kind="package",
                name=package,
                qualified_name=package,
                file_path=relative_path,
                language="java",
                start_line=_first_line_containing(lines, f"package {package}") or 1,
                end_line=_first_line_containing(lines, f"package {package}") or 1,
                start_column=0,
                end_column=0,
            )
        )
        edges.append(EdgeData(source_id=file_id, target_id=package_id, kind="contains"))

    imports = _extract_imports(project_id_value, relative_path, lines)
    symbols.extend(imports)
    for import_symbol in imports:
        edges.append(
            EdgeData(source_id=file_id, target_id=import_symbol.id, kind="contains")
        )

    annotation_buffer: list[Annotation] = []
    class_symbol: SymbolData | None = None
    class_name: str | None = None
    class_qualified_name: str | None = None
    class_route = ""
    field_types: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        annotation = _parse_annotation(stripped)
        if annotation:
            annotation_buffer.append(annotation)
            i += 1
            continue

        class_match = CLASS_RE.search(line)
        if class_match:
            class_name = class_match.group("name")
            class_qualified_name = class_fqn(package, class_name)
            kind = _class_kind(class_match.group("kind"))
            class_symbol = SymbolData(
                id=symbol_id(
                    project_id_value,
                    relative_path,
                    kind,
                    class_qualified_name,
                    i + 1,
                ),
                kind=kind,
                name=class_name,
                qualified_name=class_qualified_name,
                file_path=relative_path,
                language="java",
                start_line=i + 1,
                end_line=_block_end_line(lines, i),
                start_column=0,
                end_column=len(line),
                annotations=[item.name for item in annotation_buffer],
                metadata={"module_name": module_name, "service_name": service_name},
            )
            symbols.append(class_symbol)
            edges.append(
                EdgeData(source_id=file_id, target_id=class_symbol.id, kind="contains")
            )
            _add_annotation_usages(
                project_id_value,
                relative_path,
                class_symbol,
                annotation_buffer,
                i + 1,
                symbols,
                edges,
                module_name,
                service_name,
            )
            _add_component_bean(
                project_id_value,
                relative_path,
                class_symbol,
                annotation_buffer,
                symbols,
                edges,
                module_name,
                service_name,
            )
            class_route = _route_path_from_annotations(annotation_buffer)
            _add_inheritance_refs(
                class_symbol,
                class_match.group("tail"),
                relative_path,
                i + 1,
                unresolved,
            )
            annotation_buffer = []
            i += 1
            continue

        if class_symbol and class_name and class_qualified_name:
            field_match = FIELD_RE.match(line)
            if field_match and "(" not in line:
                field_symbol = _field_symbol(
                    project_id_value,
                    relative_path,
                    class_qualified_name,
                    field_match,
                    i + 1,
                    module_name,
                    service_name,
                    annotation_buffer,
                )
                symbols.append(field_symbol)
                edges.append(
                    EdgeData(
                        source_id=class_symbol.id,
                        target_id=field_symbol.id,
                        kind="contains",
                    )
                )
                _add_annotation_usages(
                    project_id_value,
                    relative_path,
                    field_symbol,
                    annotation_buffer,
                    i + 1,
                    symbols,
                    edges,
                    module_name,
                    service_name,
                )
                field_types[field_match.group("name")] = _clean_type(
                    field_match.group("type")
                )
                if any(item.name == "Autowired" for item in annotation_buffer):
                    unresolved.append(
                        UnresolvedRefData(
                            from_symbol_id=field_symbol.id,
                            file_path=relative_path,
                            reference_name=_clean_type(field_match.group("type")),
                            reference_kind="injects",
                            line=i + 1,
                            column_no=0,
                            metadata={"member": field_match.group("name")},
                        )
                    )
                annotation_buffer = []
                i += 1
                continue

            ctor_match = CTOR_RE.match(line)
            if ctor_match and ctor_match.group("name") == class_name:
                body, end_index = _method_body(lines, i)
                ctor_symbol = _method_like_symbol(
                    project_id_value,
                    relative_path,
                    class_qualified_name,
                    "<init>",
                    "constructor",
                    line,
                    i + 1,
                    end_index + 1,
                    annotation_buffer,
                    module_name,
                    service_name,
                )
                symbols.append(ctor_symbol)
                edges.append(
                    EdgeData(
                        source_id=class_symbol.id,
                        target_id=ctor_symbol.id,
                        kind="contains",
                    )
                )
                _add_annotation_usages(
                    project_id_value,
                    relative_path,
                    ctor_symbol,
                    annotation_buffer,
                    i + 1,
                    symbols,
                    edges,
                    module_name,
                    service_name,
                )
                _add_parameters_and_injections(
                    project_id_value,
                    relative_path,
                    ctor_symbol,
                    _method_params(line, ctor_match.group("name")),
                    i + 1,
                    symbols,
                    edges,
                    unresolved,
                )
                _add_body_refs(
                    ctor_symbol, relative_path, body, i + 1, field_types, unresolved
                )
                annotation_buffer = []
                i = max(i + 1, end_index + 1)
                continue

            method_match = METHOD_RE.match(line)
            if method_match and not _is_control_statement(method_match.group("return")):
                method_name = method_match.group("name")
                body, end_index = _method_body(lines, i)
                method_symbol = _method_like_symbol(
                    project_id_value,
                    relative_path,
                    class_qualified_name,
                    method_name,
                    "method",
                    line,
                    i + 1,
                    end_index + 1,
                    annotation_buffer,
                    module_name,
                    service_name,
                )
                symbols.append(method_symbol)
                edges.append(
                    EdgeData(
                        source_id=class_symbol.id,
                        target_id=method_symbol.id,
                        kind="contains",
                    )
                )
                _add_annotation_usages(
                    project_id_value,
                    relative_path,
                    method_symbol,
                    annotation_buffer,
                    i + 1,
                    symbols,
                    edges,
                    module_name,
                    service_name,
                )
                _add_parameters_and_injections(
                    project_id_value,
                    relative_path,
                    method_symbol,
                    _method_params(line, method_name),
                    i + 1,
                    symbols,
                    edges,
                    unresolved,
                )
                _add_return_ref(
                    method_symbol,
                    relative_path,
                    method_match.group("return"),
                    i + 1,
                    unresolved,
                )
                _add_route(
                    project_id_value,
                    relative_path,
                    method_symbol,
                    class_route,
                    annotation_buffer,
                    i + 1,
                    symbols,
                    edges,
                )
                _add_bean_method(
                    project_id_value,
                    relative_path,
                    method_symbol,
                    annotation_buffer,
                    symbols,
                    edges,
                    module_name,
                    service_name,
                )
                _add_body_refs(
                    method_symbol, relative_path, body, i + 1, field_types, unresolved
                )
                annotation_buffer = []
                i = max(i + 1, end_index + 1)
                continue

        if stripped and not stripped.startswith("@") and not _is_comment_line(stripped):
            annotation_buffer = []
        i += 1

    return ExtractionResult(symbols=symbols, edges=edges, unresolved_refs=unresolved)


class Annotation:
    """Parsed Java annotation."""

    def __init__(self, name: str, args: str | None) -> None:
        self.name = name.rsplit(".", maxsplit=1)[-1]
        self.args = args or ""


@dataclass(frozen=True)
class ParsedParameter:
    """Parsed Java method or constructor parameter."""

    type_name: str
    name: str
    annotations: list[Annotation]


def _parse_annotation(line: str) -> Annotation | None:
    match = ANNOTATION_RE.match(line)
    if not match:
        return None
    return Annotation(match.group("name"), match.group("args"))


def _extract_imports(
    project_id_value: str, relative_path: str, lines: list[str]
) -> list[SymbolData]:
    symbols: list[SymbolData] = []
    for index, line in enumerate(lines, start=1):
        match = IMPORT_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        symbols.append(
            SymbolData(
                id=symbol_id(project_id_value, relative_path, "import", name, index),
                kind="import",
                name=name,
                qualified_name=name,
                file_path=relative_path,
                language="java",
                start_line=index,
                end_line=index,
                start_column=0,
                end_column=len(line),
            )
        )
    return symbols


def _class_kind(raw: str) -> SymbolKind:
    if raw == "@interface":
        return "annotation"
    if raw == "class":
        return "class"
    if raw == "interface":
        return "interface"
    return "enum"


def _field_symbol(
    project_id_value: str,
    relative_path: str,
    class_qualified_name: str,
    match: re.Match[str],
    line_number: int,
    module_name: str | None,
    service_name: str | None,
    annotations: list[Annotation],
) -> SymbolData:
    name = match.group("name")
    qualified_name = member_fqn(class_qualified_name, name)
    field_type = _clean_type(match.group("type"))
    value = match.group("value")
    metadata: dict[str, object] = {
        "type": field_type,
        "module_name": module_name,
        "service_name": service_name,
    }
    if value is not None:
        metadata["value"] = value.strip().strip('"')
    return SymbolData(
        id=symbol_id(
            project_id_value, relative_path, "field", qualified_name, line_number
        ),
        kind="field",
        name=name,
        qualified_name=qualified_name,
        file_path=relative_path,
        language="java",
        start_line=line_number,
        end_line=line_number,
        start_column=0,
        end_column=0,
        signature=f"{field_type} {name}",
        annotations=[item.name for item in annotations],
        metadata=metadata,
    )


def _method_like_symbol(
    project_id_value: str,
    relative_path: str,
    class_qualified_name: str,
    name: str,
    kind: SymbolKind,
    line: str,
    line_number: int,
    end_line: int,
    annotations: list[Annotation],
    module_name: str | None,
    service_name: str | None,
) -> SymbolData:
    qualified_name = member_fqn(class_qualified_name, name)
    return SymbolData(
        id=symbol_id(
            project_id_value, relative_path, kind, qualified_name, line_number
        ),
        kind=kind,
        name=name,
        qualified_name=qualified_name,
        file_path=relative_path,
        language="java",
        start_line=line_number,
        end_line=max(line_number, end_line),
        start_column=0,
        end_column=len(line),
        signature=line.strip(),
        annotations=[item.name for item in annotations],
        metadata={"module_name": module_name, "service_name": service_name},
    )


def _add_component_bean(
    project_id_value: str,
    relative_path: str,
    class_symbol: SymbolData,
    annotations: list[Annotation],
    symbols: list[SymbolData],
    edges: list[EdgeData],
    module_name: str | None,
    service_name: str | None,
) -> None:
    if not any(is_component_annotation(item.name) for item in annotations):
        return
    bean_id = symbol_id(
        project_id_value,
        relative_path,
        "bean",
        class_symbol.qualified_name,
        class_symbol.start_line,
    )
    symbols.append(
        SymbolData(
            id=bean_id,
            kind="bean",
            name=class_symbol.name[:1].lower() + class_symbol.name[1:],
            qualified_name=f"bean:{class_symbol.qualified_name}",
            file_path=relative_path,
            language="java",
            start_line=class_symbol.start_line,
            end_line=class_symbol.end_line,
            start_column=0,
            end_column=0,
            annotations=[item.name for item in annotations],
            metadata={
                "class": class_symbol.qualified_name,
                "module_name": module_name,
                "service_name": service_name,
            },
        )
    )
    edges.append(
        EdgeData(source_id=class_symbol.id, target_id=bean_id, kind="contains")
    )


def _add_bean_method(
    project_id_value: str,
    relative_path: str,
    method_symbol: SymbolData,
    annotations: list[Annotation],
    symbols: list[SymbolData],
    edges: list[EdgeData],
    module_name: str | None,
    service_name: str | None,
) -> None:
    if not any(item.name == "Bean" for item in annotations):
        return
    bean_id = symbol_id(
        project_id_value,
        relative_path,
        "bean",
        method_symbol.qualified_name,
        method_symbol.start_line,
    )
    symbols.append(
        SymbolData(
            id=bean_id,
            kind="bean",
            name=method_symbol.name,
            qualified_name=f"bean:{method_symbol.qualified_name}",
            file_path=relative_path,
            language="java",
            start_line=method_symbol.start_line,
            end_line=method_symbol.end_line,
            start_column=0,
            end_column=0,
            annotations=["Bean"],
            metadata={
                "factory_method": method_symbol.qualified_name,
                "module_name": module_name,
                "service_name": service_name,
            },
        )
    )
    edges.append(
        EdgeData(source_id=method_symbol.id, target_id=bean_id, kind="contains")
    )


def _add_annotation_usages(
    project_id_value: str,
    relative_path: str,
    target_symbol: SymbolData,
    annotations: list[Annotation],
    line_number: int,
    symbols: list[SymbolData],
    edges: list[EdgeData],
    module_name: str | None,
    service_name: str | None,
) -> None:
    for index, annotation in enumerate(annotations):
        values = _parse_annotation_values(annotation.args)
        primary_value = _annotation_primary_value(values)
        qualified_name = _annotation_usage_qualified_name(
            annotation.name,
            primary_value,
            target_symbol.qualified_name,
            index,
        )
        annotation_symbol = SymbolData(
            id=symbol_id(
                project_id_value,
                relative_path,
                "annotation_usage",
                qualified_name,
                line_number,
            ),
            kind="annotation_usage",
            name=_annotation_usage_name(annotation.name, primary_value),
            qualified_name=qualified_name,
            file_path=relative_path,
            language="java",
            start_line=line_number,
            end_line=line_number,
            start_column=0,
            end_column=0,
            signature=_annotation_signature(annotation),
            metadata={
                "annotation": annotation.name,
                "raw_args": annotation.args,
                "values": values,
                "target_kind": target_symbol.kind,
                "target_symbol_id": target_symbol.id,
                "target_qualified_name": target_symbol.qualified_name,
                "module_name": module_name,
                "service_name": service_name,
            },
        )
        symbols.append(annotation_symbol)
        edges.append(
            EdgeData(
                source_id=target_symbol.id,
                target_id=annotation_symbol.id,
                kind="annotated_by",
                line=line_number,
            )
        )
        edges.append(
            EdgeData(
                source_id=annotation_symbol.id,
                target_id=target_symbol.id,
                kind="annotates",
                line=line_number,
            )
        )


def _add_parameters_and_injections(
    project_id_value: str,
    relative_path: str,
    owner_symbol: SymbolData,
    params: str,
    line_number: int,
    symbols: list[SymbolData],
    edges: list[EdgeData],
    unresolved: list[UnresolvedRefData],
) -> None:
    for param in _parse_parameters(params):
        qualified_name = member_fqn(owner_symbol.qualified_name, param.name)
        parameter = SymbolData(
            id=symbol_id(
                project_id_value,
                relative_path,
                "parameter",
                qualified_name,
                line_number,
            ),
            kind="parameter",
            name=param.name,
            qualified_name=qualified_name,
            file_path=relative_path,
            language="java",
            start_line=line_number,
            end_line=line_number,
            start_column=0,
            end_column=0,
            signature=f"{param.type_name} {param.name}",
            annotations=[item.name for item in param.annotations],
            metadata={"type": param.type_name},
        )
        symbols.append(parameter)
        edges.append(
            EdgeData(source_id=owner_symbol.id, target_id=parameter.id, kind="contains")
        )
        _add_annotation_usages(
            project_id_value,
            relative_path,
            parameter,
            param.annotations,
            line_number,
            symbols,
            edges,
            _metadata_string(owner_symbol.metadata, "module_name"),
            _metadata_string(owner_symbol.metadata, "service_name"),
        )
        if owner_symbol.kind == "constructor" and _is_project_type(param.type_name):
            unresolved.append(
                UnresolvedRefData(
                    from_symbol_id=owner_symbol.id,
                    file_path=relative_path,
                    reference_name=param.type_name,
                    reference_kind="injects",
                    line=line_number,
                    column_no=0,
                    metadata={"parameter": param.name},
                )
            )


def _add_inheritance_refs(
    class_symbol: SymbolData,
    tail: str,
    relative_path: str,
    line_number: int,
    unresolved: list[UnresolvedRefData],
) -> None:
    extends = re.search(r"extends\s+([A-Z]\w*)", tail)
    if extends:
        unresolved.append(
            UnresolvedRefData(
                from_symbol_id=class_symbol.id,
                file_path=relative_path,
                reference_name=extends.group(1),
                reference_kind="extends",
                line=line_number,
                column_no=0,
            )
        )
    implements = re.search(r"implements\s+([A-Z][\w, ]*)", tail)
    if implements:
        for name in [item.strip() for item in implements.group(1).split(",")]:
            unresolved.append(
                UnresolvedRefData(
                    from_symbol_id=class_symbol.id,
                    file_path=relative_path,
                    reference_name=name,
                    reference_kind="implements",
                    line=line_number,
                    column_no=0,
                )
            )


def _add_return_ref(
    method_symbol: SymbolData,
    relative_path: str,
    return_type: str,
    line_number: int,
    unresolved: list[UnresolvedRefData],
) -> None:
    clean = _clean_type(return_type)
    if _is_project_type(clean):
        unresolved.append(
            UnresolvedRefData(
                from_symbol_id=method_symbol.id,
                file_path=relative_path,
                reference_name=clean,
                reference_kind="references",
                line=line_number,
                column_no=0,
                metadata={"usage": "return_type"},
            )
        )


def _add_route(
    project_id_value: str,
    relative_path: str,
    method_symbol: SymbolData,
    class_route: str,
    annotations: list[Annotation],
    line_number: int,
    symbols: list[SymbolData],
    edges: list[EdgeData],
) -> None:
    route_annotation = next(
        (item for item in annotations if is_route_annotation(item.name)), None
    )
    if route_annotation is None:
        return
    http_method = _http_method(route_annotation)
    method_route = _route_path(route_annotation)
    route_path = _join_paths(class_route, method_route)
    route_id = route_symbol_id(
        project_id_value, http_method, route_path, method_symbol.qualified_name
    )
    route_symbol = SymbolData(
        id=route_id,
        kind="route",
        name=f"{http_method} {route_path}",
        qualified_name=f"route:{http_method}:{route_path}:{method_symbol.qualified_name}",
        file_path=relative_path,
        language="java",
        start_line=line_number,
        end_line=line_number,
        start_column=0,
        end_column=0,
        metadata={
            "http_method": http_method,
            "path": route_path,
            "handler": method_symbol.qualified_name,
            "class_mapping": class_route,
            "method_mapping": method_route,
            "source": "spring_mvc",
        },
    )
    symbols.append(route_symbol)
    edges.append(
        EdgeData(source_id=method_symbol.id, target_id=route_id, kind="contains")
    )
    edges.append(
        EdgeData(source_id=route_id, target_id=method_symbol.id, kind="maps_route")
    )


def _add_body_refs(
    owner_symbol: SymbolData,
    relative_path: str,
    body: list[tuple[int, str]],
    line_number: int,
    field_types: dict[str, str],
    unresolved: list[UnresolvedRefData],
) -> None:
    for current_line, line in body:
        for match in CALL_RE.finditer(line):
            receiver = match.group("receiver")
            method = match.group("method")
            reference_name = f"{receiver}.{method}"
            metadata: dict[str, object] = {"receiver": receiver, "method": method}
            if receiver in field_types:
                metadata["receiver_type"] = field_types[receiver]
                reference_name = f"{field_types[receiver]}.{method}"
            unresolved.append(
                UnresolvedRefData(
                    from_symbol_id=owner_symbol.id,
                    file_path=relative_path,
                    reference_name=reference_name,
                    reference_kind="calls",
                    line=current_line,
                    column_no=match.start(),
                    metadata=metadata,
                )
            )
        for match in NEW_RE.finditer(line):
            unresolved.append(
                UnresolvedRefData(
                    from_symbol_id=owner_symbol.id,
                    file_path=relative_path,
                    reference_name=match.group("type"),
                    reference_kind="instantiates",
                    line=current_line,
                    column_no=match.start(),
                    metadata={"constructor": True},
                )
            )


def _parse_parameters(params: str) -> list[ParsedParameter]:
    parsed: list[ParsedParameter] = []
    if not params.strip():
        return parsed
    for raw in _split_top_level(params, ","):
        text = raw.strip()
        if not text:
            continue
        annotations = [
            Annotation(match.group("name"), match.group("args"))
            for match in ANNOTATION_RE.finditer(text)
        ]
        text = ANNOTATION_RE.sub("", text).strip()
        parts = text.split()
        if len(parts) < 2:
            continue
        parsed.append(
            ParsedParameter(
                type_name=_clean_type(" ".join(parts[:-1])),
                name=parts[-1],
                annotations=annotations,
            )
        )
    return parsed


def _parse_annotation_values(args: str) -> dict[str, object]:
    values: dict[str, object] = {}
    if not args.strip():
        return values
    unnamed_index = 0
    for part in _split_top_level(args, ","):
        text = part.strip()
        if not text:
            continue
        assignment = _split_assignment(text)
        if assignment is None:
            key = "value" if unnamed_index == 0 else f"value_{unnamed_index}"
            values[key] = _clean_annotation_value(text)
            unnamed_index += 1
            continue
        key, raw_value = assignment
        values[key.strip()] = _clean_annotation_value(raw_value.strip())
    return values


def _split_top_level(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape = False
    depth = 0
    for char in value:
        if quote:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "({[":
            depth += 1
        elif char in ")}]" and depth > 0:
            depth -= 1
        if char == delimiter and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _split_assignment(value: str) -> tuple[str, str] | None:
    quote: str | None = None
    escape = False
    depth = 0
    for index, char in enumerate(value):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "({[":
            depth += 1
            continue
        if char in ")}]" and depth > 0:
            depth -= 1
            continue
        if char == "=" and depth == 0:
            return value[:index], value[index + 1 :]
    return None


def _clean_annotation_value(value: str) -> object:
    text = value.strip()
    if text.startswith("{") and text.endswith("}"):
        return [
            _clean_annotation_value(item)
            for item in _split_top_level(text[1:-1], ",")
            if item.strip()
        ]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].replace(f"\\{text[0]}", text[0]).replace("\\\\", "\\")
    return text


def _annotation_primary_value(values: dict[str, object]) -> str | None:
    value = values.get("value")
    if isinstance(value, str) and value:
        return value
    for item in values.values():
        if isinstance(item, str) and item:
            return item
        if isinstance(item, list):
            first = next((value for value in item if isinstance(value, str)), None)
            if first:
                return first
    return None


def _annotation_usage_name(annotation_name: str, primary_value: str | None) -> str:
    if primary_value:
        return f"@{annotation_name} {primary_value}"
    return f"@{annotation_name}"


def _annotation_usage_qualified_name(
    annotation_name: str,
    primary_value: str | None,
    target_qualified_name: str,
    index: int,
) -> str:
    parts = ["annotation_usage", annotation_name]
    if primary_value:
        parts.append(primary_value)
    parts.extend([target_qualified_name, str(index)])
    return ":".join(parts)


def _annotation_signature(annotation: Annotation) -> str:
    if annotation.args:
        return f"@{annotation.name}({annotation.args})"
    return f"@{annotation.name}"


def _metadata_string(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _route_path_from_annotations(annotations: list[Annotation]) -> str:
    route = next((item for item in annotations if is_route_annotation(item.name)), None)
    return _route_path(route) if route else ""


def _route_path(annotation: Annotation) -> str:
    match = re.search(r'"([^"]*)"', annotation.args)
    if match:
        return match.group(1)
    return ""


def _http_method(annotation: Annotation) -> str:
    if annotation.name != "RequestMapping":
        return http_method_for_annotation(annotation.name)
    match = re.search(r"RequestMethod\.([A-Z]+)", annotation.args)
    if match:
        return match.group(1)
    return http_method_for_annotation(annotation.name)


def _join_paths(prefix: str, suffix: str) -> str:
    joined = "/".join(part.strip("/") for part in [prefix, suffix] if part.strip("/"))
    return f"/{joined}" if joined else "/"


def _method_body(lines: list[str], start: int) -> tuple[list[tuple[int, str]], int]:
    body: list[tuple[int, str]] = []
    depth = 0
    seen_open = False
    for index in range(start, len(lines)):
        line = lines[index]
        depth += line.count("{")
        if "{" in line:
            seen_open = True
        depth -= line.count("}")
        if index > start:
            body.append((index + 1, line))
        if seen_open and depth <= 0:
            return body, index
    return body, start


def _method_params(line: str, method_name: str) -> str:
    name_index = line.find(method_name)
    if name_index < 0:
        return ""
    open_index = line.find("(", name_index + len(method_name))
    if open_index < 0:
        return ""
    depth = 0
    quote: str | None = None
    escape = False
    for index in range(open_index, len(line)):
        char = line[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return line[open_index + 1 : index]
    return ""


def _block_end_line(lines: list[str], start: int) -> int:
    return _method_body(lines, start)[1] + 1


def _clean_type(value: str) -> str:
    cleaned = value.replace("final ", "").strip()
    cleaned = re.sub(r"<.*>", "", cleaned).strip()
    return cleaned.split()[-1] if cleaned else cleaned


def _is_project_type(type_name: str) -> bool:
    return bool(
        type_name and type_name[:1].isupper() and not type_name.startswith("java.")
    )


def _is_control_statement(value: str) -> bool:
    return value.strip() in {"if", "for", "while", "switch", "catch"}


def _is_comment_line(stripped: str) -> bool:
    return (
        stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
        or stripped.startswith("*/")
    )


def _first_line_containing(lines: list[str], needle: str) -> int | None:
    for index, line in enumerate(lines, start=1):
        if needle in line:
            return index
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def _is_protected_or_binary(source: str) -> bool:
    prefix = source[:256]
    if "E-SafeNet" in prefix or "LOCK" in prefix[:64]:
        return True
    if "\x00" in prefix:
        return True
    return False
