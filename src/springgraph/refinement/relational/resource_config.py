"""Spring configuration resource extraction."""

from pathlib import Path
from urllib.parse import urlparse

import yaml

from springgraph.hashing import config_symbol_id, resource_symbol_id
from springgraph.refinement.relational.domain import (
    EdgeData,
    ExtractionResult,
    SymbolData,
)


def extract_config_file(
    project_id_value: str,
    path: Path,
    relative_path: str,
    module_name: str | None,
    service_name: str | None,
) -> ExtractionResult:
    """Extract config and resource symbols from Spring config files."""
    content = _read_text(path)
    values = _load_values(path, content)
    symbols: list[SymbolData] = []
    edges: list[EdgeData] = []

    config_key = "spring.config"
    config_id = config_symbol_id(project_id_value, relative_path, config_key)
    config_symbol = SymbolData(
        id=config_id,
        kind="config",
        name=path.name,
        qualified_name=f"config:{relative_path}",
        file_path=relative_path,
        language="config",
        start_line=1,
        end_line=max(1, len(content.splitlines())),
        start_column=0,
        end_column=0,
        metadata={
            "module_name": module_name,
            "service_name": service_name,
            "values": values,
        },
    )
    symbols.append(config_symbol)

    for resource_type, normalized_name in _resources_from_values(values):
        resource_id = resource_symbol_id(
            project_id_value, resource_type, normalized_name
        )
        symbols.append(
            SymbolData(
                id=resource_id,
                kind="resource",
                name=normalized_name,
                qualified_name=f"{resource_type}:{normalized_name}",
                file_path=relative_path,
                language="config",
                start_line=1,
                end_line=max(1, len(content.splitlines())),
                start_column=0,
                end_column=0,
                metadata={
                    "resource_type": resource_type,
                    "normalized_name": normalized_name,
                    "module_name": module_name,
                    "service_name": service_name,
                },
            )
        )
        edges.append(
            EdgeData(
                source_id=config_id,
                target_id=resource_id,
                kind="uses_resource",
                resolved_by="config",
                metadata={"service_name": service_name, "module_name": module_name},
            )
        )

    return ExtractionResult(symbols=symbols, edges=edges, unresolved_refs=[])


def _load_values(path: Path, content: str) -> dict[str, object]:
    if path.suffix == ".properties":
        return _parse_properties(content)
    data = yaml.safe_load(content) or {}
    flattened: dict[str, object] = {}
    _flatten_yaml(data, "", flattened)
    return flattened


def _parse_properties(content: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def _flatten_yaml(data: object, prefix: str, output: dict[str, object]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_yaml(value, next_prefix, output)
    else:
        output[prefix] = data


def _resources_from_values(values: dict[str, object]) -> list[tuple[str, str]]:
    resources: list[tuple[str, str]] = []
    datasource = _str_value(values, "spring.datasource.url")
    if datasource:
        resources.append(("database", _normalize_jdbc_url(datasource)))

    redis_host = _str_value(values, "spring.data.redis.host") or _str_value(
        values, "spring.redis.host"
    )
    if redis_host:
        redis_port = _str_value(values, "spring.data.redis.port") or _str_value(
            values, "spring.redis.port"
        )
        redis_database = _str_value(values, "spring.data.redis.database") or _str_value(
            values, "spring.redis.database"
        )
        resources.append(
            (
                "redis",
                f"redis://{redis_host}:{redis_port or '6379'}/{redis_database or '0'}",
            )
        )

    rabbit_host = _str_value(values, "spring.rabbitmq.host")
    if rabbit_host:
        rabbit_port = _str_value(values, "spring.rabbitmq.port") or "5672"
        resources.append(("rabbitmq", f"rabbitmq://{rabbit_host}:{rabbit_port}"))

    kafka_servers = _str_value(values, "spring.kafka.bootstrap-servers")
    if kafka_servers:
        resources.append(("kafka", f"kafka://{kafka_servers}"))

    return resources


def _str_value(values: dict[str, object], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    return str(value)


def _normalize_jdbc_url(value: str) -> str:
    if value.startswith("jdbc:"):
        value = value.removeprefix("jdbc:")
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return value


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")
