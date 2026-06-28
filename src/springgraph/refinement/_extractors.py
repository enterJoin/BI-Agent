"""Static extractors for semantic Java system refinement."""

import re
from pathlib import Path
from urllib.parse import urlparse

import yaml

from springgraph.hashing import digest
from springgraph.refinement._types import (
    ChunkFact,
    EdgeFact,
    RefinementFacts,
    SymbolFact,
)

TABLE_NAME_RE = re.compile(r'@TableName\(\s*"(?P<table>[^"]+)"\s*\)')
CLASS_RE = re.compile(r"\b(?:class|interface|enum)\s+(?P<name>[A-Za-z_]\w*)")
BASE_MAPPER_RE = re.compile(
    r"interface\s+(?P<dao>[A-Za-z_]\w*)\s+extends\s+BaseMapper<(?P<entity>[A-Za-z_]\w*)>"
)
SERVICE_IMPL_RE = re.compile(
    r"class\s+(?P<name>[A-Za-z_]\w*)\s+extends\s+ServiceImpl<"
    r"(?P<dao>[A-Za-z_]\w*)\s*,\s*(?P<entity>[A-Za-z_]\w*)>"
)
FEIGN_RE = re.compile(r'@FeignClient\(\s*"(?P<service>[^"]+)"\s*\)')
RABBIT_LISTENER_RE = re.compile(
    r'@RabbitListener\([^)]*queues\s*=\s*(?:\{)?\s*"(?P<queue>[^"]+)"'
)
KAFKA_LISTENER_RE = re.compile(
    r'@KafkaListener\([^)]*topics\s*=\s*(?:\{)?\s*"(?P<topic>[^"]+)"'
)
RABBIT_SEND_RE = re.compile(
    r"rabbitTemplate\.convertAndSend\(\s*"
    r'(?P<exchange>"[^"]+"|[A-Z][A-Z0-9_\.]*)\s*,\s*'
    r'(?P<routing>"[^"]+"|[A-Z][A-Z0-9_\.]*)'
)
KAFKA_SEND_RE = re.compile(
    r"\b(?:kafkaService|kafkaTemplate)\.send\s*\(\s*(?P<topic>[^,\n)]+)"
)
ROCKET_SEND_RE = re.compile(
    r"\brocketMQTemplate\."
    r"(?:syncSend|asyncSend|sendOneWay|convertAndSend|sendMessageInTransaction)"
    r"\s*\(\s*(?P<destination>[^,\n)]+)"
)
JAVA_ASSIGNMENT_RE = re.compile(
    r"\b(?:String\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[^;]+);"
)
REDIS_LITERAL_RE = re.compile(
    r"(?:redisTemplate|stringRedisTemplate)\.[A-Za-z0-9_().]+"
    r"\(\s*\"(?P<key>[^\"]+)\""
)
CACHE_RE = re.compile(
    r"@(?P<kind>Cacheable|CachePut|CacheEvict)\((?P<args>[^)]*)\)"
)
PERMISSION_RE = re.compile(
    r"@(?P<kind>PreAuthorize|PostAuthorize|Secured|RolesAllowed|RequiresPermissions)"
    r"\((?P<args>[^)]*)\)"
)
VALUE_RE = re.compile(r'@Value\("\$\{(?P<key>[^}:]+)(?::[^}]*)?}"\)')
METHOD_RE = re.compile(
    r"\b(public|protected|private)\s+[A-Za-z0-9_<>, ?\[\].]+\s+"
    r"(?P<name>[a-zA-Z_]\w*)\s*\("
)
XML_STATEMENT_RE = re.compile(
    r"<(?P<op>select|insert|update|delete)\b(?P<attrs>[^>]*)>"
    r"(?P<body>.*?)</(?P=op)>",
    re.IGNORECASE | re.DOTALL,
)
XML_NAMESPACE_RE = re.compile(r'<mapper\s+namespace="(?P<namespace>[^"]+)"')
XML_ID_RE = re.compile(r'\bid="(?P<id>[^"]+)"')
SQL_TABLE_PATTERNS = {
    "reads_table": [
        re.compile(r"\bfrom\s+`?(?P<table>[A-Za-z_][\w.]*)`?", re.IGNORECASE),
        re.compile(r"\bjoin\s+`?(?P<table>[A-Za-z_][\w.]*)`?", re.IGNORECASE),
    ],
    "insert_table": [
        re.compile(r"\binsert\s+into\s+`?(?P<table>[A-Za-z_][\w.]*)`?", re.IGNORECASE),
    ],
    "update_table": [
        re.compile(r"\bupdate\s+`?(?P<table>[A-Za-z_][\w.]*)`?", re.IGNORECASE),
    ],
    "delete_table": [
        re.compile(r"\bdelete\s+from\s+`?(?P<table>[A-Za-z_][\w.]*)`?", re.IGNORECASE),
    ],
}
CREATE_TABLE_RE = re.compile(
    r"\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?`?(?P<table>[A-Za-z_][\w.]*)`?",
    re.IGNORECASE,
)

PROVIDER_HINTS = {
    "gitee": "Gitee",
    "github.com": "GitHub",
    "weibo": "Weibo",
    "graph.qq.com": "QQ",
    "connect.qq.com": "QQ",
    "wechat": "WeChat",
    "weixin": "WeChat",
}

TEMPLATE_VERSION = "semantic-refinement-v1"
PARSER_VERSION = "semantic-refinement-v1"


def extract_file_facts(
    path: Path,
    relative_path: str,
    language: str,
    module_name: str | None,
    service_name: str | None,
) -> RefinementFacts:
    """Extract graph and chunk facts for a single file."""
    source = _read_text(path)
    if language == "mybatis_xml":
        return _extract_mybatis_xml(source, relative_path, module_name, service_name)
    if language == "java":
        return _extract_java(source, relative_path, module_name, service_name)
    if language == "config":
        return _extract_config(source, path, relative_path, module_name, service_name)
    if language == "html":
        return _extract_html(source, relative_path, module_name, service_name)
    if language == "sql":
        return _extract_sql(source, relative_path, module_name, service_name)
    return RefinementFacts()


def _extract_java(
    source: str,
    relative_path: str,
    module_name: str | None,
    service_name: str | None,
) -> RefinementFacts:
    lines = source.splitlines()
    symbols: list[SymbolFact] = []
    edges: list[EdgeFact] = []
    chunks: list[ChunkFact] = []
    service_key = _service_symbol(symbols, relative_path, module_name, service_name)
    class_name = _first_match(CLASS_RE, source, "name") or Path(relative_path).stem

    table_match = TABLE_NAME_RE.search(source)
    if table_match:
        table = table_match.group("table")
        line = _line_for_offset(source, table_match.start())
        entity_key = _symbol_key("data_contract", relative_path, class_name, line)
        table_key = _table_symbol(symbols, relative_path, table, line, service_name)
        symbols.append(
            SymbolFact(
                key=entity_key,
                kind="data_contract",
                name=class_name,
                qualified_name=f"data_contract:{class_name}",
                file_path=relative_path,
                language="java",
                start_line=line,
                end_line=line,
                metadata={
                    "module_name": module_name,
                    "service_name": service_name,
                    "contract_kind": "mybatis_plus_entity",
                    "table": table,
                },
            )
        )
        edges.append(
            EdgeFact(
                source_key=entity_key,
                target_key=table_key,
                kind="defines_contract",
                line=line,
                metadata={"source": "@TableName"},
            )
        )
        columns = _java_fields(lines)
        for column in columns:
            column_key = _column_symbol(
                symbols,
                relative_path,
                table,
                _camel_to_snake(column[1]),
                column[0],
                service_name,
                java_field=column[1],
                java_type=column[2],
            )
            edges.append(
                EdgeFact(
                    source_key=entity_key,
                    target_key=column_key,
                    kind="maps_field",
                    line=column[0],
                    metadata={"java_field": column[1], "java_type": column[2]},
                )
            )
        chunks.append(
            _chunk(
                "db_table_summary",
                relative_path,
                table_key,
                f"Table {table}",
                "\n".join(
                    [
                        "Chunk type: db_table_summary",
                        f"Service: {service_name or ''}",
                        f"Module: {module_name or ''}",
                        f"Table: {table}",
                        f"Entity: {class_name}",
                        f"Source: {relative_path}:{line}",
                        "Columns inferred from Java fields:",
                        ", ".join(_camel_to_snake(item[1]) for item in columns),
                    ]
                ),
                "java",
                line,
                max([line, *[item[0] for item in columns]]),
                {"table": table, "entity": class_name, "confidence": "high"},
            )
        )

    mapper_match = BASE_MAPPER_RE.search(source)
    if mapper_match:
        dao = mapper_match.group("dao")
        entity = mapper_match.group("entity")
        line = _line_for_offset(source, mapper_match.start())
        mapper_key = _symbol_key("mapper", relative_path, dao, line)
        symbols.append(
            SymbolFact(
                key=mapper_key,
                kind="mapper",
                name=dao,
                qualified_name=f"mapper:{dao}",
                file_path=relative_path,
                language="java",
                start_line=line,
                end_line=line,
                metadata={
                    "module_name": module_name,
                    "service_name": service_name,
                    "entity": entity,
                    "mapper_kind": "mybatis_plus_base_mapper",
                },
            )
        )
        inferred_table = _table_from_entity_name(entity)
        table_key = _table_symbol(
            symbols, relative_path, inferred_table, line, service_name
        )
        for edge_kind in ("reads_table", "writes_table"):
            edges.append(
                EdgeFact(
                    source_key=mapper_key,
                    target_key=table_key,
                    kind=edge_kind,
                    line=line,
                    confidence=0.72,
                    metadata={
                        "source": "BaseMapper",
                        "entity": entity,
                        "confidence_reason": "table inferred from entity name; "
                        "@TableName provides stronger evidence when present",
                    },
                )
            )
            edges.append(
                EdgeFact(
                    source_key=service_key,
                    target_key=table_key,
                    kind=edge_kind,
                    line=line,
                    confidence=0.72,
                    metadata={"source": "BaseMapper", "mapper": dao, "entity": entity},
                )
            )
        chunks.append(
            _chunk(
                "mapper_method",
                relative_path,
                mapper_key,
                f"Mapper {dao}",
                "\n".join(
                    [
                        "Chunk type: mapper_method",
                        f"Service: {service_name or ''}",
                        f"Mapper: {dao}",
                        f"Entity: {entity}",
                        f"Tables inferred: {inferred_table}",
                        "Operation: MyBatis-Plus BaseMapper CRUD",
                        f"Source: {relative_path}:{line}",
                    ]
                ),
                "java",
                line,
                line,
                {"mapper": dao, "entity": entity, "table": inferred_table},
            )
        )

    service_impl_match = SERVICE_IMPL_RE.search(source)
    if service_impl_match:
        line = _line_for_offset(source, service_impl_match.start())
        entity = service_impl_match.group("entity")
        table = _table_from_entity_name(entity)
        table_key = _table_symbol(symbols, relative_path, table, line, service_name)
        for edge_kind in ("reads_table", "writes_table"):
            edges.append(
                EdgeFact(
                    source_key=service_key,
                    target_key=table_key,
                    kind=edge_kind,
                    line=line,
                    confidence=0.72,
                    metadata={
                        "source": "ServiceImpl",
                        "entity": entity,
                        "service_impl": service_impl_match.group("name"),
                    },
                )
            )

    for match in FEIGN_RE.finditer(source):
        line = _line_for_offset(source, match.start())
        target_service = match.group("service")
        feign_key = _symbol_key("remote_service", relative_path, target_service, line)
        symbols.append(
            SymbolFact(
                key=feign_key,
                kind="remote_service",
                name=target_service,
                qualified_name=f"remote_service:{target_service}",
                file_path=relative_path,
                language="java",
                start_line=line,
                end_line=line,
                metadata={"module_name": module_name, "service_name": service_name},
            )
        )
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=feign_key,
                kind="calls_remote_service",
                line=line,
                metadata={"source": "@FeignClient"},
            )
        )

    _extract_java_messaging(source, relative_path, service_name, symbols, edges, chunks)
    _extract_java_cache(source, relative_path, service_name, symbols, edges, chunks)
    _extract_java_security(source, relative_path, service_name, symbols, edges, chunks)
    _extract_oauth_mentions(
        source, relative_path, "java", service_name, symbols, edges, chunks
    )
    _extract_value_config_refs(
        source, relative_path, service_name, symbols, edges, chunks
    )
    return RefinementFacts(symbols=symbols, edges=edges, chunks=chunks)


def _extract_mybatis_xml(
    source: str,
    relative_path: str,
    module_name: str | None,
    service_name: str | None,
) -> RefinementFacts:
    symbols: list[SymbolFact] = []
    edges: list[EdgeFact] = []
    chunks: list[ChunkFact] = []
    service_key = _service_symbol(symbols, relative_path, module_name, service_name)
    namespace = _first_match(XML_NAMESPACE_RE, source, "namespace") or Path(
        relative_path
    ).stem
    mapper_line = _line_for_offset(source, source.find("<mapper"))
    mapper_key = _symbol_key("mapper", relative_path, namespace, mapper_line)
    symbols.append(
        SymbolFact(
            key=mapper_key,
            kind="mapper",
            name=namespace.rsplit(".", maxsplit=1)[-1],
            qualified_name=f"mapper:{namespace}",
            file_path=relative_path,
            language="xml",
            start_line=mapper_line,
            end_line=mapper_line,
            metadata={"module_name": module_name, "service_name": service_name},
        )
    )
    for match in XML_STATEMENT_RE.finditer(source):
        op = match.group("op").lower()
        attrs = match.group("attrs")
        statement_id = _first_match(XML_ID_RE, attrs, "id") or f"{op}:{match.start()}"
        line = _line_for_offset(source, match.start())
        raw_xml = match.group(0).strip()
        sql_text = _strip_xml_tags(raw_xml)
        statement_key = _symbol_key(
            "sql_statement", relative_path, f"{namespace}#{statement_id}", line
        )
        symbols.append(
            SymbolFact(
                key=statement_key,
                kind="sql_statement",
                name=statement_id,
                qualified_name=f"sql_statement:{namespace}#{statement_id}",
                file_path=relative_path,
                language="xml",
                start_line=line,
                end_line=_line_for_offset(source, match.end()),
                metadata={
                    "module_name": module_name,
                    "service_name": service_name,
                    "operation": op,
                    "namespace": namespace,
                    "dynamic_sql": _has_dynamic_sql(raw_xml),
                },
            )
        )
        edges.append(
            EdgeFact(
                source_key=mapper_key,
                target_key=statement_key,
                kind="maps_to_sql",
                line=line,
            )
        )
        for edge_kind, table in _tables_from_sql(sql_text, op):
            table_key = _table_symbol(symbols, relative_path, table, line, service_name)
            edges.append(
                EdgeFact(
                    source_key=statement_key,
                    target_key=table_key,
                    kind=edge_kind,
                    line=line,
                    confidence=0.92,
                    metadata={"operation": op, "source": "mybatis_xml"},
                )
            )
            edges.append(
                EdgeFact(
                    source_key=service_key,
                    target_key=table_key,
                    kind=edge_kind,
                    line=line,
                    confidence=0.9,
                    metadata={
                        "operation": op,
                        "mapper": namespace,
                        "statement": statement_id,
                    },
                )
            )
        chunks.append(
            _chunk(
                "mybatis_sql",
                relative_path,
                statement_key,
                f"MyBatis SQL {namespace}#{statement_id}",
                "\n".join(
                    [
                        "Chunk type: mybatis_sql",
                        f"Service: {service_name or ''}",
                        f"Mapper: {namespace}",
                        f"Statement: {namespace}#{statement_id}",
                        f"Operation: {op}",
                        (
                            f"Source: {relative_path}:{line}-"
                            f"{_line_for_offset(source, match.end())}"
                        ),
                        f"Dynamic SQL: {_has_dynamic_sql(raw_xml)}",
                        "SQL/XML:",
                        raw_xml,
                    ]
                ),
                "xml",
                line,
                _line_for_offset(source, match.end()),
                {"operation": op, "namespace": namespace, "statement": statement_id},
            )
        )
    return RefinementFacts(symbols=symbols, edges=edges, chunks=chunks)


def _extract_config(
    source: str,
    path: Path,
    relative_path: str,
    module_name: str | None,
    service_name: str | None,
) -> RefinementFacts:
    symbols: list[SymbolFact] = []
    edges: list[EdgeFact] = []
    chunks: list[ChunkFact] = []
    service_key = _service_symbol(symbols, relative_path, module_name, service_name)
    values = _load_config_values(path, source)
    redacted = _redact_values(values)
    for key, value in values.items():
        lowered = key.lower()
        if "oauth" in lowered or "client.id" in lowered or "redirect_uri" in lowered:
            provider = _provider_from_text(f"{key} {value}") or "OAuth2"
            provider_key = _provider_symbol(
                symbols, relative_path, provider, _line_containing(source, str(key))
            )
            edges.append(
                EdgeFact(
                    source_key=service_key,
                    target_key=provider_key,
                    kind="uses_oauth_provider",
                    line=_line_containing(source, str(key)),
                    metadata={"config_key": key},
                )
            )
    chunk_key = _symbol_key("config", relative_path, "spring_config", 1)
    symbols.append(
        SymbolFact(
            key=chunk_key,
            kind="config",
            name=Path(relative_path).name,
            qualified_name=f"config:{relative_path}",
            file_path=relative_path,
            language="config",
            start_line=1,
            end_line=max(1, len(source.splitlines())),
            metadata={"module_name": module_name, "service_name": service_name},
        )
    )
    chunks.append(
        _chunk(
            "spring_config",
            relative_path,
            chunk_key,
            f"Spring config {relative_path}",
            "\n".join(
                [
                    "Chunk type: spring_config",
                    f"Service: {service_name or ''}",
                    f"Module: {module_name or ''}",
                    f"Config file: {relative_path}",
                    "Properties:",
                    "\n".join(
                        f"{key}={value}" for key, value in sorted(redacted.items())
                    ),
                ]
            ),
            "config",
            1,
            max(1, len(source.splitlines())),
            {"redacted": True, "keys": sorted(values)},
        )
    )
    return RefinementFacts(symbols=symbols, edges=edges, chunks=chunks)


def _extract_html(
    source: str,
    relative_path: str,
    module_name: str | None,
    service_name: str | None,
) -> RefinementFacts:
    symbols: list[SymbolFact] = []
    edges: list[EdgeFact] = []
    chunks: list[ChunkFact] = []
    _service_symbol(symbols, relative_path, module_name, service_name)
    _extract_oauth_mentions(
        source, relative_path, "html", service_name, symbols, edges, chunks
    )
    return RefinementFacts(symbols=symbols, edges=edges, chunks=chunks)


def _extract_sql(
    source: str,
    relative_path: str,
    module_name: str | None,
    service_name: str | None,
) -> RefinementFacts:
    symbols: list[SymbolFact] = []
    chunks: list[ChunkFact] = []
    _service_symbol(symbols, relative_path, module_name, service_name)
    for match in CREATE_TABLE_RE.finditer(source):
        table = match.group("table")
        line = _line_for_offset(source, match.start())
        table_key = _table_symbol(symbols, relative_path, table, line, service_name)
        chunks.append(
            _chunk(
                "db_table_summary",
                relative_path,
                table_key,
                f"DDL table {table}",
                "\n".join(
                    [
                        "Chunk type: db_table_summary",
                        f"Service: {service_name or ''}",
                        f"Table: {table}",
                        f"Source: {relative_path}:{line}",
                    ]
                ),
                "sql",
                line,
                line,
                {"table": table, "source": "ddl"},
            )
        )
    return RefinementFacts(symbols=symbols, chunks=chunks)


def _extract_java_messaging(
    source: str,
    relative_path: str,
    service_name: str | None,
    symbols: list[SymbolFact],
    edges: list[EdgeFact],
    chunks: list[ChunkFact],
) -> None:
    service_key = _service_key(service_name)
    assignments = _java_assignments(source)
    for match in RABBIT_LISTENER_RE.finditer(source):
        queue = match.group("queue")
        line = _line_for_offset(source, match.start())
        queue_key = _mq_symbol(symbols, relative_path, "mq_queue", queue, line)
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=queue_key,
                kind="consumes",
                line=line,
                metadata={"source": "@RabbitListener"},
            )
        )
        chunks.append(_mq_chunk(relative_path, queue_key, "consumes", queue, line))
    for match in KAFKA_LISTENER_RE.finditer(source):
        topic = match.group("topic")
        line = _line_for_offset(source, match.start())
        topic_key = _mq_symbol(symbols, relative_path, "mq_topic", topic, line)
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=topic_key,
                kind="consumes",
                line=line,
                metadata={"source": "@KafkaListener"},
            )
        )
        chunks.append(_mq_chunk(relative_path, topic_key, "consumes", topic, line))
    for match in RABBIT_SEND_RE.finditer(source):
        exchange = _clean_java_value(match.group("exchange"))
        routing = _clean_java_value(match.group("routing"))
        line = _line_for_offset(source, match.start())
        target = f"{exchange}:{routing}"
        mq_key = _mq_symbol(symbols, relative_path, "mq_exchange", target, line)
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=mq_key,
                kind="publishes",
                line=line,
                metadata={"exchange": exchange, "routing_key": routing},
            )
        )
        chunks.append(_mq_chunk(relative_path, mq_key, "publishes", target, line))
    for match in KAFKA_SEND_RE.finditer(source):
        line = _line_for_offset(source, match.start())
        for topic in _resolve_java_values(match.group("topic"), assignments):
            _add_topic_publish(
                symbols=symbols,
                edges=edges,
                chunks=chunks,
                source_key=service_key,
                relative_path=relative_path,
                topic=topic,
                tag=None,
                line=line,
                source="kafka_send",
            )
    for match in ROCKET_SEND_RE.finditer(source):
        line = _line_for_offset(source, match.start())
        for destination in _resolve_java_values(
            match.group("destination"),
            assignments,
        ):
            topic, tag = _topic_and_tag(destination)
            _add_topic_publish(
                symbols=symbols,
                edges=edges,
                chunks=chunks,
                source_key=service_key,
                relative_path=relative_path,
                topic=topic,
                tag=tag,
                line=line,
                source="rocketmq_send",
            )


def _extract_java_cache(
    source: str,
    relative_path: str,
    service_name: str | None,
    symbols: list[SymbolFact],
    edges: list[EdgeFact],
    chunks: list[ChunkFact],
) -> None:
    service_key = _service_key(service_name)
    for match in REDIS_LITERAL_RE.finditer(source):
        key = match.group("key")
        line = _line_for_offset(source, match.start())
        cache_key = _cache_symbol(symbols, relative_path, key, line)
        edge_kind = "writes_cache" if ".set(" in match.group(0) else "reads_cache"
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=cache_key,
                kind=edge_kind,
                line=line,
                metadata={"source": "RedisTemplate"},
            )
        )
        chunks.append(_cache_chunk(relative_path, cache_key, edge_kind, key, line))
    for match in CACHE_RE.finditer(source):
        line = _line_for_offset(source, match.start())
        cache_name = _first_quoted(match.group("args")) or match.group("kind")
        cache_key = _cache_symbol(symbols, relative_path, cache_name, line)
        edge_kind = {
            "Cacheable": "reads_cache",
            "CachePut": "writes_cache",
            "CacheEvict": "evicts_cache",
        }[match.group("kind")]
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=cache_key,
                kind=edge_kind,
                line=line,
                metadata={"source": f"@{match.group('kind')}"},
            )
        )
        chunks.append(
            _cache_chunk(relative_path, cache_key, edge_kind, cache_name, line)
        )


def _extract_java_security(
    source: str,
    relative_path: str,
    service_name: str | None,
    symbols: list[SymbolFact],
    edges: list[EdgeFact],
    chunks: list[ChunkFact],
) -> None:
    service_key = _service_key(service_name)
    for match in PERMISSION_RE.finditer(source):
        line = _line_for_offset(source, match.start())
        expression = match.group("args").strip()
        permission_key = _symbol_key("permission_rule", relative_path, expression, line)
        symbols.append(
            SymbolFact(
                key=permission_key,
                kind="permission_rule",
                name=expression[:120],
                qualified_name=f"permission_rule:{expression}",
                file_path=relative_path,
                language="java",
                start_line=line,
                end_line=line,
                metadata={
                    "service_name": service_name,
                    "rule_kind": match.group("kind"),
                },
            )
        )
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=permission_key,
                kind="requires_permission",
                line=line,
                metadata={"source": f"@{match.group('kind')}"},
            )
        )
        chunks.append(
            _chunk(
                "permission_rule",
                relative_path,
                permission_key,
                f"Permission {expression[:80]}",
                "\n".join(
                    [
                        "Chunk type: permission_rule",
                        f"Service: {service_name or ''}",
                        f"Permission: {expression}",
                        f"Source: {relative_path}:{line}",
                    ]
                ),
                "java",
                line,
                line,
                {"permission": expression, "rule_kind": match.group("kind")},
            )
        )


def _extract_oauth_mentions(
    source: str,
    relative_path: str,
    language: str,
    service_name: str | None,
    symbols: list[SymbolFact],
    edges: list[EdgeFact],
    chunks: list[ChunkFact],
) -> None:
    service_key = _service_key(service_name)
    seen: set[tuple[str, int]] = set()
    for line_number, line in enumerate(source.splitlines(), start=1):
        provider = _provider_from_text(line)
        if provider is None:
            continue
        if ("oauth" not in line.lower() and provider == "QQ" and "href=\"#\"" in line):
            continue
        marker = (provider, line_number)
        if marker in seen:
            continue
        seen.add(marker)
        provider_key = _provider_symbol(symbols, relative_path, provider, line_number)
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=provider_key,
                kind="uses_oauth_provider",
                line=line_number,
                metadata={"evidence": line.strip()[:300]},
            )
        )
        chunks.append(
            _chunk(
                "third_party_login",
                relative_path,
                provider_key,
                f"Third-party login {provider}",
                "\n".join(
                    [
                        "Chunk type: third_party_login",
                        f"Service: {service_name or ''}",
                        f"Provider: {provider}",
                        f"Source: {relative_path}:{line_number}",
                        f"Evidence: {_redact_text(line.strip())}",
                    ]
                ),
                language,
                line_number,
                line_number,
                {"provider": provider},
            )
        )


def _extract_value_config_refs(
    source: str,
    relative_path: str,
    service_name: str | None,
    symbols: list[SymbolFact],
    edges: list[EdgeFact],
    chunks: list[ChunkFact],
) -> None:
    service_key = _service_key(service_name)
    for match in VALUE_RE.finditer(source):
        config_key = match.group("key")
        if "oauth" not in config_key.lower():
            continue
        line = _line_for_offset(source, match.start())
        provider = _provider_from_text(config_key) or "OAuth2"
        provider_key = _provider_symbol(symbols, relative_path, provider, line)
        edges.append(
            EdgeFact(
                source_key=service_key,
                target_key=provider_key,
                kind="uses_oauth_provider",
                line=line,
                metadata={"config_key": config_key, "source": "@Value"},
            )
        )
        chunks.append(
            _chunk(
                "third_party_login",
                relative_path,
                provider_key,
                f"OAuth config {config_key}",
                "\n".join(
                    [
                        "Chunk type: third_party_login",
                        f"Service: {service_name or ''}",
                        f"Provider: {provider}",
                        f"Config key: {config_key}",
                        f"Source: {relative_path}:{line}",
                    ]
                ),
                "java",
                line,
                line,
                {"provider": provider, "config_key": config_key},
            )
        )


def _service_symbol(
    symbols: list[SymbolFact],
    relative_path: str,
    module_name: str | None,
    service_name: str | None,
) -> str:
    service = service_name or module_name or "unknown-service"
    key = _service_key(service)
    symbols.append(
        SymbolFact(
            key=key,
            kind="service",
            name=service,
            qualified_name=f"service:{service}",
            file_path=relative_path,
            language="service",
            start_line=1,
            end_line=1,
            metadata={"module_name": module_name, "service_name": service},
        )
    )
    return key


def _service_key(service_name: str | None) -> str:
    return f"service:{service_name or 'unknown-service'}"


def _table_symbol(
    symbols: list[SymbolFact],
    relative_path: str,
    table: str,
    line: int,
    service_name: str | None,
) -> str:
    key = f"db_table:{table}"
    symbols.append(
        SymbolFact(
            key=key,
            kind="db_table",
            name=table,
            qualified_name=key,
            file_path=relative_path,
            language="sql",
            start_line=max(1, line),
            end_line=max(1, line),
            metadata={"table": table, "service_name": service_name},
        )
    )
    return key


def _column_symbol(
    symbols: list[SymbolFact],
    relative_path: str,
    table: str,
    column: str,
    line: int,
    service_name: str | None,
    **metadata: object,
) -> str:
    key = f"db_column:{table}.{column}"
    symbols.append(
        SymbolFact(
            key=key,
            kind="db_column",
            name=column,
            qualified_name=key,
            file_path=relative_path,
            language="sql",
            start_line=max(1, line),
            end_line=max(1, line),
            metadata={"table": table, "column": column, "service_name": service_name}
            | metadata,
        )
    )
    return key


def _mq_symbol(
    symbols: list[SymbolFact],
    relative_path: str,
    kind: str,
    name: str,
    line: int,
    **metadata: object,
) -> str:
    key = f"{kind}:{name}"
    symbols.append(
        SymbolFact(
            key=key,
            kind=kind,
            name=name,
            qualified_name=key,
            file_path=relative_path,
            language="java",
            start_line=line,
            end_line=line,
            metadata={"name": name} | metadata,
        )
    )
    return key


def _add_topic_publish(
    *,
    symbols: list[SymbolFact],
    edges: list[EdgeFact],
    chunks: list[ChunkFact],
    source_key: str,
    relative_path: str,
    topic: str,
    tag: str | None,
    line: int,
    source: str,
) -> None:
    if not topic:
        return
    topic_key = _mq_symbol(
        symbols,
        relative_path,
        "mq_topic",
        topic,
        line,
        topic=topic,
        tag=tag,
        source=source,
    )
    edges.append(
        EdgeFact(
            source_key=source_key,
            target_key=topic_key,
            kind="publishes",
            line=line,
            metadata={"topic": topic, "tag": tag, "source": source},
        )
    )
    chunks.append(
        _mq_chunk(
            relative_path,
            topic_key,
            "publishes",
            _topic_label(topic, tag),
            line,
        )
    )
    if tag:
        tag_key = _mq_symbol(
            symbols,
            relative_path,
            "mq_tag",
            f"{topic}:{tag}",
            line,
            topic=topic,
            tag=tag,
            source=source,
        )
        edges.append(
            EdgeFact(
                source_key=topic_key,
                target_key=tag_key,
                kind="publishes",
                line=line,
                metadata={"topic": topic, "tag": tag, "source": source},
            )
        )
        chunks.append(
            _mq_chunk(
                relative_path,
                tag_key,
                "publishes",
                _topic_label(topic, tag),
                line,
            )
        )


def _java_assignments(source: str) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    for match in JAVA_ASSIGNMENT_RE.finditer(source):
        name = match.group("name")
        value = _clean_java_value(match.group("value"))
        if not name or not value:
            continue
        assignments.setdefault(name, []).append(value)
    return assignments


def _resolve_java_values(
    expression: str,
    assignments: dict[str, list[str]],
) -> list[str]:
    value = _clean_java_value(expression)
    if value in assignments:
        return _dedupe_strings(assignments[value])
    return [value] if value else []


def _topic_and_tag(destination: str) -> tuple[str, str | None]:
    value = _clean_java_value(destination)
    if ":" in value and "://" not in value:
        topic, tag = value.split(":", maxsplit=1)
        return (_clean_java_value(topic), _clean_java_value(tag) or None)
    if '":"' in value or '":" ' in value or '"+' in value:
        parts = [
            _clean_java_value(part)
            for part in re.split(r'\+\s*":\"\s*\+|\+\s*":"\s*\+', value)
        ]
        if len(parts) == 2 and parts[0] and parts[1]:
            return (parts[0], parts[1])
    return (value, None)


def _topic_label(topic: str, tag: str | None) -> str:
    return f"{topic}:{tag}" if tag else topic


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _cache_symbol(
    symbols: list[SymbolFact], relative_path: str, name: str, line: int
) -> str:
    key = f"cache_key:{name}"
    symbols.append(
        SymbolFact(
            key=key,
            kind="cache_key",
            name=name,
            qualified_name=key,
            file_path=relative_path,
            language="java",
            start_line=line,
            end_line=line,
            metadata={"cache_key": name},
        )
    )
    return key


def _provider_symbol(
    symbols: list[SymbolFact], relative_path: str, provider: str, line: int
) -> str:
    key = f"oauth_provider:{provider}"
    symbols.append(
        SymbolFact(
            key=key,
            kind="oauth_provider",
            name=provider,
            qualified_name=key,
            file_path=relative_path,
            language="java",
            start_line=max(1, line),
            end_line=max(1, line),
            metadata={"provider": provider},
        )
    )
    return key


def _symbol_key(kind: str, relative_path: str, name: str, line: int) -> str:
    return f"{kind}:{digest(f'{relative_path}:{name}:{line}', 24)}"


def _chunk(
    chunk_type: str,
    file_path: str,
    symbol_key: str | None,
    title: str,
    content: str,
    language: str,
    start_line: int | None,
    end_line: int | None,
    metadata: dict[str, object],
) -> ChunkFact:
    raw_key = f"{chunk_type}:{file_path}:{title}:{start_line}:{content}"
    key = f"chunk:{digest(raw_key, 32)}"
    return ChunkFact(
        key=key,
        file_path=file_path,
        symbol_key=symbol_key,
        chunk_type=chunk_type,
        title=title,
        content=_redact_text(content),
        language=language,
        start_line=start_line,
        end_line=end_line,
        metadata=metadata
        | {"parser_version": PARSER_VERSION, "template_version": TEMPLATE_VERSION},
    )


def _mq_chunk(
    relative_path: str, symbol_key: str, action: str, target: str, line: int
) -> ChunkFact:
    return _chunk(
        "mq_flow",
        relative_path,
        symbol_key,
        f"MQ {action} {target}",
        "\n".join(
            [
                "Chunk type: mq_flow",
                f"Action: {action}",
                f"Target: {target}",
                f"Source: {relative_path}:{line}",
            ]
        ),
        "java",
        line,
        line,
        {"action": action, "target": target},
    )


def _cache_chunk(
    relative_path: str, symbol_key: str, action: str, target: str, line: int
) -> ChunkFact:
    return _chunk(
        "redis_cache_flow",
        relative_path,
        symbol_key,
        f"Cache {action} {target}",
        "\n".join(
            [
                "Chunk type: redis_cache_flow",
                f"Action: {action}",
                f"Cache key: {target}",
                f"Source: {relative_path}:{line}",
            ]
        ),
        "java",
        line,
        line,
        {"action": action, "cache_key": target},
    )


def _java_fields(lines: list[str]) -> list[tuple[int, str, str]]:
    fields: list[tuple[int, str, str]] = []
    field_re = re.compile(
        r"^\s*private\s+(?P<type>[A-Za-z_][\w<>?, ]*)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*(?:=.*)?;"
    )
    for line_number, line in enumerate(lines, start=1):
        match = field_re.match(line)
        if match:
            fields.append((line_number, match.group("name"), match.group("type")))
    return fields


def _tables_from_sql(sql_text: str, operation: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if operation == "select":
        patterns = SQL_TABLE_PATTERNS["reads_table"]
    elif operation == "insert":
        patterns = SQL_TABLE_PATTERNS["insert_table"]
    elif operation == "update":
        patterns = SQL_TABLE_PATTERNS["update_table"]
    elif operation == "delete":
        patterns = SQL_TABLE_PATTERNS["delete_table"]
    else:
        patterns = []
    for pattern in patterns:
        for match in pattern.finditer(sql_text):
            table = _clean_table_name(match.group("table"))
            if table and (operation != "select" or table.lower() != "select"):
                edge_kind = "reads_table" if operation == "select" else "writes_table"
                item = (edge_kind, table)
                if item not in found:
                    found.append(item)
    return found


def _strip_xml_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def _has_dynamic_sql(value: str) -> bool:
    return any(tag in value for tag in ("<if", "<foreach", "<choose", "<where", "<set"))


def _table_from_entity_name(entity: str) -> str:
    base = entity.removesuffix("Entity")
    return _camel_to_snake(base)


def _camel_to_snake(value: str) -> str:
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _clean_table_name(value: str) -> str:
    return value.strip("`").split(".", maxsplit=1)[-1]


def _clean_java_value(value: str) -> str:
    return value.strip().strip('"')


def _provider_from_text(value: str) -> str | None:
    lowered = value.lower()
    if "@email" in lowered or re.search(r"\b[\w.-]+@qq\.com\b", lowered):
        return None
    if not _has_login_provider_context(lowered):
        return None
    for hint, provider in PROVIDER_HINTS.items():
        if hint in lowered:
            return provider
    try:
        parsed = urlparse(value)
        host = parsed.netloc.lower()
    except ValueError:
        host = ""
    for hint, provider in PROVIDER_HINTS.items():
        if hint in host:
            return provider
    return None


def _has_login_provider_context(value: str) -> bool:
    return any(
        marker in value
        for marker in (
            "oauth",
            "authorize",
            "access_token",
            "response_type=code",
            "redirect_uri",
            "social",
            "third-party",
            "third party",
            "第三方登录",
            "/api/v5/user",
        )
    )


def _first_quoted(value: str) -> str | None:
    match = re.search(r'"([^"]+)"', value)
    return match.group(1) if match else None


def _first_match(pattern: re.Pattern[str], value: str, group: str) -> str | None:
    match = pattern.search(value)
    return match.group(group) if match else None


def _line_for_offset(source: str, offset: int) -> int:
    if offset < 0:
        return 1
    return source.count("\n", 0, offset) + 1


def _line_containing(source: str, needle: str) -> int:
    for index, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return index
    return 1


def _load_config_values(path: Path, source: str) -> dict[str, object]:
    if path.suffix == ".properties":
        return _parse_properties(source)
    data = yaml.safe_load(source) or {}
    flattened: dict[str, object] = {}
    _flatten_yaml(data, "", flattened)
    return flattened


def _parse_properties(source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in source.splitlines():
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


def _redact_values(values: dict[str, object]) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, value in values.items():
        if _is_secret_key(key):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def _redact_text(value: str) -> str:
    patterns = [
        re.compile(r"(?i)(password|passwd|pwd|secret|token|access[_-]?key)=([^&\s]+)"),
        re.compile(r"(?i)(client_secret:\s*)(\S+)"),
        re.compile(r"(?i)(client\.secret[=:]\s*)(\S+)"),
    ]
    redacted = value
    for pattern in patterns:
        redacted = pattern.sub(r"\1=<redacted>", redacted)
    return redacted


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(
        marker in lowered
        for marker in ("password", "passwd", "pwd", "secret", "token", "access-key")
    )


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
