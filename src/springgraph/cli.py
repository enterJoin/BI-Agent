"""Command-line entry point."""

from pathlib import Path

import typer

from springgraph.config import get_settings
from springgraph.db import session_scope
from springgraph.logging_config import configure_logging
from springgraph.refinement import refine_project
from springgraph.refinement.relational import queries

app = typer.Typer(help="Index Java Spring Boot projects into PostgreSQL.")


@app.callback()
def main() -> None:
    """Configure CLI process."""
    configure_logging(get_settings().log_level)


@app.command()
def index(path: Path) -> None:
    """Index a Spring Boot project or workspace directory."""
    result = refine_project(path)
    typer.echo(f"Index run: {result.index_run_id}")
    typer.echo(f"Embedding job: {result.embedding_job_id}")
    typer.echo(f"Files seen: {result.files_seen}")
    typer.echo(f"Symbols upserted: {result.symbols_upserted}")
    typer.echo(f"Edges upserted: {result.edges_upserted}")
    typer.echo(f"Chunks upserted: {result.chunks_upserted}")
    typer.echo(f"Embeddings upserted: {result.embeddings_upserted}")


@app.command()
def refine(path: Path) -> None:
    """Refine a Java workspace into graph facts, chunks, and vectors."""
    result = refine_project(path)
    typer.echo(f"Index run: {result.index_run_id}")
    typer.echo(f"Embedding job: {result.embedding_job_id}")
    typer.echo(f"Files seen: {result.files_seen}")
    typer.echo(f"Symbols upserted: {result.symbols_upserted}")
    typer.echo(f"Edges upserted: {result.edges_upserted}")
    typer.echo(f"Chunks upserted: {result.chunks_upserted}")
    typer.echo(f"Embeddings upserted: {result.embeddings_upserted}")
    if result.errors:
        typer.echo("Errors:")
        for error in result.errors:
            typer.echo(f"- {error}")


@app.command()
def status(path: Path) -> None:
    """Print status counters for a project path."""
    with session_scope() as session:
        result = queries.status(session, path)
    typer.echo(f"Project: {result['project']}")
    typer.echo(f"Files: {result['files']}")
    typer.echo(f"Symbols: {result['symbols']}")
    typer.echo(f"Edges: {result['edges']}")
    typer.echo(f"Unresolved refs: {result['unresolved_refs']}")


@app.command()
def search(path: Path, query: str) -> None:
    """Search indexed symbols."""
    with session_scope() as session:
        symbols = queries.search_symbols(session, path, query)
        file_map = _file_map(session, path)
    for symbol in symbols:
        file_path = file_map.get(symbol.file_id, symbol.file_id)
        typer.echo(
            f"{symbol.kind} {symbol.name} {symbol.qualified_name} "
            f"{file_path}:{symbol.start_line}"
        )


@app.command()
def routes(path: Path) -> None:
    """Print indexed HTTP routes."""
    with session_scope() as session:
        symbols = queries.routes(session, path)
    for symbol in symbols:
        method = symbol.meta.get("http_method")
        route_path = symbol.meta.get("path")
        handler = symbol.meta.get("handler")
        typer.echo(f"{method} {route_path} -> {handler}")


@app.command()
def resources(path: Path) -> None:
    """Print shared infrastructure resources."""
    with session_scope() as session:
        rows = queries.resources(session, path)
    for resource_type, normalized_name, services in rows:
        typer.echo(f"{resource_type} {normalized_name} <- {', '.join(services)}")


@app.command()
def callers(path: Path, symbol: str) -> None:
    """Print callers for a symbol query."""
    _print_symbols(path, queries.callers, symbol)


@app.command()
def callees(path: Path, symbol: str) -> None:
    """Print callees for a symbol query."""
    _print_symbols(path, queries.callees, symbol)


@app.command()
def impact(path: Path, symbol: str, depth: int = 2) -> None:
    """Print upstream impact for a symbol query."""
    with session_scope() as session:
        symbols = queries.impact(session, path, symbol, depth)
        file_map = _file_map(session, path)
    for item in symbols:
        location = f"{file_map.get(item.file_id, item.file_id)}:{item.start_line}"
        typer.echo(
            f"{item.qualified_name} {location}"
        )


@app.command()
def files(path: Path) -> None:
    """Print indexed files."""
    with session_scope() as session:
        rows = queries.files(session, path)
    for row in rows:
        typer.echo(f"{row.path} module={row.module_name} service={row.service_name}")


def _print_symbols(path: Path, query_fn: object, symbol: str) -> None:
    with session_scope() as session:
        result = query_fn(session, path, symbol)  # type: ignore[operator]
        file_map = _file_map(session, path)
    for item in result:
        location = f"{file_map.get(item.file_id, item.file_id)}:{item.start_line}"
        typer.echo(
            f"{item.qualified_name} {location}"
        )


def _file_map(session: object, path: Path) -> dict[str, str]:
    rows = queries.files(session, path)  # type: ignore[arg-type]
    return {row.id: row.path for row in rows}


if __name__ == "__main__":
    app()
