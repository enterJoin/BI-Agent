"""Initial schema."""

from collections.abc import Sequence

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial code graph tables."""
    op.execute(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            root_path TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            module_name TEXT,
            service_name TEXT,
            language TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            modified_at TIMESTAMPTZ NOT NULL,
            indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            error TEXT,
            UNIQUE (project_id, path)
        );

        CREATE TABLE symbols (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            language TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            start_column INTEGER NOT NULL,
            end_column INTEGER NOT NULL,
            signature TEXT,
            docstring TEXT,
            annotations JSONB NOT NULL DEFAULT '[]'::jsonb,
            modifiers JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (project_id, file_id, kind, qualified_name, start_line)
        );

        CREATE TABLE edges (
            id BIGSERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            source_id TEXT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            line INTEGER,
            column_no INTEGER,
            confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            resolved_by TEXT NOT NULL DEFAULT 'extractor',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (project_id, source_id, target_id, kind, line, column_no)
        );

        CREATE TABLE unresolved_refs (
            id BIGSERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            from_symbol_id TEXT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
            file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            reference_name TEXT NOT NULL,
            reference_kind TEXT NOT NULL,
            line INTEGER NOT NULL,
            column_no INTEGER NOT NULL,
            candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (
                project_id,
                from_symbol_id,
                reference_name,
                reference_kind,
                line,
                column_no
            )
        );

        CREATE TABLE index_runs (
            id BIGSERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            status TEXT NOT NULL,
            files_seen INTEGER NOT NULL DEFAULT 0,
            files_indexed INTEGER NOT NULL DEFAULT 0,
            symbols_created INTEGER NOT NULL DEFAULT 0,
            edges_created INTEGER NOT NULL DEFAULT 0,
            unresolved_created INTEGER NOT NULL DEFAULT 0,
            errors JSONB NOT NULL DEFAULT '[]'::jsonb
        );

        CREATE INDEX idx_files_project_path ON files(project_id, path);
        CREATE INDEX idx_files_project_module ON files(project_id, module_name);
        CREATE INDEX idx_files_project_service ON files(project_id, service_name);
        CREATE INDEX idx_symbols_project_name ON symbols(project_id, name);
        CREATE INDEX idx_symbols_project_qualified
        ON symbols(project_id, qualified_name);
        CREATE INDEX idx_symbols_project_kind ON symbols(project_id, kind);
        CREATE INDEX idx_symbols_file ON symbols(file_id);
        CREATE INDEX idx_edges_source_kind ON edges(source_id, kind);
        CREATE INDEX idx_edges_target_kind ON edges(target_id, kind);
        CREATE INDEX idx_unresolved_project_name
        ON unresolved_refs(project_id, reference_name);
        CREATE INDEX idx_symbols_annotations ON symbols USING gin(annotations);
        CREATE INDEX idx_symbols_metadata ON symbols USING gin(metadata);
        """
    )


def downgrade() -> None:
    """Drop initial code graph tables."""
    op.execute(
        """
        DROP TABLE IF EXISTS index_runs;
        DROP TABLE IF EXISTS unresolved_refs;
        DROP TABLE IF EXISTS edges;
        DROP TABLE IF EXISTS symbols;
        DROP TABLE IF EXISTS files;
        DROP TABLE IF EXISTS projects;
        """
    )
