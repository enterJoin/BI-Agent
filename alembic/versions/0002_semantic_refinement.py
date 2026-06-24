"""Add semantic refinement chunk and embedding tables."""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_semantic_refinement"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create semantic chunk, embedding, and job tables."""
    op.execute(
        """
        CREATE EXTENSION IF NOT EXISTS pg_trgm;

        DO $$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS vector;
        EXCEPTION
            WHEN undefined_file THEN
                RAISE NOTICE 'pgvector is not installed; using TEXT embedding fallback';
        END $$;

        CREATE TABLE code_chunks (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            symbol_id TEXT REFERENCES symbols(id) ON DELETE SET NULL,
            index_run_id BIGINT REFERENCES index_runs(id) ON DELETE SET NULL,
            chunk_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            language TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            parser_version TEXT NOT NULL,
            template_version TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (
                project_id,
                file_id,
                symbol_id,
                chunk_type,
                content_hash,
                template_version
            )
        );

        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
                CREATE TABLE chunk_embeddings (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding vector(1024) NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (chunk_id, embedding_model, embedding_dim, content_hash)
                );
            ELSE
                CREATE TABLE chunk_embeddings (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (chunk_id, embedding_model, embedding_dim, content_hash)
                );
            END IF;
        END $$;

        CREATE TABLE embedding_jobs (
            id BIGSERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            index_run_id BIGINT REFERENCES index_runs(id) ON DELETE SET NULL,
            status TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            chunks_total INTEGER NOT NULL DEFAULT 0,
            chunks_embedded INTEGER NOT NULL DEFAULT 0,
            errors JSONB NOT NULL DEFAULT '[]'::jsonb,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ
        );

        CREATE INDEX idx_code_chunks_project_type_status
        ON code_chunks(project_id, chunk_type, status);

        CREATE INDEX idx_code_chunks_file ON code_chunks(file_id);
        CREATE INDEX idx_code_chunks_symbol ON code_chunks(symbol_id);
        CREATE INDEX idx_code_chunks_metadata ON code_chunks USING gin(metadata);
        CREATE INDEX idx_code_chunks_content_trgm
        ON code_chunks USING gin(content gin_trgm_ops);
        CREATE INDEX idx_code_chunks_fts
        ON code_chunks USING gin(to_tsvector('simple', content));

        CREATE INDEX idx_chunk_embeddings_project_model
        ON chunk_embeddings(project_id, embedding_model, embedding_dim, status);
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
                CREATE INDEX idx_chunk_embeddings_hnsw
                ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    """Drop semantic chunk, embedding, and job tables."""
    op.execute(
        """
        DROP TABLE IF EXISTS embedding_jobs;
        DROP TABLE IF EXISTS chunk_embeddings;
        DROP TABLE IF EXISTS code_chunks;
        """
    )
