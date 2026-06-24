"""Rebuild chunk embeddings for 1024-dimensional vectors."""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_embedding_dim_1024"
down_revision: str | None = "0002_semantic_refinement"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace existing embedding rows with a 1024-dimensional vector table."""
    op.execute(
        """
        DROP TABLE IF EXISTS chunk_embeddings;

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
    """Restore the previous 384-dimensional vector table shape."""
    op.execute(
        """
        DROP TABLE IF EXISTS chunk_embeddings;

        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
                CREATE TABLE chunk_embeddings (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding vector(384) NOT NULL,
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
