"""Add RAG chat thread and message tables."""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_rag_chat_threads"
down_revision: str | None = "0003_embedding_dim_1024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create persisted RAG conversation tables."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_threads (
            id BIGSERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (project_id, user_id, thread_id)
        );

        CREATE INDEX IF NOT EXISTS idx_rag_threads_project_user_updated
        ON rag_threads(project_id, user_id, updated_at);

        CREATE TABLE IF NOT EXISTS rag_messages (
            id BIGSERIAL PRIMARY KEY,
            thread_db_id BIGINT NOT NULL
                REFERENCES rag_threads(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_rag_messages_thread_created
        ON rag_messages(thread_db_id, created_at, id);
        """
    )


def downgrade() -> None:
    """Drop persisted RAG conversation tables."""
    op.execute(
        """
        DROP TABLE IF EXISTS rag_messages;
        DROP TABLE IF EXISTS rag_threads;
        """
    )
