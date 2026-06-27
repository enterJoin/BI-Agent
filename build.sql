-- springgraph relational schema
-- Execute this script after connecting to the target PostgreSQL database.
-- Default database name used by the project: springgraph.
CREATE DATABASE springgraph
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    CONNECTION LIMIT = -1;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS files (
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

CREATE TABLE IF NOT EXISTS symbols (
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

CREATE TABLE IF NOT EXISTS edges (
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

CREATE TABLE IF NOT EXISTS unresolved_refs (
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

CREATE TABLE IF NOT EXISTS index_runs (
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

CREATE INDEX IF NOT EXISTS idx_files_project_path
ON files(project_id, path);

CREATE INDEX IF NOT EXISTS idx_files_project_module
ON files(project_id, module_name);

CREATE INDEX IF NOT EXISTS idx_files_project_service
ON files(project_id, service_name);

CREATE INDEX IF NOT EXISTS idx_symbols_project_name
ON symbols(project_id, name);

CREATE INDEX IF NOT EXISTS idx_symbols_project_qualified
ON symbols(project_id, qualified_name);

CREATE INDEX IF NOT EXISTS idx_symbols_project_kind
ON symbols(project_id, kind);

CREATE INDEX IF NOT EXISTS idx_symbols_file
ON symbols(file_id);

CREATE INDEX IF NOT EXISTS idx_edges_source_kind
ON edges(source_id, kind);

CREATE INDEX IF NOT EXISTS idx_edges_target_kind
ON edges(target_id, kind);

CREATE INDEX IF NOT EXISTS idx_unresolved_project_name
ON unresolved_refs(project_id, reference_name);

CREATE INDEX IF NOT EXISTS idx_symbols_annotations
ON symbols USING gin(annotations);

CREATE INDEX IF NOT EXISTS idx_symbols_metadata
ON symbols USING gin(metadata);

-- Semantic refinement and vector retrieval schema.
-- Embeddings are 1024-dimensional. Existing databases that still have
-- vector(384) must run Alembic revision 0003_embedding_dim_1024 and then
-- rerun refinement to regenerate embeddings.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION
    WHEN undefined_file THEN
        RAISE NOTICE 'pgvector is not installed; using TEXT embedding fallback';
END $$;

CREATE TABLE IF NOT EXISTS code_chunks (
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
    IF to_regclass('public.chunk_embeddings') IS NULL THEN
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
                UNIQUE (
                    chunk_id,
                    embedding_model,
                    embedding_dim,
                    content_hash
                )
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
                UNIQUE (
                    chunk_id,
                    embedding_model,
                    embedding_dim,
                    content_hash
                )
            );
        END IF;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS embedding_jobs (
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

-- RAG chat threads and messages.
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

CREATE TABLE IF NOT EXISTS rag_messages (
    id BIGSERIAL PRIMARY KEY,
    thread_db_id BIGINT NOT NULL REFERENCES rag_threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- RAG execution, evidence, feedback, and memory are relational tables.
-- The vector table remains chunk_embeddings.embedding.
CREATE TABLE IF NOT EXISTS rag_runs (
    id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL,
    user_id TEXT,
    question TEXT NOT NULL,
    rewritten_query TEXT,
    intent TEXT NOT NULL,
    used_vector_search BOOLEAN NOT NULL DEFAULT FALSE,
    used_relational_search BOOLEAN NOT NULL DEFAULT FALSE,
    used_source_reading BOOLEAN NOT NULL DEFAULT FALSE,
    source_reading_skipped_reason TEXT,
    answer TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_evidence (
    id BIGSERIAL PRIMARY KEY,
    rag_run_id BIGINT NOT NULL REFERENCES rag_runs(id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    file_id TEXT REFERENCES files(id) ON DELETE SET NULL,
    symbol_id TEXT REFERENCES symbols(id) ON DELETE SET NULL,
    chunk_id TEXT REFERENCES code_chunks(id) ON DELETE SET NULL,
    edge_id BIGINT REFERENCES edges(id) ON DELETE SET NULL,
    file_path TEXT,
    start_line INTEGER,
    end_line INTEGER,
    score DOUBLE PRECISION NOT NULL DEFAULT 0,
    content_excerpt TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_feedback (
    id BIGSERIAL PRIMARY KEY,
    rag_run_id BIGINT NOT NULL REFERENCES rag_runs(id) ON DELETE CASCADE,
    user_id TEXT,
    feedback_type TEXT NOT NULL,
    comment TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_memories (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    source TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_code_chunks_project_type_status
ON code_chunks(project_id, chunk_type, status);

CREATE INDEX IF NOT EXISTS idx_code_chunks_file
ON code_chunks(file_id);

CREATE INDEX IF NOT EXISTS idx_code_chunks_symbol
ON code_chunks(symbol_id);

CREATE INDEX IF NOT EXISTS idx_code_chunks_metadata
ON code_chunks USING gin(metadata);

CREATE INDEX IF NOT EXISTS idx_code_chunks_content_trgm
ON code_chunks USING gin(content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_code_chunks_fts
ON code_chunks USING gin(to_tsvector('simple', content));

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_project_model
ON chunk_embeddings(project_id, embedding_model, embedding_dim, status);

CREATE INDEX IF NOT EXISTS idx_rag_threads_project_user_updated
ON rag_threads(project_id, user_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_rag_messages_thread_created
ON rag_messages(thread_db_id, created_at, id);

CREATE INDEX IF NOT EXISTS idx_rag_runs_project_thread
ON rag_runs(project_id, thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rag_runs_project_intent
ON rag_runs(project_id, intent, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rag_evidence_run
ON rag_evidence(rag_run_id);

CREATE INDEX IF NOT EXISTS idx_rag_evidence_file
ON rag_evidence(file_id);

CREATE INDEX IF NOT EXISTS idx_rag_evidence_symbol
ON rag_evidence(symbol_id);

CREATE INDEX IF NOT EXISTS idx_rag_evidence_chunk
ON rag_evidence(chunk_id);

CREATE INDEX IF NOT EXISTS idx_rag_feedback_run
ON rag_feedback(rag_run_id);

CREATE INDEX IF NOT EXISTS idx_rag_memories_user_project
ON rag_memories(user_id, project_id, namespace, status);

CREATE INDEX IF NOT EXISTS idx_rag_memories_project_namespace
ON rag_memories(project_id, namespace, status);

CREATE INDEX IF NOT EXISTS idx_rag_memories_metadata
ON rag_memories USING gin(metadata);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw
        ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
    END IF;
END $$;
