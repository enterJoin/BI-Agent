"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed settings."""

    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/springgraph"
    )
    log_level: str = "INFO"
    embedding_provider: str = "local"
    embedding_model: str = "local-hash-embedding-v1"
    embedding_dim: int = 1024
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_timeout_seconds: float = 60.0
    embedding_batch_size: int = 8
    embedding_max_concurrency: int = 4
    embedding_max_retries: int = 3
    embedding_retry_backoff_seconds: float = 1.0
    llm_provider: str = "openai-compatible"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_timeout_seconds: float = 60.0
    llm_temperature: float = 0.0
    planner_llm_provider: str = ""
    planner_llm_model: str = ""
    planner_llm_base_url: str = ""
    planner_llm_api_key: str = ""
    planner_llm_timeout_seconds: float | None = None
    planner_llm_temperature: float | None = None

    model_config = SettingsConfigDict(
        env_prefix="SPRINGGRAPH_",
        env_file=(
            _PROJECT_ROOT / ".env",
            _PROJECT_ROOT / "llm.env",
        ),
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return application settings."""
    return Settings()
