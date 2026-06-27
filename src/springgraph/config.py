"""Application configuration."""

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
    llm_provider: str = "openai-compatible"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_timeout_seconds: float = 60.0
    llm_temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_prefix="SPRINGGRAPH_",
        env_file=(
            _PROJECT_ROOT / ".env",
            _PROJECT_ROOT / "llm.env",
        ),
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return application settings."""
    return Settings()
