"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(
        env_prefix="SPRINGGRAPH_",
        env_file=(".env", "llm.env"),
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return application settings."""
    return Settings()
