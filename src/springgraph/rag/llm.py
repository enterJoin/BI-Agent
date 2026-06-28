"""LLM provider integration for RAG answer generation."""

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from springgraph.config import get_settings


class LlmConfigurationError(RuntimeError):
    """Raised when answer generation cannot call an LLM."""


class LlmInvocationError(RuntimeError):
    """Raised when the configured LLM call fails."""


@dataclass(frozen=True)
class _LlmConfig:
    """Resolved OpenAI-compatible chat model config."""

    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float
    temperature: float


def invoke_answer_model(prompt: str) -> str:
    """Call the configured chat model and return text content."""
    return _invoke_model(prompt, _answer_config(), "answer")


def invoke_agent_model(prompt: str) -> str:
    """Call the configured chat model for agent planning decisions."""
    return _invoke_model(prompt, _planner_config(), "planner")


def invoke_query_resolver_model(prompt: str) -> str:
    """Call the configured chat model for follow-up query resolution."""
    return _invoke_model(prompt, _query_resolver_config(), "query resolver")


def _invoke_model(prompt: str, config: _LlmConfig, purpose: str) -> str:
    """Call one configured chat model and return text content."""
    provider = config.provider.strip().lower()
    if provider not in {
        "openai",
        "openai-compatible",
        "openai_compatible",
        "apirouter",
    }:
        raise LlmConfigurationError(f"Unsupported LLM provider: {config.provider}")
    if not config.api_key.strip():
        raise LlmConfigurationError(
            f"SPRINGGRAPH_LLM_API_KEY is required for /api/rag/ask {purpose} "
            "generation."
        )

    model = ChatOpenAI(
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )
    try:
        response = model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise LlmInvocationError(f"LLM invocation failed: {exc}") from exc
    return _content_to_text(response.content)


def _answer_config() -> _LlmConfig:
    settings = get_settings()
    return _LlmConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
    )


def _planner_config() -> _LlmConfig:
    settings = get_settings()
    return _LlmConfig(
        provider=settings.planner_llm_provider or settings.llm_provider,
        model=settings.planner_llm_model or settings.llm_model,
        base_url=settings.planner_llm_base_url or settings.llm_base_url,
        api_key=settings.planner_llm_api_key or settings.llm_api_key,
        timeout_seconds=(
            settings.planner_llm_timeout_seconds
            if settings.planner_llm_timeout_seconds is not None
            else settings.llm_timeout_seconds
        ),
        temperature=(
            settings.planner_llm_temperature
            if settings.planner_llm_temperature is not None
            else settings.llm_temperature
        ),
    )


def _query_resolver_config() -> _LlmConfig:
    settings = get_settings()
    planner_config = _planner_config()
    return _LlmConfig(
        provider=settings.query_resolver_llm_provider or planner_config.provider,
        model=settings.query_resolver_llm_model or planner_config.model,
        base_url=settings.query_resolver_llm_base_url or planner_config.base_url,
        api_key=settings.query_resolver_llm_api_key or planner_config.api_key,
        timeout_seconds=(
            settings.query_resolver_llm_timeout_seconds
            if settings.query_resolver_llm_timeout_seconds is not None
            else planner_config.timeout_seconds
        ),
        temperature=(
            settings.query_resolver_llm_temperature
            if settings.query_resolver_llm_temperature is not None
            else planner_config.temperature
        ),
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()
