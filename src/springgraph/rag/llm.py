"""LLM provider integration for RAG answer generation."""

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from springgraph.config import get_settings


class LlmConfigurationError(RuntimeError):
    """Raised when answer generation cannot call an LLM."""


class LlmInvocationError(RuntimeError):
    """Raised when the configured LLM call fails."""


def invoke_answer_model(prompt: str) -> str:
    """Call the configured chat model and return text content."""
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    if provider not in {
        "openai",
        "openai-compatible",
        "openai_compatible",
        "apirouter",
    }:
        raise LlmConfigurationError(
            f"Unsupported LLM provider: {settings.llm_provider}"
        )
    if not settings.llm_api_key.strip():
        raise LlmConfigurationError(
            "SPRINGGRAPH_LLM_API_KEY is required for /api/rag/ask answer generation."
        )

    model = ChatOpenAI(
        api_key=SecretStr(settings.llm_api_key),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )
    try:
        response = model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise LlmInvocationError(f"LLM invocation failed: {exc}") from exc
    return _content_to_text(response.content)


def invoke_agent_model(prompt: str) -> str:
    """Call the configured chat model for agent planning decisions."""
    return invoke_answer_model(prompt)


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
