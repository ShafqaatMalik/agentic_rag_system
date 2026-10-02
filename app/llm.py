"""
LLM setup and utilities using Google Gemini via LangChain.
"""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


@lru_cache
def get_rate_limiter() -> InMemoryRateLimiter:
    """
    Get the process-wide rate limiter shared by every LLM instance.

    Calls wait for a slot instead of failing with 429 when they come in bursts.
    """
    settings = get_settings()
    limiter = InMemoryRateLimiter(
        requests_per_second=settings.llm_requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=settings.llm_max_burst,
    )
    # The bucket starts empty; fill it so the first query after startup doesn't wait
    limiter.available_tokens = float(settings.llm_max_burst)
    return limiter


def get_llm(temperature: float | None = None, model: str | None = None) -> BaseChatModel:
    """
    Get a configured LLM instance.

    Args:
        temperature: Override default temperature
        model: Override default model

    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    settings = get_settings()

    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        google_api_key=settings.google_api_key,
        rate_limiter=get_rate_limiter(),
    )


def get_llm_with_structured_output(
    schema, temperature: float | None = None, model: str | None = None
) -> BaseChatModel:
    """
    Get an LLM configured for structured output.

    Args:
        schema: Pydantic model or dict schema for output
        temperature: Override default temperature
        model: Override default model

    Returns:
        LLM configured to output structured data
    """
    llm = get_llm(temperature=temperature, model=model)
    return llm.with_structured_output(schema)
