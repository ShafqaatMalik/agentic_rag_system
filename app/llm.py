"""
LLM setup and utilities using Google Gemini via LangChain.
"""

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import chat_models as genai_chat_models
from tenacity import retry, stop_after_attempt

from app.config import get_settings


def _single_attempt():
    return retry(stop=stop_after_attempt(1), reraise=True)


# langchain-google-genai 2.0.8 wraps every request in a hardcoded retry (2 attempts on
# any Google API error). Make it a single attempt so app.errors.llm_retry alone decides
# what is retried and how often.
genai_chat_models._create_retry_decorator = _single_attempt


class TimeoutChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    ChatGoogleGenerativeAI that applies `timeout` to every request.

    langchain-google-genai 2.0.8 accepts a `timeout` field but never passes it to the
    Gemini client, whose default deadline is 600 s. This also passes retry=None, which
    turns off the client's silent retries of 503 errors; app.errors.llm_retry handles
    timeouts and 503s instead.
    """

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("retry", None)
        return kwargs

    def _generate(self, *args: Any, **kwargs: Any):
        return super()._generate(*args, **self._request_kwargs(kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any):
        return await super()._agenerate(*args, **self._request_kwargs(kwargs))

    def _stream(self, *args: Any, **kwargs: Any):
        yield from super()._stream(*args, **self._request_kwargs(kwargs))

    async def _astream(self, *args: Any, **kwargs: Any):
        async for chunk in super()._astream(*args, **self._request_kwargs(kwargs)):
            yield chunk


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

    return TimeoutChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        google_api_key=settings.google_api_key,
        rate_limiter=get_rate_limiter(),
        timeout=settings.llm_timeout_seconds,
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
