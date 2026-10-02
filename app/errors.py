"""
Custom exceptions and error handling utilities.

Provides consistent error handling across the application
with proper logging and user-friendly messages.
"""

import logging
import re
import traceback
from functools import wraps
from typing import Any

import structlog
from google.api_core import exceptions as google_exceptions
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt

from app.config import get_settings

logger = structlog.get_logger()


# ============================================
# Custom Exceptions
# ============================================


class AgenticRAGError(Exception):
    """Base exception for Agentic RAG system."""

    def __init__(
        self, message: str, details: dict | None = None, original_error: Exception | None = None
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {"error": self.__class__.__name__, "message": self.message, "details": self.details}


class ConfigurationError(AgenticRAGError):
    """Raised when configuration is invalid or missing."""

    pass


class VectorStoreError(AgenticRAGError):
    """Raised when vector store operations fail."""

    pass


class DocumentIngestionError(AgenticRAGError):
    """Raised when document ingestion fails."""

    pass


class RetrievalError(AgenticRAGError):
    """Raised when document retrieval fails."""

    pass


class LLMError(AgenticRAGError):
    """Raised when LLM operations fail."""

    pass


class ChainError(AgenticRAGError):
    """Raised when a LangChain chain fails."""

    pass


class GraphExecutionError(AgenticRAGError):
    """Raised when graph execution fails."""

    pass


class ValidationError(AgenticRAGError):
    """Raised when input validation fails."""

    pass


# ============================================
# Error Handlers
# ============================================

RATE_LIMIT_MESSAGE = (
    "The Gemini free-tier rate limit was reached, so no answer could be produced. "
    "Please try again in about {seconds} seconds."
)
TIMEOUT_MESSAGE = (
    "Gemini did not respond within {seconds} seconds, so no answer could be produced. "
    "Please try again."
)
LLM_FAILURE_MESSAGE = (
    "The language model request failed, so no answer could be produced. Please try again."
)


def handle_llm_error(func):
    """
    Decorator that turns any LLM failure into an LLMError.

    There is deliberately no fallback value: a failed grade or hallucination
    check must never be treated as relevant or grounded. The LLMError carries
    a user-facing message and whether the failure was a rate limit.
    """

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LLMError:
            raise
        except Exception as e:
            raise to_llm_error(e, func.__name__) from e

    return sync_wrapper


def to_llm_error(error: Exception, function: str) -> LLMError:
    """Log an LLM failure and wrap it in an LLMError with a user-facing message."""
    rate_limited = is_rate_limit_error(error)
    timed_out = is_timeout_error(error)
    retry_after = suggested_retry_wait(error) if rate_limited else None

    logger.error(
        "LLM operation failed",
        function=function,
        rate_limited=rate_limited,
        timed_out=timed_out,
        retry_after=retry_after,
        error=str(error)[:300],
    )

    if rate_limited:
        message = RATE_LIMIT_MESSAGE.format(seconds=round(retry_after or DEFAULT_RETRY_WAIT))
    elif timed_out:
        message = TIMEOUT_MESSAGE.format(seconds=round(get_settings().llm_timeout_seconds))
    else:
        message = LLM_FAILURE_MESSAGE

    return LLMError(
        message=message,
        details={
            "function": function,
            "rate_limited": rate_limited,
            "timed_out": timed_out,
            "retry_after": retry_after,
        },
        original_error=error,
    )


def handle_retrieval_error(func):
    """
    Decorator to handle retrieval-related errors.
    """

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RetrievalError:
            raise
        except Exception as e:
            logger.error(
                "Retrieval operation failed",
                function=func.__name__,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            raise RetrievalError(
                message=f"Retrieval failed: {str(e)}",
                details={"function": func.__name__},
                original_error=e,
            ) from e

    return sync_wrapper


# ============================================
# Retry Logic
# ============================================

# Longest server-suggested wait we accept before retrying; longer waits fail fast
MAX_RETRY_WAIT = 40.0
# Wait when a retryable error carries no suggested delay
DEFAULT_RETRY_WAIT = 5.0

_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_SECONDS = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)")


def is_rate_limit_error(error: BaseException) -> bool:
    """Whether the error is a 429 / quota-exhausted response."""
    return isinstance(error, google_exceptions.ResourceExhausted) or "429" in str(error)[:100]


def is_timeout_error(error: BaseException) -> bool:
    """Whether the request hit the client-side timeout (LLM_TIMEOUT_SECONDS)."""
    return isinstance(error, google_exceptions.DeadlineExceeded | TimeoutError)


def suggested_retry_wait(error: BaseException) -> float | None:
    """The wait the server suggested in a 429 response, in seconds, if any."""
    text = str(error)
    match = _RETRY_IN.search(text) or _RETRY_DELAY_SECONDS.search(text)
    return float(match.group(1)) if match else None


def retry_wait(error: BaseException) -> float | None:
    """
    How long to wait before retrying an LLM error, or None to not retry.

    Rate limits (429), unavailability (503) and timeouts are retried after the
    server's suggested wait, or DEFAULT_RETRY_WAIT if there is none. If the server
    asks for more than MAX_RETRY_WAIT, the call fails fast instead of waiting for a
    retry that would also be rejected.
    """
    if not (
        is_rate_limit_error(error)
        or is_timeout_error(error)
        or isinstance(error, google_exceptions.ServiceUnavailable)
    ):
        return None
    wait = suggested_retry_wait(error)
    if wait is None:
        return DEFAULT_RETRY_WAIT
    return wait if wait <= MAX_RETRY_WAIT else None


def _wait_for_server(retry_state) -> float:
    return retry_wait(retry_state.outcome.exception()) or 0.0


# Retry LLM calls once (429, 503 or timeout), honouring the server's suggested wait (max 40 s)
llm_retry = retry(
    stop=stop_after_attempt(2),
    wait=_wait_for_server,
    retry=retry_if_exception(lambda e: retry_wait(e) is not None),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)


# ============================================
# Utilities
# ============================================


def asyncio_iscoroutinefunction(func) -> bool:
    """Check if function is async."""
    import asyncio

    return asyncio.iscoroutinefunction(func)


def safe_get(data: dict, *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.

    Example:
        safe_get(data, "user", "profile", "name", default="Unknown")
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


# ============================================
# API Error Responses
# ============================================


def create_error_response(
    error: Exception, status_code: int = 500, include_details: bool = False
) -> dict:
    """
    Create standardized error response for API.

    Args:
        error: The exception that occurred
        status_code: HTTP status code
        include_details: Whether to include detailed error info

    Returns:
        Dictionary suitable for FastAPI JSONResponse
    """
    if isinstance(error, AgenticRAGError):
        response = error.to_dict()
    else:
        response = {"error": error.__class__.__name__, "message": str(error)}

    response["status_code"] = status_code

    if include_details and hasattr(error, "__traceback__"):
        response["traceback"] = traceback.format_exc()

    return response
