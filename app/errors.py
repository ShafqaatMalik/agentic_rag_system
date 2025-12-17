"""
Custom exceptions and error handling utilities.

Provides consistent error handling across the application
with proper logging and user-friendly messages.
"""

from typing import Optional, Any
from functools import wraps
import traceback

import structlog

logger = structlog.get_logger()


# ============================================
# Custom Exceptions
# ============================================

class AgenticRAGError(Exception):
    """Base exception for Agentic RAG system."""
    
    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


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

def chain_error_handler(fallback_factory, error_message="Chain operation failed"):
    """
    Unified error handler for chain functions with fallback support.

    This decorator provides consistent error handling across all chain functions,
    with customizable fallback values.

    Args:
        fallback_factory: Callable that takes the same args as the wrapped function
                         and returns a fallback value on error
        error_message: Base error message for logging

    Returns:
        Decorator function

    Example:
        @chain_error_handler(
            fallback_factory=lambda query: RouteQuery(query_type="simple"),
            error_message="Router failed"
        )
        def route_query(query: str) -> RouteQuery:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(
                    error_message,
                    function=func.__name__,
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                return fallback_factory(*args, **kwargs)
        return wrapper
    return decorator


def handle_llm_error(func):
    """
    Decorator to handle LLM-related errors gracefully.

    Catches exceptions from LLM calls and wraps them
    in LLMError with proper logging.
    """
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LLMError:
            raise
        except Exception as e:
            logger.error(
                "LLM operation failed",
                function=func.__name__,
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise LLMError(
                message=f"LLM operation failed: {str(e)}",
                details={"function": func.__name__},
                original_error=e
            )

    return sync_wrapper


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
                traceback=traceback.format_exc()
            )
            raise RetrievalError(
                message=f"Retrieval failed: {str(e)}",
                details={"function": func.__name__},
                original_error=e
            )

    return sync_wrapper


# ============================================
# Retry Logic
# ============================================

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

# Configure retry for LLM calls
llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((LLMError, ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING)
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
    return text[:max_length - len(suffix)] + suffix


# ============================================
# API Error Responses
# ============================================

def create_error_response(
    error: Exception,
    status_code: int = 500,
    include_details: bool = False
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
        response = {
            "error": error.__class__.__name__,
            "message": str(error)
        }
    
    response["status_code"] = status_code
    
    if include_details and hasattr(error, "__traceback__"):
        response["traceback"] = traceback.format_exc()
    
    return response
