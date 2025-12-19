"""
LLM setup and utilities using Google Gemini via LangChain.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


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
