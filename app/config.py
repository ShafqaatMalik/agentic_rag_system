"""
Configuration settings for Agentic RAG system.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")

    # LLM Configuration
    llm_model: str = Field(default="gemini-flash-lite-latest", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")
    # Client-side rate limit shared by all LLM calls. The Gemini free tier allows
    # 15 requests/min per model; a burst of 3 at 0.2/s never exceeds that in any minute.
    llm_requests_per_second: float = Field(default=0.2, env="LLM_REQUESTS_PER_SECOND")
    llm_max_burst: int = Field(default=3, env="LLM_MAX_BURST")
    # Per-request timeout; a timed-out call is retried once, like a 503
    llm_timeout_seconds: float = Field(default=30.0, env="LLM_TIMEOUT_SECONDS")

    # Vector Store Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="documents", env="COLLECTION_NAME")

    # Embedding Configuration
    embedding_model: str = Field(default="models/gemini-embedding-001", env="EMBEDDING_MODEL")

    # Retrieval Configuration
    retrieval_k: int = Field(default=4, env="RETRIEVAL_K")

    # Agent Configuration
    max_rewrite_iterations: int = Field(default=3, env="MAX_REWRITE_ITERATIONS")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
