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
    llm_model: str = Field(default="gemini-1.5-flash", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")

    # Vector Store Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="documents", env="COLLECTION_NAME")

    # Embedding Configuration
    embedding_model: str = Field(default="models/embedding-001", env="EMBEDDING_MODEL")

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
