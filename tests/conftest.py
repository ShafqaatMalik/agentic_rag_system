"""
Pytest configuration and fixtures.
"""

import os
from unittest.mock import MagicMock

import pytest

# Set test environment variables before importing app modules
os.environ["GOOGLE_API_KEY"] = "test-api-key"
os.environ["CHROMA_PERSIST_DIR"] = "./test_chroma_db"
os.environ["COLLECTION_NAME"] = "test_documents"


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    from app.config import Settings

    return Settings(
        google_api_key="test-api-key",
        llm_model="gemini-1.5-flash",
        llm_temperature=0.0,
        chroma_persist_directory="./test_chroma_db",
        collection_name="test_documents",
        embedding_model="models/embedding-001",
        retrieval_k=4,
        max_rewrite_iterations=3,
        api_host="0.0.0.0",
        api_port=8000,
        log_level="INFO",
    )


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Artificial intelligence is transforming healthcare by enabling faster diagnosis and personalized treatment plans.",
            metadata={"source": "ai_healthcare.pdf", "page": 1},
        ),
        Document(
            page_content="Machine learning models can analyze medical images to detect diseases like cancer with high accuracy.",
            metadata={"source": "ai_healthcare.pdf", "page": 2},
        ),
        Document(
            page_content="The company reported quarterly revenue of $50 million, representing a 25% increase year-over-year.",
            metadata={"source": "financial_report.pdf", "page": 1},
        ),
        Document(
            page_content="Climate change is causing rising sea levels and more frequent extreme weather events globally.",
            metadata={"source": "climate_report.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_query():
    """Provide a sample query for testing."""
    return "How is AI being used in healthcare?"


@pytest.fixture
def sample_agent_state():
    """Provide a sample agent state for testing."""
    from langchain_core.documents import Document

    from app.agents.state import AgentState

    return AgentState(
        query="How is AI being used in healthcare?",
        documents=[
            Document(
                page_content="AI is transforming healthcare through faster diagnosis.",
                metadata={"source": "test.pdf"},
            )
        ],
        documents_relevant=True,
        generation=None,
        iteration_count=0,
        query_type="simple",
        rewrite_history=[],
    )


@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test response")
    return mock


@pytest.fixture
def mock_vectorstore():
    """Provide a mock vector store for testing."""
    from langchain_core.documents import Document

    mock = MagicMock()
    mock.similarity_search.return_value = [
        Document(page_content="AI is transforming healthcare.", metadata={"source": "test.pdf"})
    ]
    return mock
