"""
Unit tests for Generator Chain.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from app.chains.generator import (
    GenerationResult,
    generate_answer,
    get_generator_chain,
    get_no_context_chain,
    format_documents,
    extract_sources,
    GENERATOR_SYSTEM_PROMPT,
    NO_CONTEXT_PROMPT
)


class TestGenerationResult:
    """Tests for GenerationResult schema."""

    def test_generation_result_with_answer(self):
        """Test creating a generation result with answer."""
        result = GenerationResult(
            answer="AI is used for diagnosis.",
            sources=["doc1.pdf"],
            has_answer=True
        )
        assert result.has_answer is True
        assert len(result.sources) == 1

    def test_generation_result_no_answer(self):
        """Test creating a generation result without answer."""
        result = GenerationResult(
            answer="No relevant information found.",
            sources=[],
            has_answer=False
        )
        assert result.has_answer is False
        assert len(result.sources) == 0


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.unit
    def test_format_documents(self):
        """Test document formatting."""
        docs = [
            Document(page_content="Content 1", metadata={"source": "doc1.pdf"}),
            Document(page_content="Content 2", metadata={"source": "doc2.pdf"}),
        ]

        result = format_documents(docs)

        assert "Content 1" in result
        assert "Content 2" in result
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    @pytest.mark.unit
    def test_format_documents_empty(self):
        """Test formatting empty document list."""
        result = format_documents([])
        assert result == "No relevant documents found."

    @pytest.mark.unit
    def test_extract_sources(self):
        """Test extracting sources from documents."""
        docs = [
            Document(page_content="Content", metadata={"source": "doc1.pdf"}),
            Document(page_content="Content", metadata={"source": "doc2.pdf"}),
            Document(page_content="Content", metadata={}),  # No source
        ]

        sources = extract_sources(docs)

        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources
        assert "Unknown" in sources  # Third doc has no source, gets "Unknown"
        assert len(sources) == 3

    @pytest.mark.unit
    def test_extract_sources_empty(self):
        """Test extracting sources from empty list."""
        sources = extract_sources([])
        assert sources == []


class TestGeneratorChain:
    """Tests for generator chain functionality."""

    @pytest.mark.unit
    def test_generator_system_prompt_content(self):
        """Test that system prompt contains key instructions."""
        assert "answer" in GENERATOR_SYSTEM_PROMPT.lower()
        assert "context" in GENERATOR_SYSTEM_PROMPT.lower()

    @pytest.mark.unit
    def test_no_context_prompt_content(self):
        """Test that no-context prompt contains key instructions."""
        assert "no relevant" in NO_CONTEXT_PROMPT.lower() or "don't have" in NO_CONTEXT_PROMPT.lower()
        assert "knowledge base" in NO_CONTEXT_PROMPT.lower() or "information" in NO_CONTEXT_PROMPT.lower()

    @pytest.mark.unit
    @patch("app.chains.generator.get_llm")
    def test_get_generator_chain(self, mock_get_llm):
        """Test generator chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_generator_chain()

        mock_get_llm.assert_called_once()
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.generator.get_llm")
    def test_get_no_context_chain(self, mock_get_llm):
        """Test no-context chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_no_context_chain()

        mock_get_llm.assert_called_once()
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.generator.get_generator_chain")
    def test_generate_answer_with_documents(self, mock_get_chain):
        """Test generating answer with documents."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "AI is used for faster diagnosis."
        mock_get_chain.return_value = mock_chain

        docs = [
            Document(
                page_content="AI improves healthcare diagnostics",
                metadata={"source": "healthcare.pdf"}
            )
        ]

        result = generate_answer("How is AI used in healthcare?", docs)

        assert result.has_answer is True
        assert "AI" in result.answer or "diagnosis" in result.answer
        assert len(result.sources) == 1
        assert "healthcare.pdf" in result.sources
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.generator.get_no_context_chain")
    def test_generate_answer_no_documents(self, mock_get_chain):
        """Test generating answer without documents."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "I apologize, but I couldn't find relevant information."
        mock_get_chain.return_value = mock_chain

        result = generate_answer("Test query", [])

        assert result.has_answer is False
        assert len(result.sources) == 0
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.generator.get_generator_chain")
    def test_generate_answer_fallback_on_error(self, mock_get_chain):
        """Test that generator falls back on error."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        docs = [Document(page_content="Test", metadata={"source": "test.pdf"})]

        result = generate_answer("Test query", docs)

        # Should fallback with apology message
        assert result.has_answer is False
        assert "apologize" in result.answer.lower() or "couldn't" in result.answer.lower()
        assert "test.pdf" in result.sources
