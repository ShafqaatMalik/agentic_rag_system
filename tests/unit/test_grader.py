"""
Unit tests for Grader Chain.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.chains.grader import (
    GRADER_SYSTEM_PROMPT,
    GradeDocument,
    GradingResult,
    get_grader_chain,
    grade_document,
    grade_documents,
)


class TestGradeDocument:
    """Tests for GradeDocument schema."""

    def test_grade_document_relevant(self):
        """Test creating a relevant grade."""
        grade = GradeDocument(
            is_relevant="yes", reasoning="Document contains information about the query topic"
        )
        assert grade.is_relevant == "yes"

    def test_grade_document_irrelevant(self):
        """Test creating an irrelevant grade."""
        grade = GradeDocument(is_relevant="no", reasoning="Document is about a different topic")
        assert grade.is_relevant == "no"

    def test_grade_document_invalid(self):
        """Test that invalid values raise error."""
        with pytest.raises(ValueError):
            GradeDocument(is_relevant="maybe", reasoning="Uncertain")


class TestGradingResult:
    """Tests for GradingResult schema."""

    def test_grading_result_with_docs(self):
        """Test grading result with relevant documents."""
        docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
        result = GradingResult(relevant_docs=docs, irrelevant_count=1, has_relevant_docs=True)
        assert result.has_relevant_docs is True
        assert len(result.relevant_docs) == 1
        assert result.irrelevant_count == 1

    def test_grading_result_no_docs(self):
        """Test grading result with no relevant documents."""
        result = GradingResult(relevant_docs=[], irrelevant_count=3, has_relevant_docs=False)
        assert result.has_relevant_docs is False
        assert len(result.relevant_docs) == 0


class TestGraderChain:
    """Tests for grader chain functionality."""

    @pytest.mark.unit
    def test_grader_system_prompt_content(self):
        """Test that system prompt contains key instructions."""
        assert "relevant" in GRADER_SYSTEM_PROMPT.lower()
        assert "semantic" in GRADER_SYSTEM_PROMPT.lower()
        assert "generous" in GRADER_SYSTEM_PROMPT.lower()

    @pytest.mark.unit
    @patch("app.chains.grader.get_llm_with_structured_output")
    def test_get_grader_chain(self, mock_get_llm):
        """Test grader chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_grader_chain()

        mock_get_llm.assert_called_once_with(GradeDocument)
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.grader.get_grader_chain")
    def test_grade_document_relevant(self, mock_get_chain):
        """Test grading of a relevant document."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = GradeDocument(
            is_relevant="yes", reasoning="Document discusses AI in healthcare"
        )
        mock_get_chain.return_value = mock_chain

        doc = Document(
            page_content="AI is transforming healthcare diagnostics",
            metadata={"source": "healthcare.pdf"},
        )

        result = grade_document("How is AI used in healthcare?", doc)

        assert result.is_relevant == "yes"
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.grader.get_grader_chain")
    def test_grade_document_irrelevant(self, mock_get_chain):
        """Test grading of an irrelevant document."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = GradeDocument(
            is_relevant="no", reasoning="Document is about finance, not healthcare"
        )
        mock_get_chain.return_value = mock_chain

        doc = Document(
            page_content="Q3 revenue increased by 25%", metadata={"source": "finance.pdf"}
        )

        result = grade_document("How is AI used in healthcare?", doc)

        assert result.is_relevant == "no"

    @pytest.mark.unit
    @patch("app.chains.grader.get_grader_chain")
    def test_grade_document_fallback_on_error(self, mock_get_chain):
        """Test that grader falls back to relevant on error."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        doc = Document(page_content="Test content", metadata={})

        result = grade_document("Test query", doc)

        # Should fallback to relevant to avoid losing docs
        assert result.is_relevant == "yes"
        assert "error" in result.reasoning.lower()


class TestGradeDocuments:
    """Tests for batch document grading."""

    @pytest.mark.unit
    @patch("app.chains.grader.grade_document")
    def test_grade_documents_mixed(self, mock_grade):
        """Test grading multiple documents with mixed relevance."""
        # Setup mock to return different results
        mock_grade.side_effect = [
            GradeDocument(is_relevant="yes", reasoning="Relevant"),
            GradeDocument(is_relevant="no", reasoning="Irrelevant"),
            GradeDocument(is_relevant="yes", reasoning="Relevant"),
        ]

        docs = [
            Document(page_content="Content 1", metadata={"source": "1.pdf"}),
            Document(page_content="Content 2", metadata={"source": "2.pdf"}),
            Document(page_content="Content 3", metadata={"source": "3.pdf"}),
        ]

        result = grade_documents("Test query", docs)

        assert result.has_relevant_docs is True
        assert len(result.relevant_docs) == 2
        assert result.irrelevant_count == 1

    @pytest.mark.unit
    @patch("app.chains.grader.grade_document")
    def test_grade_documents_all_irrelevant(self, mock_grade):
        """Test when all documents are irrelevant."""
        mock_grade.return_value = GradeDocument(is_relevant="no", reasoning="Not relevant")

        docs = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={}),
        ]

        result = grade_documents("Test query", docs)

        assert result.has_relevant_docs is False
        assert len(result.relevant_docs) == 0
        assert result.irrelevant_count == 2

    @pytest.mark.unit
    def test_grade_documents_empty_list(self):
        """Test grading empty document list."""
        result = grade_documents("Test query", [])

        assert result.has_relevant_docs is False
        assert len(result.relevant_docs) == 0
        assert result.irrelevant_count == 0
