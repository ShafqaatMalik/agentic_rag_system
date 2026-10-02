"""
Unit tests for Grader Chain.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.chains.grader import (
    GRADER_SYSTEM_PROMPT,
    BatchGrade,
    GradeDocument,
    GradingResult,
    IndexedGrade,
    format_documents_for_grading,
    get_grader_chain,
    grade_batch,
    grade_documents,
)
from app.errors import LLMError


def make_batch(*verdicts):
    """Build a BatchGrade with 1-based indexes from 'yes'/'no' verdicts."""
    return BatchGrade(
        grades=[
            IndexedGrade(index=i, is_relevant=v, reasoning="test")
            for i, v in enumerate(verdicts, 1)
        ]
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
        """Test grader chain creation uses the batch schema."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_grader_chain()

        mock_get_llm.assert_called_once_with(BatchGrade)
        assert chain is not None

    @pytest.mark.unit
    def test_format_documents_for_grading_numbers_documents(self):
        """Test documents are numbered so grades can be matched back."""
        docs = [Document(page_content="Alpha"), Document(page_content="Beta")]

        formatted = format_documents_for_grading(docs)

        assert "[Document 1]\nAlpha" in formatted
        assert "[Document 2]\nBeta" in formatted

    @pytest.mark.unit
    @patch("app.chains.grader.get_grader_chain")
    def test_grade_batch_single_call(self, mock_get_chain):
        """Test all documents are graded in one chain call."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = make_batch("yes", "no", "yes")
        mock_get_chain.return_value = mock_chain

        docs = [Document(page_content=f"Content {i}") for i in range(3)]

        result = grade_batch("Test query", docs)

        mock_chain.invoke.assert_called_once()
        inputs = mock_chain.invoke.call_args.args[0]
        assert inputs["query"] == "Test query"
        assert inputs["count"] == 3
        assert "[Document 3]" in inputs["documents"]
        assert [g.is_relevant for g in result.grades] == ["yes", "no", "yes"]

    @pytest.mark.unit
    @patch("app.chains.grader.get_grader_chain")
    def test_grade_batch_fails_closed_on_error(self, mock_get_chain):
        """Test a failed grading call raises instead of marking documents relevant."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        docs = [Document(page_content="A"), Document(page_content="B")]

        with pytest.raises(LLMError):
            grade_batch("Test query", docs)

    @pytest.mark.unit
    @patch("app.chains.grader.get_grader_chain")
    def test_grade_documents_propagates_grading_failure(self, mock_get_chain):
        """Test grade_documents never turns a grading failure into relevant documents."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        with pytest.raises(LLMError):
            grade_documents("Test query", [Document(page_content="A")])


class TestGradeDocuments:
    """Tests for batch document grading."""

    @pytest.mark.unit
    @patch("app.chains.grader.grade_batch")
    def test_grade_documents_mixed(self, mock_grade_batch):
        """Test grading multiple documents with mixed relevance."""
        mock_grade_batch.return_value = make_batch("yes", "no", "yes")

        docs = [
            Document(page_content="Content 1", metadata={"source": "1.pdf"}),
            Document(page_content="Content 2", metadata={"source": "2.pdf"}),
            Document(page_content="Content 3", metadata={"source": "3.pdf"}),
        ]

        result = grade_documents("Test query", docs)

        mock_grade_batch.assert_called_once_with("Test query", docs)
        assert result.has_relevant_docs is True
        assert result.relevant_docs == [docs[0], docs[2]]
        assert result.irrelevant_count == 1

    @pytest.mark.unit
    @patch("app.chains.grader.grade_batch")
    def test_grade_documents_all_irrelevant(self, mock_grade_batch):
        """Test when all documents are irrelevant."""
        mock_grade_batch.return_value = make_batch("no", "no")

        docs = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={}),
        ]

        result = grade_documents("Test query", docs)

        assert result.has_relevant_docs is False
        assert len(result.relevant_docs) == 0
        assert result.irrelevant_count == 2

    @pytest.mark.unit
    @patch("app.chains.grader.grade_batch")
    def test_grade_documents_matches_by_index_not_order(self, mock_grade_batch):
        """Test grades are matched to documents by index, even if returned out of order."""
        mock_grade_batch.return_value = BatchGrade(
            grades=[
                IndexedGrade(index=2, is_relevant="yes", reasoning="test"),
                IndexedGrade(index=1, is_relevant="no", reasoning="test"),
            ]
        )
        docs = [Document(page_content="First"), Document(page_content="Second")]

        result = grade_documents("Test query", docs)

        assert result.relevant_docs == [docs[1]]

    @pytest.mark.unit
    @patch("app.chains.grader.grade_batch")
    def test_grade_documents_missing_grade_is_not_relevant(self, mock_grade_batch):
        """Test a document the grader omitted is treated as not relevant."""
        mock_grade_batch.return_value = make_batch("yes")
        docs = [Document(page_content="Graded"), Document(page_content="Omitted")]

        result = grade_documents("Test query", docs)

        assert result.relevant_docs == [docs[0]]
        assert result.irrelevant_count == 1

    @pytest.mark.unit
    @patch("app.chains.grader.grade_batch")
    def test_grade_documents_empty_list(self, mock_grade_batch):
        """Test grading empty document list makes no LLM call."""
        result = grade_documents("Test query", [])

        mock_grade_batch.assert_not_called()
        assert result.has_relevant_docs is False
        assert len(result.relevant_docs) == 0
        assert result.irrelevant_count == 0
