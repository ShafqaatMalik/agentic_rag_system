"""
Unit tests for Hallucination Checker Chain.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.chains.hallucination_checker import (
    HALLUCINATION_SYSTEM_PROMPT,
    RELEVANCE_SYSTEM_PROMPT,
    AnswerRelevanceCheck,
    HallucinationCheck,
    check_answer_relevance,
    check_hallucination,
    get_hallucination_chain,
    get_relevance_chain,
)


class TestHallucinationCheck:
    """Tests for HallucinationCheck schema."""

    def test_hallucination_check_grounded(self):
        """Test creating a grounded check."""
        check = HallucinationCheck(is_grounded="yes", confidence="high", issues="None")
        assert check.is_grounded == "yes"
        assert check.confidence == "high"

    def test_hallucination_check_not_grounded(self):
        """Test creating a not grounded check."""
        check = HallucinationCheck(
            is_grounded="no", confidence="high", issues="Answer contains fabricated statistics"
        )
        assert check.is_grounded == "no"
        assert "fabricated" in check.issues

    def test_hallucination_check_invalid_grounded(self):
        """Test that invalid grounded values raise error."""
        with pytest.raises(ValueError):
            HallucinationCheck(is_grounded="maybe", confidence="high", issues="Uncertain")

    def test_hallucination_check_invalid_confidence(self):
        """Test that invalid confidence values raise error."""
        with pytest.raises(ValueError):
            HallucinationCheck(is_grounded="yes", confidence="very high", issues="None")


class TestAnswerRelevanceCheck:
    """Tests for AnswerRelevanceCheck schema."""

    def test_answer_relevance_check_relevant(self):
        """Test creating a relevant check."""
        check = AnswerRelevanceCheck(
            is_relevant="yes", reasoning="Answer directly addresses the query"
        )
        assert check.is_relevant == "yes"

    def test_answer_relevance_check_not_relevant(self):
        """Test creating a not relevant check."""
        check = AnswerRelevanceCheck(is_relevant="no", reasoning="Answer is off-topic")
        assert check.is_relevant == "no"

    def test_answer_relevance_check_invalid(self):
        """Test that invalid values raise error."""
        with pytest.raises(ValueError):
            AnswerRelevanceCheck(is_relevant="maybe", reasoning="Uncertain")


class TestHallucinationCheckerChain:
    """Tests for hallucination checker chain functionality."""

    @pytest.mark.unit
    def test_hallucination_system_prompt_content(self):
        """Test that system prompt contains key instructions."""
        assert "hallucination" in HALLUCINATION_SYSTEM_PROMPT.lower()
        assert "grounded" in HALLUCINATION_SYSTEM_PROMPT.lower()
        assert "context" in HALLUCINATION_SYSTEM_PROMPT.lower()

    @pytest.mark.unit
    def test_relevance_system_prompt_content(self):
        """Test that relevance prompt contains key instructions."""
        assert "relevance" in RELEVANCE_SYSTEM_PROMPT.lower()
        assert "query" in RELEVANCE_SYSTEM_PROMPT.lower()

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_llm_with_structured_output")
    def test_get_hallucination_chain(self, mock_get_llm):
        """Test hallucination chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_hallucination_chain()

        mock_get_llm.assert_called_once_with(HallucinationCheck)
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_llm_with_structured_output")
    def test_get_relevance_chain(self, mock_get_llm):
        """Test relevance chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_relevance_chain()

        mock_get_llm.assert_called_once_with(AnswerRelevanceCheck)
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_hallucination_chain")
    def test_check_hallucination_grounded(self, mock_get_chain):
        """Test checking a grounded answer."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = HallucinationCheck(
            is_grounded="yes", confidence="high", issues="None"
        )
        mock_get_chain.return_value = mock_chain

        docs = [
            Document(
                page_content="Revenue was $50 million in Q3 2024", metadata={"source": "report.pdf"}
            )
        ]
        answer = "The revenue was $50 million in Q3 2024."

        result = check_hallucination(answer, docs)

        assert result.is_grounded == "yes"
        assert result.confidence == "high"
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_hallucination_chain")
    def test_check_hallucination_not_grounded(self, mock_get_chain):
        """Test checking a hallucinated answer."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = HallucinationCheck(
            is_grounded="no", confidence="high", issues="Answer contains fabricated revenue figure"
        )
        mock_get_chain.return_value = mock_chain

        docs = [Document(page_content="Revenue was $50 million", metadata={"source": "report.pdf"})]
        answer = "Revenue was $100 million, a 50% increase."

        result = check_hallucination(answer, docs)

        assert result.is_grounded == "no"
        assert "fabricated" in result.issues

    @pytest.mark.unit
    def test_check_hallucination_no_documents(self):
        """Test checking hallucination with no documents."""
        answer = "Some answer"

        result = check_hallucination(answer, [])

        # Should return not grounded when no docs
        assert result.is_grounded == "no"
        assert result.confidence == "high"
        assert "no context" in result.issues.lower()

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_hallucination_chain")
    def test_check_hallucination_fallback_on_error(self, mock_get_chain):
        """Test that hallucination checker falls back on error."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        docs = [Document(page_content="Test", metadata={})]
        result = check_hallucination("Answer", docs)

        # Should fallback to grounded (conservative)
        assert result.is_grounded == "yes"
        assert result.confidence == "low"
        assert "failed" in result.issues.lower()

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_relevance_chain")
    def test_check_answer_relevance_relevant(self, mock_get_chain):
        """Test checking a relevant answer."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = AnswerRelevanceCheck(
            is_relevant="yes", reasoning="Answer directly addresses revenue question"
        )
        mock_get_chain.return_value = mock_chain

        query = "What is the revenue?"
        answer = "Revenue is $50 million."

        result = check_answer_relevance(query, answer)

        assert result.is_relevant == "yes"
        assert "revenue" in result.reasoning.lower()
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_relevance_chain")
    def test_check_answer_relevance_not_relevant(self, mock_get_chain):
        """Test checking an irrelevant answer."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = AnswerRelevanceCheck(
            is_relevant="no", reasoning="Answer discusses weather instead of revenue"
        )
        mock_get_chain.return_value = mock_chain

        query = "What is the revenue?"
        answer = "The weather is sunny today."

        result = check_answer_relevance(query, answer)

        assert result.is_relevant == "no"
        assert "weather" in result.reasoning.lower()

    @pytest.mark.unit
    @patch("app.chains.hallucination_checker.get_relevance_chain")
    def test_check_answer_relevance_fallback_on_error(self, mock_get_chain):
        """Test that relevance checker falls back on error."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        result = check_answer_relevance("Query", "Answer")

        # Should fallback to relevant (conservative)
        assert result.is_relevant == "yes"
        assert "failed" in result.reasoning.lower()
