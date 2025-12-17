"""
Unit tests for Rewriter Chain.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.chains.rewriter import (
    RewrittenQuery,
    rewrite_query,
    get_rewriter_chain,
    format_previous_attempts,
    REWRITER_SYSTEM_PROMPT,
    REWRITER_HUMAN_PROMPT,
    REWRITER_SIMPLE_PROMPT
)


class TestRewrittenQuery:
    """Tests for RewrittenQuery schema."""

    def test_rewritten_query_creation(self):
        """Test creating a rewritten query."""
        query = RewrittenQuery(
            rewritten_query="AI in medical diagnosis",
            strategy="Expanded with domain-specific terms"
        )
        assert query.rewritten_query == "AI in medical diagnosis"
        assert "domain-specific" in query.strategy


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.unit
    def test_format_previous_attempts_empty(self):
        """Test formatting empty previous attempts."""
        result = format_previous_attempts([])
        assert result == "None"

    @pytest.mark.unit
    def test_format_previous_attempts_none(self):
        """Test formatting None previous attempts."""
        result = format_previous_attempts(None)
        assert result == "None"

    @pytest.mark.unit
    def test_format_previous_attempts_with_data(self):
        """Test formatting previous attempts with data."""
        attempts = ["First try", "Second try", "Third try"]
        result = format_previous_attempts(attempts)

        assert "First try" in result
        assert "Second try" in result
        assert "Third try" in result
        # Should be formatted as bullet list
        assert result.count("-") == 3


class TestRewriterChain:
    """Tests for rewriter chain functionality."""

    @pytest.mark.unit
    def test_rewriter_system_prompt_content(self):
        """Test that system prompt contains key instructions."""
        assert "rewrite" in REWRITER_SYSTEM_PROMPT.lower()
        assert "query" in REWRITER_SYSTEM_PROMPT.lower()

    @pytest.mark.unit
    def test_rewriter_prompts_content(self):
        """Test that rewriter prompts contain key instructions."""
        assert "previous" in REWRITER_HUMAN_PROMPT.lower()
        assert "attempt" in REWRITER_HUMAN_PROMPT.lower()
        assert "rewrite" in REWRITER_SIMPLE_PROMPT.lower()

    @pytest.mark.unit
    @patch("app.chains.rewriter.get_llm_with_structured_output")
    def test_get_rewriter_chain_without_history(self, mock_get_llm):
        """Test rewriter chain creation without history."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_rewriter_chain(with_history=False)

        mock_get_llm.assert_called_once_with(RewrittenQuery)
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.rewriter.get_llm_with_structured_output")
    def test_get_rewriter_chain_with_history(self, mock_get_llm):
        """Test rewriter chain creation with history."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_rewriter_chain(with_history=True)

        mock_get_llm.assert_called_once_with(RewrittenQuery)
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.rewriter.get_rewriter_chain")
    def test_rewrite_query_first_attempt(self, mock_get_chain):
        """Test rewriting query on first attempt."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RewrittenQuery(
            rewritten_query="AI medical diagnosis healthcare",
            strategy="Expanded with synonyms"
        )
        mock_get_chain.return_value = mock_chain

        result = rewrite_query("AI diagnosis")

        assert result.rewritten_query == "AI medical diagnosis healthcare"
        assert "synonyms" in result.strategy
        mock_chain.invoke.assert_called_once()
        # Should be called without history (None means no history)
        call_args = mock_get_chain.call_args
        assert call_args[1]['with_history'] in [False, None, []]

    @pytest.mark.unit
    @patch("app.chains.rewriter.get_rewriter_chain")
    def test_rewrite_query_with_history(self, mock_get_chain):
        """Test rewriting query with previous attempts."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RewrittenQuery(
            rewritten_query="artificial intelligence diagnostics",
            strategy="Tried different terminology"
        )
        mock_get_chain.return_value = mock_chain

        previous = ["AI diagnosis", "AI in medicine"]
        result = rewrite_query("AI medical", previous_attempts=previous)

        assert result.rewritten_query == "artificial intelligence diagnostics"
        # Should be called with with_history=True
        mock_get_chain.assert_called_once_with(with_history=True)
        # Should invoke with previous attempts
        call_args = mock_chain.invoke.call_args[0][0]
        assert "query" in call_args
        assert "previous_attempts" in call_args

    @pytest.mark.unit
    @patch("app.chains.rewriter.get_rewriter_chain")
    def test_rewrite_query_fallback_on_error(self, mock_get_chain):
        """Test that rewriter falls back on error."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        result = rewrite_query("Test query")

        # Should fallback with simple expansion
        assert "information about" in result.rewritten_query
        assert "Test query" in result.rewritten_query
        assert "fallback" in result.strategy.lower()

    @pytest.mark.unit
    @patch("app.chains.rewriter.get_rewriter_chain")
    def test_rewrite_query_empty_previous_attempts(self, mock_get_chain):
        """Test rewriting with empty previous attempts list."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RewrittenQuery(
            rewritten_query="Better query",
            strategy="Improved"
        )
        mock_get_chain.return_value = mock_chain

        # Empty list should be treated as no history
        result = rewrite_query("Query", previous_attempts=[])

        # Empty list is falsy, so with_history will be the empty list itself
        call_args = mock_get_chain.call_args
        assert call_args[1]['with_history'] in [False, None, []]
