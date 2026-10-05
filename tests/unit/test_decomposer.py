"""
Unit tests for the Decomposer Chain.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.chains.decomposer import (
    SubQueries,
    decompose_query,
    get_decomposer_chain,
    normalize_sub_queries,
)
from app.errors import LLMError


class TestNormalizeSubQueries:
    """Tests for clamping the model's output to 2-3 sub-queries."""

    @pytest.mark.unit
    def test_keeps_two_or_three(self):
        assert normalize_sub_queries("q", ["a", "b"]) == ["a", "b"]
        assert normalize_sub_queries("q", ["a", "b", "c"]) == ["a", "b", "c"]

    @pytest.mark.unit
    def test_truncates_to_three(self):
        assert normalize_sub_queries("q", ["a", "b", "c", "d"]) == ["a", "b", "c"]

    @pytest.mark.unit
    def test_drops_blanks_and_duplicates(self):
        assert normalize_sub_queries("q", ["a", " ", "a ", "b"]) == ["a", "b"]

    @pytest.mark.unit
    def test_pads_a_single_sub_query_with_the_question(self):
        assert normalize_sub_queries("question", ["only one"]) == ["question", "only one"]

    @pytest.mark.unit
    def test_falls_back_to_the_question_when_empty(self):
        assert normalize_sub_queries("question", []) == ["question"]


class TestDecomposeQuery:
    """Tests for the decompose_query chain call."""

    @pytest.mark.unit
    @patch("app.chains.decomposer.get_llm_with_structured_output")
    def test_get_decomposer_chain(self, mock_get_llm):
        mock_get_llm.return_value = MagicMock()

        chain = get_decomposer_chain()

        mock_get_llm.assert_called_once_with(SubQueries)
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.decomposer.get_decomposer_chain")
    def test_one_call_returns_normalized_sub_queries(self, mock_get_chain):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = SubQueries(sub_queries=["HNSW", "ScaNN", "LSH", "trees"])
        mock_get_chain.return_value = mock_chain

        result = decompose_query("Compare ANN algorithms")

        mock_chain.invoke.assert_called_once_with({"query": "Compare ANN algorithms"})
        assert result == ["HNSW", "ScaNN", "LSH"]

    @pytest.mark.unit
    @patch("app.chains.decomposer.get_decomposer_chain")
    def test_fails_closed_on_error(self, mock_get_chain):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        with pytest.raises(LLMError):
            decompose_query("Compare ANN algorithms")
