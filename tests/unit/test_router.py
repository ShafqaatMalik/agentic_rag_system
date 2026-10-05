"""
Unit tests for Router Chain.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.chains.router import ROUTER_SYSTEM_PROMPT, RouteQuery, get_router_chain, route_query
from app.errors import LLMError


class TestRouteQuery:
    """Tests for RouteQuery schema."""

    def test_route_query_simple(self):
        """Test creating a simple route query."""
        route = RouteQuery(query_type="simple", reasoning="This is a direct factual question")
        assert route.query_type == "simple"
        assert route.reasoning == "This is a direct factual question"

    def test_route_query_complex(self):
        """Test creating a complex route query."""
        route = RouteQuery(query_type="complex", reasoning="This requires multi-step analysis")
        assert route.query_type == "complex"

    def test_route_query_invalid_type(self):
        """Test that invalid query types raise error."""
        with pytest.raises(ValueError):
            RouteQuery(query_type="invalid", reasoning="Test")


class TestRouterChain:
    """Tests for router chain functionality."""

    @pytest.mark.unit
    def test_router_system_prompt_content(self):
        """Test the prompt defaults to simple and limits complex to comparisons or separate parts."""
        assert 'Default to "simple"' in ROUTER_SYSTEM_PROMPT
        assert "Compares or contrasts" in ROUTER_SYSTEM_PROMPT
        assert "clearly separate questions" in ROUTER_SYSTEM_PROMPT
        assert 'single topic is "simple"' in ROUTER_SYSTEM_PROMPT
        # No longer sends "analytical" or "reasoning-heavy" questions to complex
        assert "analytical" not in ROUTER_SYSTEM_PROMPT.lower()
        assert "reasoning-heavy" not in ROUTER_SYSTEM_PROMPT.lower()

    @pytest.mark.unit
    def test_router_prompt_has_labelled_examples_of_both_kinds(self):
        examples = [ln for ln in ROUTER_SYSTEM_PROMPT.splitlines() if "->" in ln]
        assert sum("-> simple" in ln for ln in examples) >= 3
        assert sum("-> complex" in ln for ln in examples) >= 3

    @pytest.mark.unit
    def test_router_examples_do_not_leak_the_evaluation_set(self):
        """Prompt examples must not overlap the live evaluation questions."""
        import json
        from pathlib import Path

        dataset = Path(__file__).resolve().parents[2] / "evaluation" / "whitepaper_eval.json"
        data = json.loads(dataset.read_text())
        questions = [q["question"] for q in data["in_scope"] + data["out_of_scope"]]
        prompt = ROUTER_SYSTEM_PROMPT.lower()
        for question in questions:
            assert question.lower() not in prompt
        # Nor the whitepaper's own topics
        import re

        for term in ["embeddings?", "vectors?", "hnsw", "scann", "bert", "retrieval", "rag"]:
            assert not re.search(rf"\b{term}\b", prompt), term

    @pytest.mark.unit
    @patch("app.chains.router.get_llm_with_structured_output")
    def test_rules_and_examples_reach_the_model(self, mock_get_llm):
        from langchain_core.runnables import RunnableLambda

        seen = []

        def respond(prompt_value):
            seen.append(prompt_value.to_messages())
            return RouteQuery(query_type="simple", reasoning="single fact")

        mock_get_llm.return_value = RunnableLambda(respond)

        result = route_query("What does LSH stand for?")

        system, human = seen[0]
        assert 'Default to "simple"' in system.content
        assert "-> complex (comparison)" in system.content
        assert "What does LSH stand for?" in human.content
        assert result.query_type == "simple"

    @pytest.mark.unit
    @patch("app.chains.router.get_llm_with_structured_output")
    def test_get_router_chain(self, mock_get_llm):
        """Test router chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_router_chain()

        mock_get_llm.assert_called_once_with(RouteQuery)
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.router.get_router_chain")
    def test_route_query_simple(self, mock_get_chain):
        """Test routing of a simple query."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RouteQuery(
            query_type="simple", reasoning="Direct factual question"
        )
        mock_get_chain.return_value = mock_chain

        result = route_query("What is the company revenue?")

        assert result.query_type == "simple"
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.router.get_router_chain")
    def test_route_query_complex(self, mock_get_chain):
        """Test routing of a complex query."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RouteQuery(
            query_type="complex", reasoning="Multi-part analytical question"
        )
        mock_get_chain.return_value = mock_chain

        result = route_query(
            "How has revenue changed over the past 3 years and what factors contributed?"
        )

        assert result.query_type == "complex"

    @pytest.mark.unit
    @patch("app.chains.router.get_router_chain")
    def test_route_query_raises_on_error(self, mock_get_chain):
        """Test a failed classification raises instead of defaulting to simple."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        with pytest.raises(LLMError):
            route_query("Test query")


class TestRouterClassification:
    """Integration-style tests for classification logic."""

    @pytest.mark.unit
    def test_simple_query_examples(self):
        """Verify simple query patterns are understood."""
        simple_queries = [
            "What is the company revenue?",
            "Who is the CEO?",
            "What year was it founded?",
            "How many employees work here?",
        ]
        # These are documented examples - the prompt should handle them
        for query in simple_queries:
            assert len(query) > 0  # Placeholder for actual classification test

    @pytest.mark.unit
    def test_complex_query_examples(self):
        """Verify complex query patterns are understood."""
        complex_queries = [
            "How has revenue changed over the past 3 years and what factors contributed?",
            "Compare the AI strategies of different departments",
            "What are the implications of the new policy?",
            "Analyze the market trends and provide recommendations",
        ]
        for query in complex_queries:
            assert len(query) > 0  # Placeholder for actual classification test
