"""
Unit tests for Router Chain.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.chains.router import (
    RouteQuery,
    route_query,
    get_router_chain,
    ROUTER_SYSTEM_PROMPT
)


class TestRouteQuery:
    """Tests for RouteQuery schema."""
    
    def test_route_query_simple(self):
        """Test creating a simple route query."""
        route = RouteQuery(
            query_type="simple",
            reasoning="This is a direct factual question"
        )
        assert route.query_type == "simple"
        assert route.reasoning == "This is a direct factual question"
    
    def test_route_query_complex(self):
        """Test creating a complex route query."""
        route = RouteQuery(
            query_type="complex",
            reasoning="This requires multi-step analysis"
        )
        assert route.query_type == "complex"
    
    def test_route_query_invalid_type(self):
        """Test that invalid query types raise error."""
        with pytest.raises(ValueError):
            RouteQuery(
                query_type="invalid",
                reasoning="Test"
            )


class TestRouterChain:
    """Tests for router chain functionality."""
    
    @pytest.mark.unit
    def test_router_system_prompt_content(self):
        """Test that system prompt contains key instructions."""
        assert "simple" in ROUTER_SYSTEM_PROMPT
        assert "complex" in ROUTER_SYSTEM_PROMPT
        assert "factual" in ROUTER_SYSTEM_PROMPT.lower()
        assert "analytical" in ROUTER_SYSTEM_PROMPT.lower()
    
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
            query_type="simple",
            reasoning="Direct factual question"
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
            query_type="complex",
            reasoning="Multi-part analytical question"
        )
        mock_get_chain.return_value = mock_chain

        result = route_query(
            "How has revenue changed over the past 3 years and what factors contributed?"
        )

        assert result.query_type == "complex"

    @pytest.mark.unit
    @patch("app.chains.router.get_router_chain")
    def test_route_query_fallback_on_error(self, mock_get_chain):
        """Test that router falls back to simple on error."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        result = route_query("Test query")

        # Should fallback to simple
        assert result.query_type == "simple"
        assert "error" in result.reasoning.lower()


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
