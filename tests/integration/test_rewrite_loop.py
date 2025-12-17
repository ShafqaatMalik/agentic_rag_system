"""
Integration tests for query rewrite loop.

Tests the self-correction mechanism that rewrites queries
when initial retrieval fails to find relevant documents.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from app.agents.state import AgentState, create_initial_state
from app.agents.nodes import (
    rewrite_query_node,
    should_rewrite_or_generate
)
from app.chains.rewriter import RewrittenQuery


class TestRewriteLoopMaxIterations:
    """Tests for rewrite loop iteration limits."""
    
    @pytest.mark.integration
    @patch("app.agents.nodes.get_settings")
    def test_stops_at_max_iterations(self, mock_settings):
        """Test that loop stops at max iterations."""
        mock_settings.return_value.max_rewrite_iterations = 3
        
        # State at max iterations
        state = AgentState(
            query="Rewritten query 3",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=3,
            query_type="simple",
            rewrite_history=["original", "rewrite1", "rewrite2"]
        )
        
        decision = should_rewrite_or_generate(state)
        
        assert decision == "no_relevant_docs"
    
    @pytest.mark.integration
    @patch("app.agents.nodes.get_settings")
    def test_continues_below_max_iterations(self, mock_settings):
        """Test that loop continues below max iterations."""
        mock_settings.return_value.max_rewrite_iterations = 3
        
        # State below max iterations
        state = AgentState(
            query="Rewritten query",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=1,
            query_type="simple",
            rewrite_history=["original"]
        )
        
        decision = should_rewrite_or_generate(state)
        
        assert decision == "rewrite"
    
    @pytest.mark.integration
    @patch("app.agents.nodes.get_settings")
    def test_exits_when_docs_relevant(self, mock_settings):
        """Test that loop exits when relevant docs found."""
        mock_settings.return_value.max_rewrite_iterations = 3
        
        # State with relevant docs found
        state = AgentState(
            query="Rewritten query",
            documents=[Document(page_content="Relevant content", metadata={})],
            documents_relevant=True,
            generation=None,
            iteration_count=2,
            query_type="simple",
            rewrite_history=["original", "rewrite1"]
        )
        
        decision = should_rewrite_or_generate(state)
        
        assert decision == "generate"


class TestRewriteNodeBehavior:
    """Tests for rewrite node behavior."""
    
    @pytest.mark.integration
    @patch("app.agents.nodes.rewrite_query")
    def test_rewrite_increments_iteration(self, mock_rewrite):
        """Test that rewrite node increments iteration count."""
        mock_rewrite.return_value = RewrittenQuery(
            rewritten_query="Improved query",
            strategy="Expanded with synonyms"
        )
        
        state = AgentState(
            query="Original query",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=0,
            query_type="simple",
            rewrite_history=[]
        )
        
        result = rewrite_query_node(state)
        
        assert result["iteration_count"] == 1
        assert result["query"] == "Improved query"
        assert "Original query" in result["rewrite_history"]
    
    @pytest.mark.integration
    @patch("app.agents.nodes.rewrite_query")
    def test_rewrite_tracks_history(self, mock_rewrite):
        """Test that rewrite history is tracked correctly."""
        mock_rewrite.return_value = RewrittenQuery(
            rewritten_query="Third attempt",
            strategy="Simplified"
        )
        
        state = AgentState(
            query="Second attempt",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=2,
            query_type="simple",
            rewrite_history=["Original", "First attempt"]
        )
        
        result = rewrite_query_node(state)
        
        assert result["iteration_count"] == 3
        assert len(result["rewrite_history"]) == 3
        assert "Second attempt" in result["rewrite_history"]
    
    @pytest.mark.integration
    @patch("app.agents.nodes.rewrite_query")
    def test_rewrite_passes_history_to_chain(self, mock_rewrite):
        """Test that previous attempts are passed to rewriter."""
        mock_rewrite.return_value = RewrittenQuery(
            rewritten_query="New query",
            strategy="Different approach"
        )
        
        state = AgentState(
            query="Current query",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=1,
            query_type="simple",
            rewrite_history=["First query"]
        )
        
        rewrite_query_node(state)
        
        # Check that rewrite was called with history
        call_args = mock_rewrite.call_args
        assert call_args is not None


class TestRewriteLoopScenarios:
    """Tests for complete rewrite loop scenarios."""
    
    @pytest.mark.integration
    def test_scenario_success_first_try(self):
        """Scenario: Relevant docs found on first retrieval."""
        states = []
        
        # Initial state
        state = create_initial_state("What is the revenue?")
        states.append(("initial", state.copy()))
        
        # After retrieval with relevant docs
        state["documents"] = [Document(page_content="Revenue is $50M", metadata={})]
        state["documents_relevant"] = True
        states.append(("after_grade", state.copy()))
        
        # Decision should be generate
        with patch("app.agents.nodes.get_settings") as mock:
            mock.return_value.max_rewrite_iterations = 3
            decision = should_rewrite_or_generate(state)
        
        assert decision == "generate"
        assert state["iteration_count"] == 0  # No rewrites needed
    
    @pytest.mark.integration
    def test_scenario_success_after_rewrite(self):
        """Scenario: Relevant docs found after one rewrite."""
        # Initial retrieval fails
        state = AgentState(
            query="obscure query",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=0,
            query_type="simple",
            rewrite_history=[]
        )
        
        with patch("app.agents.nodes.get_settings") as mock:
            mock.return_value.max_rewrite_iterations = 3
            decision1 = should_rewrite_or_generate(state)
        
        assert decision1 == "rewrite"
        
        # After rewrite and successful retrieval
        state["query"] = "clearer query"
        state["iteration_count"] = 1
        state["rewrite_history"] = ["obscure query"]
        state["documents"] = [Document(page_content="Found it!", metadata={})]
        state["documents_relevant"] = True
        
        with patch("app.agents.nodes.get_settings") as mock:
            mock.return_value.max_rewrite_iterations = 3
            decision2 = should_rewrite_or_generate(state)
        
        assert decision2 == "generate"
    
    @pytest.mark.integration
    def test_scenario_failure_all_retries(self):
        """Scenario: All rewrite attempts fail to find relevant docs."""
        iteration_decisions = []
        
        state = create_initial_state("impossible query")
        state["documents_relevant"] = False
        
        with patch("app.agents.nodes.get_settings") as mock:
            mock.return_value.max_rewrite_iterations = 3
            
            # Simulate 3 failed iterations
            for i in range(4):
                decision = should_rewrite_or_generate(state)
                iteration_decisions.append((i, decision))
                
                if decision == "rewrite":
                    state["iteration_count"] += 1
                    state["rewrite_history"].append(f"attempt_{i}")
                else:
                    break
        
        # Should have 3 rewrites then fallback
        assert iteration_decisions[-1][1] == "no_relevant_docs"
        assert state["iteration_count"] == 3
