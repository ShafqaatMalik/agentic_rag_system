"""
Integration tests for LangGraph workflow.
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.agents.graph import create_rag_graph
from app.agents.nodes import route_by_query_type, should_rewrite_or_generate
from app.agents.state import AgentState, create_initial_state


class TestAgentState:
    """Tests for AgentState creation and management."""

    @pytest.mark.integration
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state("Test query")

        assert state["query"] == "Test query"
        assert state["documents"] == []
        assert state["documents_relevant"] is False
        assert state["generation"] is None
        assert state["iteration_count"] == 0
        assert state["query_type"] is None
        assert state["rewrite_history"] == []

    @pytest.mark.integration
    def test_state_with_documents(self):
        """Test state with documents populated."""
        docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]

        state = AgentState(
            query="Test query",
            documents=docs,
            documents_relevant=True,
            generation="Test answer",
            iteration_count=1,
            query_type="simple",
            rewrite_history=["original query"],
        )

        assert len(state["documents"]) == 1
        assert state["documents_relevant"] is True
        assert state["generation"] == "Test answer"


class TestConditionalEdges:
    """Tests for conditional edge functions."""

    @pytest.mark.integration
    @patch("app.agents.nodes.get_settings")
    def test_should_rewrite_or_generate_relevant(self, mock_settings):
        """Test decision when documents are relevant."""
        mock_settings.return_value.max_rewrite_iterations = 3

        state = AgentState(
            query="Test",
            documents=[Document(page_content="Content", metadata={})],
            documents_relevant=True,
            generation=None,
            iteration_count=0,
            query_type="simple",
            rewrite_history=[],
        )

        result = should_rewrite_or_generate(state)

        assert result == "generate"

    @pytest.mark.integration
    @patch("app.agents.nodes.get_settings")
    def test_should_rewrite_or_generate_not_relevant(self, mock_settings):
        """Test decision when documents are not relevant."""
        mock_settings.return_value.max_rewrite_iterations = 3

        state = AgentState(
            query="Test",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=0,
            query_type="simple",
            rewrite_history=[],
        )

        result = should_rewrite_or_generate(state)

        assert result == "rewrite"

    @pytest.mark.integration
    @patch("app.agents.nodes.get_settings")
    def test_should_rewrite_or_generate_max_iterations(self, mock_settings):
        """Test decision when max iterations reached."""
        mock_settings.return_value.max_rewrite_iterations = 3

        state = AgentState(
            query="Test",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=3,  # Max reached
            query_type="simple",
            rewrite_history=["q1", "q2", "q3"],
        )

        result = should_rewrite_or_generate(state)

        assert result == "no_relevant_docs"

    @pytest.mark.integration
    def test_route_by_query_type_simple(self):
        """Test routing for simple query."""
        state = AgentState(
            query="What is the revenue?",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=0,
            query_type="simple",
            rewrite_history=[],
        )

        result = route_by_query_type(state)

        assert result == "retrieve"

    @pytest.mark.integration
    def test_route_by_query_type_complex(self):
        """Test routing for complex query."""
        state = AgentState(
            query="Analyze the revenue trends",
            documents=[],
            documents_relevant=False,
            generation=None,
            iteration_count=0,
            query_type="complex",
            rewrite_history=[],
        )

        result = route_by_query_type(state)

        # Currently both go to retrieve
        assert result == "retrieve"


class TestGraphStructure:
    """Tests for graph structure and compilation."""

    @pytest.mark.integration
    def test_create_rag_graph(self):
        """Test graph creation."""
        graph = create_rag_graph()

        assert graph is not None
        # Check nodes are defined
        assert "route" in graph.nodes
        assert "retrieve" in graph.nodes
        assert "grade" in graph.nodes
        assert "generate" in graph.nodes
        assert "rewrite" in graph.nodes
        assert "check_hallucination" in graph.nodes
        assert "no_relevant_docs" in graph.nodes

    @pytest.mark.integration
    def test_compile_rag_graph(self):
        """Test graph compilation."""
        graph = create_rag_graph()
        compiled = graph.compile()

        assert compiled is not None


class TestGraphStatePersistence:
    """Tests for state persistence between nodes."""

    @pytest.mark.integration
    def test_state_update_accumulation(self):
        """Test that state updates accumulate correctly."""
        initial = create_initial_state("Test query")

        # Simulate state updates from different nodes
        update1 = {"query_type": "simple"}
        update2 = {"documents": [Document(page_content="Doc", metadata={})]}
        update3 = {"documents_relevant": True}

        # Merge updates (simulating graph behavior)
        state = {**initial, **update1, **update2, **update3}

        assert state["query"] == "Test query"
        assert state["query_type"] == "simple"
        assert len(state["documents"]) == 1
        assert state["documents_relevant"] is True

    @pytest.mark.integration
    def test_iteration_count_increment(self):
        """Test iteration count increments correctly."""
        state = create_initial_state("Test")

        # Simulate rewrite node updating iteration
        for i in range(3):
            state = {
                **state,
                "iteration_count": state["iteration_count"] + 1,
                "rewrite_history": state["rewrite_history"] + [f"query_{i}"],
            }

        assert state["iteration_count"] == 3
        assert len(state["rewrite_history"]) == 3
