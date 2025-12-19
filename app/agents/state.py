"""
Agent state schema for LangGraph workflow.
"""

from typing import Literal

from langchain_core.documents import Document
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    State schema for the Agentic RAG workflow.

    This state is passed between nodes in the LangGraph and
    maintains all information needed throughout the workflow.
    """

    # Input
    query: str

    # Retrieval
    documents: list[Document]

    # Processing flags
    documents_relevant: bool

    # Generation
    generation: str | None

    # Control flow
    iteration_count: int

    # Routing
    query_type: Literal["simple", "complex"] | None

    # Metadata
    rewrite_history: list[str]

    # Timing
    start_time: float | None
    timing: dict[str, float]  # Maps node name to duration in seconds


def create_initial_state(query: str) -> AgentState:
    """
    Create initial state for a new query.

    Args:
        query: The user's input query

    Returns:
        Initialized AgentState
    """
    import time

    return AgentState(
        query=query,
        documents=[],
        documents_relevant=False,
        generation=None,
        iteration_count=0,
        query_type=None,
        rewrite_history=[],
        start_time=time.time(),
        timing={},
    )
