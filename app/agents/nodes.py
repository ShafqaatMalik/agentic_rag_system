"""
Node functions for LangGraph workflow.

Each node function takes the current state, performs an operation,
and returns state updates.
"""

from typing import Dict, Any, Callable
import time
from functools import wraps

from langchain_core.documents import Document

from app.agents.state import AgentState
from app.chains.router import route_query
from app.chains.grader import grade_documents
from app.chains.generator import generate_answer
from app.chains.rewriter import rewrite_query
from app.chains.hallucination_checker import check_hallucination
from app.retrieval.vectorstore import get_vectorstore_manager
from app.config import get_settings

import structlog

logger = structlog.get_logger()


def time_node(node_name: str):
    """
    Decorator to track execution time of a node.

    Args:
        node_name: Name of the node for timing tracking
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: AgentState) -> Dict[str, Any]:
            start = time.time()
            result = func(state)
            duration = time.time() - start

            # Add timing to result
            if result is None:
                result = {}

            # Merge existing timing with new measurement
            timing = state.get("timing", {}).copy()

            # For nodes that may be called multiple times (retrieve, grade, generate, rewrite)
            # Track them with iteration suffixes
            if node_name in timing and node_name in ["retrieve", "grade", "generate", "rewrite"]:
                iteration = state.get("iteration_count", 0)
                timing[f"{node_name}_{iteration}"] = duration
            else:
                timing[node_name] = duration

            result["timing"] = timing

            logger.debug(f"Node timing", node=node_name, duration_ms=duration * 1000)

            return result
        return wrapper
    return decorator


@time_node("route")
def route_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Route the query to determine processing path.

    Args:
        state: Current agent state

    Returns:
        State update with query_type
    """
    logger.info("Node: route_query", query=state["query"][:50])

    result = route_query(state["query"])

    return {
        "query_type": result.query_type
    }


@time_node("retrieve")
def retrieve_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Retrieve documents from vector store.

    Args:
        state: Current agent state

    Returns:
        State update with retrieved documents
    """
    query = state["query"]
    logger.info("Node: retrieve_documents", query=query[:50])

    # Get vector store manager
    vectorstore = get_vectorstore_manager()
    settings = get_settings()

    # Perform retrieval
    documents = vectorstore.vectorstore.similarity_search(
        query,
        k=settings.retrieval_k
    )

    logger.info(
        "Documents retrieved",
        count=len(documents),
        sources=[doc.metadata.get("source", "unknown") for doc in documents]
    )

    return {
        "documents": documents
    }


@time_node("grade")
def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Grade retrieved documents for relevance.

    Args:
        state: Current agent state

    Returns:
        State update with graded documents and relevance flag
    """
    logger.info(
        "Node: grade_documents",
        query=state["query"][:50],
        doc_count=len(state["documents"])
    )

    result = grade_documents(state["query"], state["documents"])

    logger.info(
        "Grading complete",
        relevant=len(result.relevant_docs),
        irrelevant=result.irrelevant_count
    )

    return {
        "documents": result.relevant_docs,
        "documents_relevant": result.has_relevant_docs
    }


@time_node("generate")
def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Generate answer from relevant documents.

    Args:
        state: Current agent state

    Returns:
        State update with generated answer
    """
    logger.info(
        "Node: generate_answer",
        query=state["query"][:50],
        doc_count=len(state["documents"])
    )

    result = generate_answer(state["query"], state["documents"])

    logger.info(
        "Answer generated",
        has_answer=result.has_answer,
        sources=result.sources
    )

    return {
        "generation": result.answer
    }


async def generate_answer_node_stream(state: AgentState):
    """
    Node: Generate answer with streaming support.
    
    Args:
        state: Current agent state
        
    Yields:
        Answer tokens as they're generated
    """
    from app.chains.generator import generate_answer_stream
    
    logger.info(
        "Node: generate_answer_stream",
        query=state["query"][:50],
        doc_count=len(state["documents"])
    )
    
    async for token in generate_answer_stream(state["query"], state["documents"]):
        yield token


@time_node("rewrite")
def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Rewrite query for better retrieval.

    Args:
        state: Current agent state

    Returns:
        State update with rewritten query and updated iteration count
    """
    current_query = state["query"]
    iteration = state["iteration_count"]
    rewrite_history = state.get("rewrite_history", [])

    logger.info(
        "Node: rewrite_query",
        original_query=current_query[:50],
        iteration=iteration
    )

    # Add current query to history before rewriting
    updated_history = rewrite_history + [current_query]

    result = rewrite_query(current_query, updated_history)

    logger.info(
        "Query rewritten",
        new_query=result.rewritten_query[:50],
        strategy=result.strategy
    )

    return {
        "query": result.rewritten_query,
        "iteration_count": iteration + 1,
        "rewrite_history": updated_history
    }


@time_node("check_hallucination")
def check_hallucination_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Check generated answer for hallucinations.

    Args:
        state: Current agent state

    Returns:
        State update (currently just logs, could add hallucination flag)
    """
    logger.info("Node: check_hallucination")

    result = check_hallucination(
        state["generation"],
        state["documents"]
    )

    logger.info(
        "Hallucination check complete",
        is_grounded=result.is_grounded,
        confidence=result.confidence,
        issues=result.issues[:50] if result.issues != "None" else "None"
    )

    # If hallucination detected, we could trigger regeneration
    # For now, we log and continue (could extend state with hallucination_detected flag)
    return {}


@time_node("no_relevant_docs")
def no_relevant_docs_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Handle case when no relevant documents found after max retries.

    Args:
        state: Current agent state

    Returns:
        State update with fallback answer
    """
    logger.info(
        "Node: no_relevant_docs",
        iterations=state["iteration_count"]
    )

    fallback_answer = (
        "I apologize, but I couldn't find relevant information in the knowledge base "
        "to answer your question. Please try rephrasing your question or ensure the "
        "relevant documents have been uploaded."
    )
    
    return {
        "generation": fallback_answer
    }


# --- Conditional Edge Functions ---

def should_rewrite_or_generate(state: AgentState) -> str:
    """
    Conditional edge: Decide whether to rewrite query or generate answer.
    
    Args:
        state: Current agent state
        
    Returns:
        "generate" if documents are relevant, "rewrite" otherwise
    """
    settings = get_settings()
    
    if state["documents_relevant"]:
        logger.info("Decision: documents relevant → generate")
        return "generate"
    
    if state["iteration_count"] >= settings.max_rewrite_iterations:
        logger.info(
            "Decision: max iterations reached → no_relevant_docs",
            iterations=state["iteration_count"]
        )
        return "no_relevant_docs"
    
    logger.info(
        "Decision: documents not relevant → rewrite",
        iteration=state["iteration_count"]
    )
    return "rewrite"


def route_by_query_type(state: AgentState) -> str:
    """
    Conditional edge: Route based on query type.
    
    Currently both simple and complex queries go to retrieval,
    but this could be extended to handle them differently.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name
    """
    query_type = state.get("query_type", "simple")
    
    # For now, both types go to retrieval
    # Could extend to use different retrieval strategies
    logger.info(f"Routing query type: {query_type} → retrieve")
    return "retrieve"
