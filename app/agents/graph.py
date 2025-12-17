"""
LangGraph workflow definition for Agentic RAG.

This module defines the complete workflow graph that orchestrates
the RAG pipeline with self-correction capabilities.
"""

from typing import Optional

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState, create_initial_state
from app.agents.nodes import (
    route_query_node,
    retrieve_documents_node,
    grade_documents_node,
    generate_answer_node,
    rewrite_query_node,
    check_hallucination_node,
    no_relevant_docs_node,
    should_rewrite_or_generate,
    route_by_query_type
)

import structlog

logger = structlog.get_logger()


def create_rag_graph() -> StateGraph:
    """
    Create the Agentic RAG workflow graph.
    
    Workflow:
    1. Route query (classify as simple/complex)
    2. Retrieve documents
    3. Grade documents for relevance
    4. If relevant → Generate answer → Check hallucination → END
    5. If not relevant → Rewrite query → Back to retrieval (max 3 times)
    6. If max iterations → Return fallback message → END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # --- Add Nodes ---
    workflow.add_node("route", route_query_node)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("check_hallucination", check_hallucination_node)
    workflow.add_node("no_relevant_docs", no_relevant_docs_node)
    
    # --- Define Edges ---
    
    # Entry point: Start with routing
    workflow.set_entry_point("route")
    
    # Route → Retrieve (could be extended for different query types)
    workflow.add_conditional_edges(
        "route",
        route_by_query_type,
        {
            "retrieve": "retrieve"
        }
    )
    
    # Retrieve → Grade
    workflow.add_edge("retrieve", "grade")
    
    # Grade → Conditional: Generate or Rewrite
    workflow.add_conditional_edges(
        "grade",
        should_rewrite_or_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "no_relevant_docs": "no_relevant_docs"
        }
    )
    
    # Generate → Check Hallucination
    workflow.add_edge("generate", "check_hallucination")
    
    # Check Hallucination → END
    workflow.add_edge("check_hallucination", END)
    
    # Rewrite → Retrieve (loop back)
    workflow.add_edge("rewrite", "retrieve")
    
    # No Relevant Docs → END
    workflow.add_edge("no_relevant_docs", END)
    
    return workflow


def compile_rag_graph():
    """
    Create and compile the RAG graph for execution.
    
    Returns:
        Compiled graph ready to invoke
    """
    workflow = create_rag_graph()
    return workflow.compile()


# Singleton compiled graph
_compiled_graph = None


def get_compiled_graph():
    """
    Get the singleton compiled graph instance.
    
    Returns:
        Compiled RAG graph
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_rag_graph()
        logger.info("RAG graph compiled successfully")
    return _compiled_graph


async def run_rag_pipeline(query: str) -> dict:
    """
    Run the complete RAG pipeline for a query.
    
    Args:
        query: User's input query
        
    Returns:
        Final state containing the generated answer
    """
    logger.info("Starting RAG pipeline", query=query[:50])
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Get compiled graph
    graph = get_compiled_graph()
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state)
    
    logger.info(
        "RAG pipeline complete",
        iterations=final_state.get("iteration_count", 0),
        has_answer=final_state.get("generation") is not None
    )
    
    return final_state


def run_rag_pipeline_sync(query: str) -> dict:
    """
    Synchronous version of run_rag_pipeline.
    
    Args:
        query: User's input query
        
    Returns:
        Final state containing the generated answer
    """
    logger.info("Starting RAG pipeline (sync)", query=query[:50])
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Get compiled graph
    graph = get_compiled_graph()
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    logger.info(
        "RAG pipeline complete",
        iterations=final_state.get("iteration_count", 0),
        has_answer=final_state.get("generation") is not None
    )
    
    return final_state


async def run_rag_pipeline_stream(query: str):
    """
    Run RAG pipeline with streaming state updates.
    
    Args:
        query: User's input query
        
    Yields:
        State updates at each node
    """
    logger.info("Starting RAG pipeline (streaming)", query=query[:50])
    
    initial_state = create_initial_state(query)
    graph = get_compiled_graph()
    
    async for state_update in graph.astream(initial_state):
        yield state_update
    
    logger.info("RAG pipeline stream complete")


async def run_rag_pipeline_stream_tokens(query: str):
    """
    Run RAG pipeline with token-by-token streaming.
    
    Args:
        query: User's input query
        
    Yields:
        Dict with:
        - 'type': 'state_update' | 'token' | 'done'
        - 'data': State dict or token string
    """
    from app.agents.nodes import generate_answer_node_stream
    from app.chains.generator import generate_answer_stream
    
    logger.info("Starting RAG pipeline (token streaming)", query=query[:50])
    
    initial_state = create_initial_state(query)
    graph = get_compiled_graph()
    
    final_state = None
    documents = []
    
    # Run through pipeline until we reach generation
    async for state_update in graph.astream(initial_state):
        final_state = state_update
        
        # Extract documents from any node
        for node_name, node_state in state_update.items():
            if isinstance(node_state, dict) and node_state.get("documents"):
                documents = node_state["documents"]
        
        # Check if we've reached the generate node
        if "generate" in state_update:
            # Yield the pre-generation state
            yield {
                "type": "state_update",
                "data": state_update
            }
            
            # Now stream tokens from the generator
            async for token in generate_answer_stream(query, documents):
                yield {
                    "type": "token",
                    "data": token
                }
            
            # Continue to get remaining nodes (like check_hallucination)
            continue
        else:
            # Yield other node updates
            yield {
                "type": "state_update",
                "data": state_update
            }
    
    # Send completion signal with final state
    yield {
        "type": "done",
        "data": final_state
    }
    
    logger.info("RAG pipeline token stream complete")


def get_graph_visualization() -> str:
    """
    Get a Mermaid diagram of the graph for visualization.
    
    Returns:
        Mermaid diagram string
    """
    graph = get_compiled_graph()
    
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning("Could not generate graph visualization", error=str(e))
        return """
        graph TD
            A[route] --> B[retrieve]
            B --> C[grade]
            C -->|relevant| D[generate]
            C -->|not relevant| E[rewrite]
            C -->|max iterations| F[no_relevant_docs]
            D --> G[check_hallucination]
            G --> H[END]
            E --> B
            F --> H
        """
