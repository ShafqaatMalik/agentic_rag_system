"""
Router Chain - Classifies queries to determine processing path.

Routes queries as either:
- "simple": Factual, straightforward questions
- "complex": Analytical, multi-part, or reasoning-heavy questions
"""

from typing import Literal

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.errors import chain_error_handler
from app.llm import get_llm_with_structured_output

logger = structlog.get_logger()


class RouteQuery(BaseModel):
    """Schema for query routing decision."""

    query_type: Literal["simple", "complex"] = Field(
        description="The type of query: 'simple' for factual lookups, 'complex' for analytical questions"
    )
    reasoning: str = Field(description="Brief explanation for the routing decision")


ROUTER_SYSTEM_PROMPT = """You are a query classifier for a RAG system. Your job is to analyze incoming queries and classify them.

Classify the query as one of:
- "simple": Direct factual questions, single-topic lookups, straightforward information retrieval
  Examples: "What is the company revenue?", "Who is the CEO?", "What year was it founded?"

- "complex": Multi-part questions, analytical queries, comparisons, reasoning-heavy questions
  Examples: "How has revenue changed over the past 3 years and what factors contributed?",
            "Compare the AI strategies of different departments",
            "What are the implications of the new policy?"

Analyze the query and provide your classification with brief reasoning."""

ROUTER_HUMAN_PROMPT = """Query: {query}

Classify this query."""


def get_router_chain():
    """
    Create the router chain with structured output.

    Returns:
        Chain that outputs RouteQuery schema
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_SYSTEM_PROMPT), ("human", ROUTER_HUMAN_PROMPT)]
    )

    llm = get_llm_with_structured_output(RouteQuery)

    chain = prompt | llm

    return chain


@chain_error_handler(
    fallback_factory=lambda query: RouteQuery(
        query_type="simple", reasoning="Default routing due to classification error"
    ),
    error_message="Router failed, defaulting to simple",
)
def route_query(query: str) -> RouteQuery:
    """
    Route a query to determine processing path.

    Args:
        query: The user's input query

    Returns:
        RouteQuery with query_type and reasoning
    """
    chain = get_router_chain()
    result = chain.invoke({"query": query})
    logger.info(
        "Query routed", query=query[:50], query_type=result.query_type, reasoning=result.reasoning
    )
    return result
