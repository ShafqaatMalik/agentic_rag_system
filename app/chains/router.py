"""
Router Chain - Classifies queries to determine processing path.

Routes queries as either:
- "simple": Factual, straightforward questions
- "complex": Analytical, multi-part, or reasoning-heavy questions

Complex queries are decomposed into sub-queries before retrieval.
"""

from typing import Literal

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.errors import handle_llm_error, llm_retry
from app.llm import get_llm_with_structured_output

logger = structlog.get_logger()


class RouteQuery(BaseModel):
    """Schema for query routing decision."""

    query_type: Literal["simple", "complex"] = Field(
        description="'complex' only for comparisons or clearly separate parts; otherwise 'simple'"
    )
    reasoning: str = Field(description="Brief explanation for the routing decision")


ROUTER_SYSTEM_PROMPT = """You classify questions for a document search system. The label decides how the documents are searched:
- "simple": one search is enough
- "complex": the question is split into separate searches, one per part

Default to "simple". Use "complex" only when the question:
1. Compares or contrasts two or more named things (for example "X vs Y" or "How do X and Y differ?"), or
2. Asks two or more clearly separate questions that would need different passages to answer.

A question about a single topic is "simple", even if it asks how or why something works, asks for several details about that one thing, or needs a longer explanation.

Examples:
- "What year was the company founded?" -> simple
- "How does a hash map handle collisions?" -> simple (one topic, needs an explanation)
- "What are the steps to reset a password, and how long does each take?" -> simple (one topic, several details)
- "Compare PostgreSQL and MongoDB for storing user sessions." -> complex (comparison)
- "How do TCP and UDP differ in reliability?" -> complex (comparison)
- "What is the refund policy, and which countries do you ship to?" -> complex (two unrelated parts)

Give the label and a brief reason."""

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


@handle_llm_error
@llm_retry
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
