"""
Decomposer Chain - Splits a complex question into sub-queries.

Each sub-query is retrieved for separately on the complex path, so
chunks covering different aspects of the question are all found.
"""

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.errors import handle_llm_error, llm_retry
from app.llm import get_llm_with_structured_output

logger = structlog.get_logger()

MIN_SUB_QUERIES = 2
MAX_SUB_QUERIES = 3


class SubQueries(BaseModel):
    """Schema for a decomposed question."""

    sub_queries: list[str] = Field(
        description="2-3 self-contained search queries, each covering a distinct aspect"
    )


DECOMPOSER_SYSTEM_PROMPT = """You split complex questions into search queries for a document retrieval system.

Write 2 or 3 sub-queries that together cover everything the question asks:
- Each sub-query targets one distinct aspect (for example, each item in a comparison)
- Each is self-contained and specific, using the question's own terms
- Do not answer the question; only write the queries"""

DECOMPOSER_HUMAN_PROMPT = """Question: {query}

Write the sub-queries."""


def get_decomposer_chain():
    """
    Create the decomposer chain with structured output.

    Returns:
        Chain that outputs SubQueries schema
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", DECOMPOSER_SYSTEM_PROMPT), ("human", DECOMPOSER_HUMAN_PROMPT)]
    )
    return prompt | get_llm_with_structured_output(SubQueries)


def normalize_sub_queries(query: str, sub_queries: list[str]) -> list[str]:
    """Clamp the model's output to 2-3 distinct, non-empty sub-queries."""
    cleaned: list[str] = []
    for q in (s.strip() for s in sub_queries):
        if q and q not in cleaned:
            cleaned.append(q)
    cleaned = cleaned[:MAX_SUB_QUERIES]
    if len(cleaned) < MIN_SUB_QUERIES and query not in cleaned:
        cleaned.insert(0, query)
    return cleaned


@handle_llm_error
@llm_retry
def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex question into 2-3 sub-queries.

    Args:
        query: The user's question

    Returns:
        List of 2-3 sub-queries
    """
    result = get_decomposer_chain().invoke({"query": query})
    sub_queries = normalize_sub_queries(query, result.sub_queries)
    logger.info("Query decomposed", query=query[:50], sub_queries=sub_queries)
    return sub_queries
