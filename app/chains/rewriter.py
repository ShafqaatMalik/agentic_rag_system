"""
Rewriter Chain - Reformulates queries for better retrieval.

When initial retrieval fails to find relevant documents,
this chain rewrites the query to improve search results.
"""

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.errors import chain_error_handler
from app.llm import get_llm_with_structured_output

logger = structlog.get_logger()


class RewrittenQuery(BaseModel):
    """Schema for rewritten query output."""

    rewritten_query: str = Field(description="The reformulated query optimized for retrieval")
    strategy: str = Field(description="Brief explanation of the rewriting strategy used")


REWRITER_SYSTEM_PROMPT = """You are a query rewriter for a RAG system. Your job is to reformulate queries that failed to retrieve relevant documents.

Strategies to improve retrieval:
1. **Expand**: Add synonyms or related terms
2. **Simplify**: Remove unnecessary words, focus on key concepts
3. **Rephrase**: Use different terminology that might match documents better
4. **Decompose**: If complex, focus on the core question
5. **Generalize**: If too specific, broaden the scope slightly

Guidelines:
- Keep the semantic meaning intact
- Optimize for vector similarity search
- Use clear, specific nouns and concepts
- Avoid filler words and overly complex phrasing

Previous attempts that failed will be provided so you don't repeat them."""

REWRITER_HUMAN_PROMPT = """Original query: {query}

Previous failed attempts:
{previous_attempts}

Rewrite this query to improve document retrieval. Use a different approach than previous attempts."""

REWRITER_SIMPLE_PROMPT = """Original query: {query}

Rewrite this query to improve document retrieval."""


def get_rewriter_chain(with_history: bool = False):
    """
    Create the rewriter chain with structured output.

    Args:
        with_history: Whether to include previous attempts in prompt

    Returns:
        Chain that outputs RewrittenQuery schema
    """
    if with_history:
        prompt = ChatPromptTemplate.from_messages(
            [("system", REWRITER_SYSTEM_PROMPT), ("human", REWRITER_HUMAN_PROMPT)]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [("system", REWRITER_SYSTEM_PROMPT), ("human", REWRITER_SIMPLE_PROMPT)]
        )

    llm = get_llm_with_structured_output(RewrittenQuery)

    chain = prompt | llm

    return chain


def format_previous_attempts(attempts: list[str]) -> str:
    """Format previous query attempts for the prompt."""
    if not attempts:
        return "None"

    return "\n".join(f"- {attempt}" for attempt in attempts)


@chain_error_handler(
    fallback_factory=lambda query, previous_attempts=None: RewrittenQuery(
        rewritten_query=f"information about {query}",
        strategy="Fallback: simple expansion due to rewriter error",
    ),
    error_message="Rewriter failed",
)
def rewrite_query(query: str, previous_attempts: list[str] | None = None) -> RewrittenQuery:
    """
    Rewrite a query to improve retrieval.

    Args:
        query: The original query that failed
        previous_attempts: List of previous rewrite attempts

    Returns:
        RewrittenQuery with new query and strategy explanation
    """
    has_history = previous_attempts and len(previous_attempts) > 0
    chain = get_rewriter_chain(with_history=has_history)

    if has_history:
        result = chain.invoke(
            {"query": query, "previous_attempts": format_previous_attempts(previous_attempts)}
        )
    else:
        result = chain.invoke({"query": query})

    logger.info(
        "Query rewritten",
        original=query[:50],
        rewritten=result.rewritten_query[:50],
        strategy=result.strategy,
    )

    return result
