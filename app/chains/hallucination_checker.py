"""
Hallucination Checker Chain - Validates generated answers.

Checks if the generated answer is grounded in the retrieved
context and doesn't contain fabricated information.
"""

from typing import Literal

import structlog
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.chains.generator import format_documents
from app.errors import chain_error_handler
from app.llm import get_llm_with_structured_output

logger = structlog.get_logger()


class HallucinationCheck(BaseModel):
    """Schema for hallucination check result."""

    is_grounded: Literal["yes", "no"] = Field(
        description="Whether the answer is grounded in the context: 'yes' or 'no'"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the assessment"
    )
    issues: str = Field(
        description="Description of any hallucination issues found, or 'None' if grounded"
    )


class AnswerRelevanceCheck(BaseModel):
    """Schema for answer relevance check result."""

    is_relevant: Literal["yes", "no"] = Field(
        description="Whether the answer addresses the query: 'yes' or 'no'"
    )
    reasoning: str = Field(description="Explanation of the relevance assessment")


HALLUCINATION_SYSTEM_PROMPT = """You are a hallucination detector for a RAG system. Your job is to verify that generated answers are grounded in the provided context.

Check for:
1. **Fabricated facts**: Information not present in the context
2. **Misattributed information**: Correct info attributed to wrong source
3. **Exaggerated claims**: Overstating what the context says
4. **Unsupported conclusions**: Inferences not supported by context

Guidelines:
- Be strict - the answer should be traceable to the context
- Minor paraphrasing is acceptable
- General knowledge filler (like "AI is important") is acceptable
- Specific facts, numbers, and claims must be in the context"""

HALLUCINATION_HUMAN_PROMPT = """Context:
{context}

Generated Answer:
{answer}

Is this answer grounded in the provided context? Check for hallucinations."""


RELEVANCE_SYSTEM_PROMPT = """You are an answer relevance evaluator. Your job is to check if the generated answer actually addresses the user's query.

Check for:
1. **Direct response**: Does the answer address the question asked?
2. **Completeness**: Does it cover the main aspects of the query?
3. **Off-topic content**: Is there irrelevant information?

An answer can be grounded in context but still not relevant to the query."""

RELEVANCE_HUMAN_PROMPT = """Query: {query}

Generated Answer:
{answer}

Does this answer properly address the query?"""


def get_hallucination_chain():
    """
    Create the hallucination checker chain.

    Returns:
        Chain that outputs HallucinationCheck schema
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", HALLUCINATION_SYSTEM_PROMPT), ("human", HALLUCINATION_HUMAN_PROMPT)]
    )

    llm = get_llm_with_structured_output(HallucinationCheck)

    chain = prompt | llm

    return chain


def get_relevance_chain():
    """
    Create the answer relevance chain.

    Returns:
        Chain that outputs AnswerRelevanceCheck schema
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", RELEVANCE_SYSTEM_PROMPT), ("human", RELEVANCE_HUMAN_PROMPT)]
    )

    llm = get_llm_with_structured_output(AnswerRelevanceCheck)

    chain = prompt | llm

    return chain


@chain_error_handler(
    fallback_factory=lambda answer, documents: HallucinationCheck(
        is_grounded="yes", confidence="low", issues="Check failed - defaulting to grounded"
    ),
    error_message="Hallucination check failed",
)
def check_hallucination(answer: str, documents: list[Document]) -> HallucinationCheck:
    """
    Check if an answer contains hallucinations.

    Args:
        answer: The generated answer to check
        documents: The context documents used for generation

    Returns:
        HallucinationCheck with grounding assessment
    """
    if not documents:
        logger.warning("No documents to check hallucination against")
        return HallucinationCheck(
            is_grounded="no",
            confidence="high",
            issues="No context documents provided - cannot verify grounding",
        )

    chain = get_hallucination_chain()
    context = format_documents(documents)

    result = chain.invoke({"context": context, "answer": answer})

    logger.info(
        "Hallucination check complete", is_grounded=result.is_grounded, confidence=result.confidence
    )

    return result


@chain_error_handler(
    fallback_factory=lambda query, answer: AnswerRelevanceCheck(
        is_relevant="yes", reasoning="Check failed - defaulting to relevant"
    ),
    error_message="Relevance check failed",
)
def check_answer_relevance(query: str, answer: str) -> AnswerRelevanceCheck:
    """
    Check if an answer is relevant to the query.

    Args:
        query: The original user query
        answer: The generated answer to check

    Returns:
        AnswerRelevanceCheck with relevance assessment
    """
    chain = get_relevance_chain()

    result = chain.invoke({"query": query, "answer": answer})

    logger.info("Answer relevance check complete", is_relevant=result.is_relevant)

    return result
