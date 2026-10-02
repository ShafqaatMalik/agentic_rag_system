"""
Grader Chain - Evaluates document relevance to the query.

Grades all retrieved documents as relevant or not relevant in a single
LLM call, to determine if re-retrieval is needed.
"""

from typing import Literal

import structlog
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.errors import chain_error_handler
from app.llm import get_llm_with_structured_output

logger = structlog.get_logger()


class GradeDocument(BaseModel):
    """Schema for document grading decision."""

    is_relevant: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to the query: 'yes' or 'no'"
    )
    reasoning: str = Field(description="Brief explanation for the relevance decision")


class IndexedGrade(GradeDocument):
    """Grade for one document in a batch, identified by its position."""

    index: int = Field(description="The document's number from the [Document N] header")


class BatchGrade(BaseModel):
    """Schema for grading all retrieved documents in one call."""

    grades: list[IndexedGrade] = Field(description="One grade per document, in order")


class GradingResult(BaseModel):
    """Aggregated grading results for all documents."""

    relevant_docs: list[Document]
    irrelevant_count: int
    has_relevant_docs: bool


GRADER_SYSTEM_PROMPT = """You are a document relevance grader for a RAG system. Your job is to assess whether each retrieved document contains information relevant to answering the user's query.

Guidelines:
- A document is relevant if it contains information that could help answer the query
- Partial relevance counts as relevant
- Be generous - if there's any useful information, mark as relevant
- Consider semantic relevance, not just keyword matching

Grade every document independently. For each one, return its number, whether it is relevant, and brief reasoning."""

GRADER_HUMAN_PROMPT = """Query: {query}

{documents}

Grade each of the {count} documents above for relevance to the query."""


def format_documents_for_grading(documents: list[Document]) -> str:
    """Number documents so each grade can be matched back to its document."""
    return "\n\n".join(f"[Document {i}]\n{doc.page_content}" for i, doc in enumerate(documents, 1))


def get_grader_chain():
    """
    Create the grader chain with structured output.

    Returns:
        Chain that outputs BatchGrade schema
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", GRADER_SYSTEM_PROMPT), ("human", GRADER_HUMAN_PROMPT)]
    )

    llm = get_llm_with_structured_output(BatchGrade)

    chain = prompt | llm

    return chain


@chain_error_handler(
    fallback_factory=lambda query, documents: BatchGrade(
        grades=[
            IndexedGrade(
                index=i, is_relevant="yes", reasoning="Default to relevant due to grading error"
            )
            for i in range(1, len(documents) + 1)
        ]
    ),
    error_message="Grading failed",
)
def grade_batch(query: str, documents: list[Document]) -> BatchGrade:
    """
    Grade all documents' relevance to the query in one LLM call.

    Args:
        query: The user's input query
        documents: The documents to grade

    Returns:
        BatchGrade with one indexed grade per document
    """
    chain = get_grader_chain()
    return chain.invoke(
        {
            "query": query,
            "documents": format_documents_for_grading(documents),
            "count": len(documents),
        }
    )


def grade_documents(query: str, documents: list[Document]) -> GradingResult:
    """
    Grade all documents and return aggregated results.

    Args:
        query: The user's input query
        documents: List of documents to grade

    Returns:
        GradingResult with relevant docs and statistics
    """
    if not documents:
        logger.warning("No documents to grade")
        return GradingResult(relevant_docs=[], irrelevant_count=0, has_relevant_docs=False)

    batch = grade_batch(query, documents)
    verdicts = {g.index: g.is_relevant for g in batch.grades}

    missing = [i for i in range(1, len(documents) + 1) if i not in verdicts]
    if missing:
        # An ungraded document is treated as not relevant
        logger.warning("Grader omitted documents", missing=missing)

    relevant_docs = [doc for i, doc in enumerate(documents, 1) if verdicts.get(i) == "yes"]
    irrelevant_count = len(documents) - len(relevant_docs)

    logger.info(
        "Document grading complete",
        total=len(documents),
        relevant=len(relevant_docs),
        irrelevant=irrelevant_count,
    )

    return GradingResult(
        relevant_docs=relevant_docs,
        irrelevant_count=irrelevant_count,
        has_relevant_docs=len(relevant_docs) > 0,
    )
