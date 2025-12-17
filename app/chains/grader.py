"""
Grader Chain - Evaluates document relevance to the query.

Grades each retrieved document as either relevant or not relevant
to determine if re-retrieval is needed.
"""

from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.llm import get_llm_with_structured_output
from app.errors import chain_error_handler
import structlog

logger = structlog.get_logger()


class GradeDocument(BaseModel):
    """Schema for document grading decision."""
    
    is_relevant: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to the query: 'yes' or 'no'"
    )
    reasoning: str = Field(
        description="Brief explanation for the relevance decision"
    )


class GradingResult(BaseModel):
    """Aggregated grading results for all documents."""
    
    relevant_docs: List[Document]
    irrelevant_count: int
    has_relevant_docs: bool


GRADER_SYSTEM_PROMPT = """You are a document relevance grader for a RAG system. Your job is to assess whether a retrieved document contains information relevant to answering the user's query.

Guidelines:
- A document is relevant if it contains information that could help answer the query
- Partial relevance counts as relevant
- Be generous - if there's any useful information, mark as relevant
- Consider semantic relevance, not just keyword matching

Respond with whether the document is relevant and brief reasoning."""

GRADER_HUMAN_PROMPT = """Query: {query}

Document content:
{document}

Is this document relevant to answering the query?"""


def get_grader_chain():
    """
    Create the grader chain with structured output.
    
    Returns:
        Chain that outputs GradeDocument schema
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_SYSTEM_PROMPT),
        ("human", GRADER_HUMAN_PROMPT)
    ])
    
    llm = get_llm_with_structured_output(GradeDocument)
    
    chain = prompt | llm
    
    return chain


@chain_error_handler(
    fallback_factory=lambda query, document: GradeDocument(
        is_relevant="yes",
        reasoning="Default to relevant due to grading error"
    ),
    error_message="Grading failed"
)
def grade_document(query: str, document: Document) -> GradeDocument:
    """
    Grade a single document's relevance to the query.

    Args:
        query: The user's input query
        document: The document to grade

    Returns:
        GradeDocument with is_relevant and reasoning
    """
    chain = get_grader_chain()
    result = chain.invoke({
        "query": query,
        "document": document.page_content
    })
    return result


def grade_documents(query: str, documents: List[Document]) -> GradingResult:
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
        return GradingResult(
            relevant_docs=[],
            irrelevant_count=0,
            has_relevant_docs=False
        )

    relevant_docs = []
    irrelevant_count = 0

    for doc in documents:
        grade = grade_document(query, doc)

        if grade.is_relevant == "yes":
            relevant_docs.append(doc)
        else:
            irrelevant_count += 1

    logger.info(
        "Document grading complete",
        total=len(documents),
        relevant=len(relevant_docs),
        irrelevant=irrelevant_count
    )

    return GradingResult(
        relevant_docs=relevant_docs,
        irrelevant_count=irrelevant_count,
        has_relevant_docs=len(relevant_docs) > 0
    )
