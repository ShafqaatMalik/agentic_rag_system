"""
Generator Chain - Produces answers from retrieved context.

Generates grounded answers based on the retrieved documents,
with clear attribution to sources.
"""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.errors import chain_error_handler
import structlog

logger = structlog.get_logger()


class GenerationResult(BaseModel):
    """Schema for generation output."""
    
    answer: str
    sources: List[str]
    has_answer: bool


GENERATOR_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
- Answer the question using ONLY the information from the provided context
- If the context doesn't contain enough information to answer, say so clearly
- Be concise but thorough
- Do not make up information not present in the context
- If multiple sources provide relevant information, synthesize them coherently

Context will be provided as numbered passages. Use them to formulate your answer."""

GENERATOR_HUMAN_PROMPT = """Context:
{context}

Question: {query}

Provide a helpful answer based on the context above."""

NO_CONTEXT_PROMPT = """Question: {query}

I don't have any relevant documents to answer this question. Please let the user know that no relevant information was found in the knowledge base."""


def format_documents(documents: List[Document]) -> str:
    """
    Format documents into a numbered context string.
    
    Args:
        documents: List of documents to format
        
    Returns:
        Formatted string with numbered passages
    """
    if not documents:
        return "No relevant documents found."
    
    formatted_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown source")
        formatted_parts.append(
            f"[{i}] Source: {source}\n{doc.page_content}"
        )
    
    return "\n\n".join(formatted_parts)


def extract_sources(documents: List[Document]) -> List[str]:
    """Extract unique source names from documents."""
    sources = set()
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        sources.add(source)
    return list(sources)


def get_generator_chain():
    """
    Create the generator chain.
    
    Returns:
        Chain that outputs answer string
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM_PROMPT),
        ("human", GENERATOR_HUMAN_PROMPT)
    ])
    
    llm = get_llm()
    
    chain = prompt | llm | StrOutputParser()
    
    return chain


def get_no_context_chain():
    """
    Create chain for when no relevant documents are found.
    
    Returns:
        Chain that outputs a no-context response
    """
    prompt = ChatPromptTemplate.from_messages([
        ("human", NO_CONTEXT_PROMPT)
    ])
    
    llm = get_llm()
    
    chain = prompt | llm | StrOutputParser()
    
    return chain


@chain_error_handler(
    fallback_factory=lambda query, documents: GenerationResult(
        answer="I apologize, but I couldn't generate a response. Please try again.",
        sources=extract_sources(documents) if documents else [],
        has_answer=False
    ),
    error_message="Generation failed"
)
def generate_answer(
    query: str,
    documents: List[Document]
) -> GenerationResult:
    """
    Generate an answer based on the query and documents.

    Args:
        query: The user's input query
        documents: List of relevant documents

    Returns:
        GenerationResult with answer, sources, and status
    """
    if not documents:
        chain = get_no_context_chain()
        answer = chain.invoke({"query": query})
        logger.info("Generated no-context response")
        return GenerationResult(
            answer=answer,
            sources=[],
            has_answer=False
        )

    chain = get_generator_chain()
    context = format_documents(documents)
    sources = extract_sources(documents)

    answer = chain.invoke({
        "query": query,
        "context": context
    })

    logger.info(
        "Answer generated",
        query=query[:50],
        num_sources=len(sources),
        answer_length=len(answer)
    )

    return GenerationResult(
        answer=answer,
        sources=sources,
        has_answer=True
    )


# Streaming version for API
async def generate_answer_stream(
    query: str,
    documents: List[Document]
):
    """
    Stream generated answer tokens.
    
    Args:
        query: The user's input query
        documents: List of relevant documents
        
    Yields:
        Answer tokens as they're generated
    """
    if not documents:
        chain = get_no_context_chain()
        try:
            async for chunk in chain.astream({"query": query}):
                yield chunk
        except Exception as e:
            logger.error("Streaming generation failed", error=str(e))
            yield "I apologize, but I couldn't generate a response. Please try again."
        return
    
    chain = get_generator_chain()
    context = format_documents(documents)
    sources = extract_sources(documents)
    
    try:
        logger.info(
            "Starting answer streaming",
            query=query[:50],
            num_sources=len(sources)
        )
        async for chunk in chain.astream({
            "query": query,
            "context": context
        }):
            yield chunk
        logger.info("Answer streaming complete")
    except Exception as e:
        logger.error("Streaming generation failed", error=str(e))
        yield "I apologize, but I couldn't generate a response. Please try again."
