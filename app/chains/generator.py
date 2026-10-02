"""
Generator Chain - Produces answers from retrieved context.

Generates grounded answers based on the retrieved documents,
with clear attribution to sources.
"""

import asyncio

import structlog
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.errors import handle_llm_error, llm_retry, retry_wait, to_llm_error
from app.llm import get_llm

logger = structlog.get_logger()


class GenerationResult(BaseModel):
    """Schema for generation output."""

    answer: str
    sources: list[str]
    has_answer: bool


GENERATOR_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

CRITICAL RULES - FOLLOW STRICTLY:
1. If the user specifies "in X lines" or "in X sentences", OUTPUT EXACTLY THAT MANY - NO MORE, NO LESS
2. NEVER include mathematical formulas, equations, or notation (e.g., "MultiHead(Q,K,V) = ...", "headi = ...")
3. NEVER include variable names, parameter matrices, or code-like syntax (e.g., "WQ", "KW K", "i,VW V")
4. NEVER include implementation details, figure references, or citations unless explicitly asked
5. Extract ONLY the high-level concepts and key ideas - ignore all technical notation

Answer Guidelines:
- Answer using ONLY the conceptual information from the provided context
- Be concise and focused - extract key ideas, not formulas or implementation
- Write in plain, natural language suitable for a general technical audience
- Synthesize information - do not copy verbatim from the context
- If the context doesn't contain enough information, say so clearly
- Do not make up information not present in the context

Context will be provided as numbered passages. Extract the core concepts to formulate your answer."""

GENERATOR_HUMAN_PROMPT = """Context:
{context}

Question: {query}

Provide a helpful answer based on the context above."""

NO_CONTEXT_PROMPT = """Question: {query}

I don't have any relevant documents to answer this question. Please let the user know that no relevant information was found in the knowledge base."""


def format_documents(documents: list[Document]) -> str:
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
        formatted_parts.append(f"[{i}] Source: {source}\n{doc.page_content}")

    return "\n\n".join(formatted_parts)


def extract_sources(documents: list[Document]) -> list[str]:
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
    prompt = ChatPromptTemplate.from_messages(
        [("system", GENERATOR_SYSTEM_PROMPT), ("human", GENERATOR_HUMAN_PROMPT)]
    )

    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    return chain


def get_no_context_chain():
    """
    Create chain for when no relevant documents are found.

    Returns:
        Chain that outputs a no-context response
    """
    prompt = ChatPromptTemplate.from_messages([("human", NO_CONTEXT_PROMPT)])

    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    return chain


@handle_llm_error
@llm_retry
def generate_answer(query: str, documents: list[Document]) -> GenerationResult:
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
        return GenerationResult(answer=answer, sources=[], has_answer=False)

    chain = get_generator_chain()
    context = format_documents(documents)
    sources = extract_sources(documents)

    answer = chain.invoke({"query": query, "context": context})

    logger.info(
        "Answer generated", query=query[:50], num_sources=len(sources), answer_length=len(answer)
    )

    return GenerationResult(answer=answer, sources=sources, has_answer=True)


# Streaming version for API
async def generate_answer_stream(query: str, documents: list[Document]):
    """
    Stream generated answer tokens.

    A rate-limited request is retried once (after the server's suggested
    wait) if it fails before the first token; failures after that raise
    LLMError, since part of the answer has already been sent.

    Args:
        query: The user's input query
        documents: List of relevant documents

    Yields:
        Answer tokens as they're generated

    Raises:
        LLMError: If generation fails
    """
    if not documents:
        chain = get_no_context_chain()
        inputs = {"query": query}
    else:
        chain = get_generator_chain()
        inputs = {"query": query, "context": format_documents(documents)}
        logger.info(
            "Starting answer streaming",
            query=query[:50],
            num_sources=len(extract_sources(documents)),
        )

    for attempt in (1, 2):
        stream = chain.astream(inputs)
        try:
            first = await anext(stream)
        except StopAsyncIteration:
            return
        except Exception as e:
            wait = retry_wait(e)
            if attempt == 1 and wait is not None:
                logger.warning("Retrying streaming generation", wait_seconds=wait)
                await asyncio.sleep(wait)
                continue
            raise to_llm_error(e, "generate_answer_stream") from e

        yield first
        try:
            async for chunk in stream:
                yield chunk
        except Exception as e:
            raise to_llm_error(e, "generate_answer_stream") from e
        logger.info("Answer streaming complete")
        return
