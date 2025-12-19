"""Chains module for LangChain logic components."""

from app.chains.generator import (
    GenerationResult,
    format_documents,
    generate_answer,
    generate_answer_stream,
    get_generator_chain,
)
from app.chains.grader import (
    GradeDocument,
    GradingResult,
    get_grader_chain,
    grade_document,
    grade_documents,
)
from app.chains.hallucination_checker import (
    AnswerRelevanceCheck,
    HallucinationCheck,
    check_answer_relevance,
    check_hallucination,
    get_hallucination_chain,
    get_relevance_chain,
)
from app.chains.rewriter import RewrittenQuery, get_rewriter_chain, rewrite_query
from app.chains.router import RouteQuery, get_router_chain, route_query

__all__ = [
    # Router
    "RouteQuery",
    "route_query",
    "get_router_chain",
    # Grader
    "GradeDocument",
    "GradingResult",
    "grade_document",
    "grade_documents",
    "get_grader_chain",
    # Generator
    "GenerationResult",
    "generate_answer",
    "generate_answer_stream",
    "get_generator_chain",
    "format_documents",
    # Rewriter
    "RewrittenQuery",
    "rewrite_query",
    "get_rewriter_chain",
    # Hallucination Checker
    "HallucinationCheck",
    "AnswerRelevanceCheck",
    "check_hallucination",
    "check_answer_relevance",
    "get_hallucination_chain",
    "get_relevance_chain",
]
