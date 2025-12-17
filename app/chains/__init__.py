"""Chains module for LangChain logic components."""

from app.chains.router import (
    RouteQuery,
    route_query,
    get_router_chain
)

from app.chains.grader import (
    GradeDocument,
    GradingResult,
    grade_document,
    grade_documents,
    get_grader_chain
)

from app.chains.generator import (
    GenerationResult,
    generate_answer,
    generate_answer_stream,
    get_generator_chain,
    format_documents
)

from app.chains.rewriter import (
    RewrittenQuery,
    rewrite_query,
    get_rewriter_chain
)

from app.chains.hallucination_checker import (
    HallucinationCheck,
    AnswerRelevanceCheck,
    check_hallucination,
    check_answer_relevance,
    get_hallucination_chain,
    get_relevance_chain
)

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
