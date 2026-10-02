"""
End-to-end tests for complete RAG workflow.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.agents.graph import run_rag_pipeline, run_rag_pipeline_stream_tokens
from app.agents.state import create_initial_state


class TestFullFlow:
    """End-to-end tests for the complete RAG pipeline."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_simple_query_full_flow(
        self, mock_hallucination, mock_generate, mock_grade, mock_vectorstore
    ):
        """Test complete flow for a simple query with relevant documents."""
        from app.chains.generator import GenerationResult
        from app.chains.grader import GradingResult
        from app.chains.hallucination_checker import HallucinationCheck

        # Setup mocks

        mock_vs_instance = MagicMock()
        mock_vs_instance.vectorstore.similarity_search.return_value = [
            Document(
                page_content="AI is transforming healthcare with faster diagnosis.",
                metadata={"source": "healthcare.pdf"},
            )
        ]
        mock_vectorstore.return_value = mock_vs_instance

        mock_grade.return_value = GradingResult(
            relevant_docs=[
                Document(
                    page_content="AI is transforming healthcare with faster diagnosis.",
                    metadata={"source": "healthcare.pdf"},
                )
            ],
            irrelevant_count=0,
            has_relevant_docs=True,
        )

        mock_generate.return_value = GenerationResult(
            answer="AI is used in healthcare for faster diagnosis.",
            sources=["healthcare.pdf"],
            has_answer=True,
        )

        mock_hallucination.return_value = HallucinationCheck(
            is_grounded="yes", confidence="high", issues="None"
        )

        # Run pipeline
        result = await run_rag_pipeline("How is AI used in healthcare?")

        # Verify
        assert result["generation"] is not None
        assert "AI" in result["generation"] or "healthcare" in result["generation"]
        assert result["iteration_count"] == 0
        # Router classification is skipped while both labels share a path
        assert result["query_type"] is None
        assert result["is_grounded"] is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.rewrite_query")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.check_hallucination")
    @patch("app.agents.nodes.get_settings")
    async def test_query_with_rewrite(
        self,
        mock_settings,
        mock_hallucination,
        mock_generate,
        mock_rewrite,
        mock_grade,
        mock_vectorstore,
    ):
        """Test flow where initial retrieval fails and query is rewritten."""
        from app.chains.generator import GenerationResult
        from app.chains.grader import GradingResult
        from app.chains.hallucination_checker import HallucinationCheck
        from app.chains.rewriter import RewrittenQuery

        mock_settings.return_value.max_rewrite_iterations = 3
        mock_settings.return_value.retrieval_k = 4

        # First retrieval returns docs
        mock_vs_instance = MagicMock()
        mock_vs_instance.vectorstore.similarity_search.return_value = [
            Document(page_content="Irrelevant content", metadata={})
        ]
        mock_vectorstore.return_value = mock_vs_instance

        # First grading fails, second succeeds
        mock_grade.side_effect = [
            GradingResult(relevant_docs=[], irrelevant_count=1, has_relevant_docs=False),
            GradingResult(
                relevant_docs=[Document(page_content="Relevant!", metadata={})],
                irrelevant_count=0,
                has_relevant_docs=True,
            ),
        ]

        mock_rewrite.return_value = RewrittenQuery(
            rewritten_query="Better query", strategy="Expanded with synonyms"
        )

        mock_generate.return_value = GenerationResult(
            answer="Found the answer after rewrite!", sources=["doc.pdf"], has_answer=True
        )

        mock_hallucination.return_value = HallucinationCheck(
            is_grounded="yes", confidence="high", issues="None"
        )

        # Run pipeline
        result = await run_rag_pipeline("obscure query")

        # Verify rewrite happened
        assert result["iteration_count"] >= 1
        assert result["generation"] is not None


class TestNoRelevantDocsFlow:
    """Tests for when no relevant documents are found."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.rewrite_query")
    @patch("app.agents.nodes.get_settings")
    async def test_fallback_after_max_iterations(
        self, mock_settings, mock_rewrite, mock_grade, mock_vectorstore
    ):
        """Test fallback message when max rewrite iterations reached."""
        from app.chains.grader import GradingResult
        from app.chains.rewriter import RewrittenQuery

        mock_settings.return_value.max_rewrite_iterations = 3
        mock_settings.return_value.retrieval_k = 4

        mock_vs_instance = MagicMock()
        mock_vs_instance.vectorstore.similarity_search.return_value = [
            Document(page_content="Irrelevant", metadata={})
        ]
        mock_vectorstore.return_value = mock_vs_instance

        # Always return no relevant docs
        mock_grade.return_value = GradingResult(
            relevant_docs=[], irrelevant_count=1, has_relevant_docs=False
        )

        mock_rewrite.return_value = RewrittenQuery(
            rewritten_query="Still not working", strategy="Tried everything"
        )

        # Run pipeline
        result = await run_rag_pipeline("impossible query")

        # Should have fallback message
        assert result["generation"] is not None
        assert (
            "couldn't find" in result["generation"].lower()
            or "apologize" in result["generation"].lower()
        )
        assert result["iteration_count"] == 3
        # No answer was generated, so nothing was checked
        assert result["is_grounded"] is None
        assert result["query"] == "Still not working"


class TestStreamingFlow:
    """Tests for the token-streaming pipeline."""

    @staticmethod
    async def collect(query):
        """Run the streaming pipeline and return (events, streamed text)."""
        events = [u async for u in run_rag_pipeline_stream_tokens(query)]
        text = "".join(u["data"] for u in events if u["type"] == "token")
        return events, text

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.chains.generator.generate_answer_stream")
    @patch("app.agents.nodes.check_hallucination")
    async def test_stream_generates_once_and_checks_streamed_answer(
        self, mock_hallucination, mock_stream, mock_generate, mock_grade, mock_vectorstore
    ):
        """Test the answer is generated once and the check sees the streamed text."""
        from app.chains.grader import GradingResult
        from app.chains.hallucination_checker import HallucinationCheck

        doc = Document(page_content="Vector databases store embeddings.", metadata={})
        mock_vectorstore.return_value.vectorstore.similarity_search.return_value = [doc]
        mock_grade.return_value = GradingResult(
            relevant_docs=[doc], irrelevant_count=0, has_relevant_docs=True
        )

        async def fake_stream(query, documents):
            for token in ["Vector databases ", "store embeddings."]:
                yield token

        mock_stream.side_effect = fake_stream
        mock_hallucination.return_value = HallucinationCheck(
            is_grounded="yes", confidence="high", issues="None"
        )

        events, text = await self.collect("What is a vector database?")

        assert text == "Vector databases store embeddings."
        mock_stream.assert_called_once()
        mock_generate.assert_not_called()
        mock_hallucination.assert_called_once_with(text, [doc])
        assert events[-1]["data"]["check_hallucination"]["is_grounded"] is True
        assert events[-1]["type"] == "done"
        timing = events[-1]["data"]["check_hallucination"]["timing"]
        assert {"route", "retrieve", "grade", "generate", "check_hallucination"} <= set(timing)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.rewrite_query")
    @patch("app.chains.generator.generate_answer_stream")
    @patch("app.agents.nodes.get_settings")
    async def test_stream_sends_fallback_when_no_relevant_docs(
        self, mock_settings, mock_stream, mock_rewrite, mock_grade, mock_vectorstore
    ):
        """Test the no-relevant-documents message is streamed to the client."""
        from app.chains.grader import GradingResult
        from app.chains.rewriter import RewrittenQuery

        mock_settings.return_value.max_rewrite_iterations = 1
        mock_settings.return_value.retrieval_k = 4
        mock_vectorstore.return_value.vectorstore.similarity_search.return_value = [
            Document(page_content="Irrelevant", metadata={})
        ]
        mock_grade.return_value = GradingResult(
            relevant_docs=[], irrelevant_count=1, has_relevant_docs=False
        )
        mock_rewrite.return_value = RewrittenQuery(rewritten_query="q2", strategy="test")

        events, text = await self.collect("impossible query")

        assert "couldn't find relevant information" in text
        mock_stream.assert_not_called()
        assert events[-1]["type"] == "done"


class TestEmptyQueryHandling:
    """Tests for edge cases with query input."""

    @pytest.mark.e2e
    def test_create_state_with_empty_query(self):
        """Test state creation handles empty-ish queries."""
        # Single space should still create state
        state = create_initial_state(" ")
        assert state["query"] == " "

        # Normal query
        state = create_initial_state("Normal query")
        assert state["query"] == "Normal query"

    @pytest.mark.e2e
    def test_create_state_with_long_query(self):
        """Test state creation with very long query."""
        long_query = "What is " + "very " * 500 + "important?"
        state = create_initial_state(long_query)

        assert state["query"] == long_query
        assert len(state["query"]) > 2000
