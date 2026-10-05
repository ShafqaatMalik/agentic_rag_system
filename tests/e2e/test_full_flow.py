"""
End-to-end tests for complete RAG workflow.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.agents.graph import run_rag_pipeline, run_rag_pipeline_stream_tokens
from app.agents.state import create_initial_state


@pytest.fixture(autouse=True)
def router():
    """Route every query as simple; a test can set router.return_value for complex."""
    from app.chains.router import RouteQuery

    with patch(
        "app.agents.nodes.route_query",
        return_value=RouteQuery(query_type="simple", reasoning="test"),
    ) as mock_route:
        yield mock_route


class TestFullFlow:
    """End-to-end tests for the complete RAG pipeline."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_simple_query_full_flow(
        self, mock_hallucination, mock_generate, mock_grade, mock_vectorstore, router
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
        assert result["query_type"] == "simple"
        assert result["sub_queries"] == []
        assert result["is_grounded"] is True
        # Simple path budget: route, grade, generate, check = 4 LLM calls
        llm_calls = [router, mock_grade, mock_generate, mock_hallucination]
        assert [m.call_count for m in llm_calls] == [1, 1, 1, 1]
        mock_vectorstore.return_value.vectorstore.similarity_search.assert_called_once()

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


class TestComplexPath:
    """Tests for complex queries: decomposition, multi-retrieval and one batched grade."""

    @staticmethod
    def setup_complex(router, mock_vectorstore, docs):
        from app.chains.router import RouteQuery

        router.return_value = RouteQuery(query_type="complex", reasoning="comparison")
        mock_vectorstore.return_value.retrieve_for_question.return_value = docs

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.decompose_query")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_complex_query_decomposes_and_grades_once(
        self,
        mock_hallucination,
        mock_generate,
        mock_grade,
        mock_decompose,
        mock_vectorstore,
        router,
    ):
        """Test a complex query retrieves per sub-query and grades the merged set once."""
        from app.chains.generator import GenerationResult
        from app.chains.grader import GradingResult
        from app.chains.hallucination_checker import HallucinationCheck

        docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(6)]
        self.setup_complex(router, mock_vectorstore, docs)
        sub_queries = ["How does HNSW trade accuracy for speed?", "How does ScaNN do it?"]
        mock_decompose.return_value = sub_queries
        mock_grade.return_value = GradingResult(
            relevant_docs=docs[:3], irrelevant_count=3, has_relevant_docs=True
        )
        mock_generate.return_value = GenerationResult(answer="A", sources=[], has_answer=True)
        mock_hallucination.return_value = HallucinationCheck(
            is_grounded="yes", confidence="high", issues="None"
        )

        question = "Compare how HNSW and ScaNN trade accuracy for speed"
        result = await run_rag_pipeline(question)

        assert result["query_type"] == "complex"
        assert result["sub_queries"] == sub_queries
        mock_decompose.assert_called_once_with(question)
        # Retrieval covers the original question as well as its sub-queries
        mock_vectorstore.return_value.retrieve_for_question.assert_called_once_with(
            question, sub_queries, k=4
        )
        mock_vectorstore.return_value.vectorstore.similarity_search.assert_not_called()
        # The grader sees every merged chunk and the sub-queries
        mock_grade.assert_called_once_with(question, docs, sub_queries=sub_queries)
        # Complex path budget: route, decompose, grade, generate, check = 5 LLM calls
        llm_calls = [router, mock_decompose, mock_grade, mock_generate, mock_hallucination]
        assert [m.call_count for m in llm_calls] == [1, 1, 1, 1, 1]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.decompose_query")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.rewrite_query")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_complex_rewrite_is_decomposed_again(
        self,
        mock_hallucination,
        mock_generate,
        mock_rewrite,
        mock_grade,
        mock_decompose,
        mock_vectorstore,
        router,
    ):
        """Test a rewritten complex query goes back through decomposition."""
        from app.chains.generator import GenerationResult
        from app.chains.grader import GradingResult
        from app.chains.hallucination_checker import HallucinationCheck
        from app.chains.rewriter import RewrittenQuery

        doc = Document(page_content="chunk", metadata={})
        self.setup_complex(router, mock_vectorstore, [doc])
        mock_decompose.side_effect = [["a1", "a2"], ["b1", "b2", "b3"]]
        mock_grade.side_effect = [
            GradingResult(relevant_docs=[], irrelevant_count=1, has_relevant_docs=False),
            GradingResult(relevant_docs=[doc], irrelevant_count=0, has_relevant_docs=True),
        ]
        mock_rewrite.return_value = RewrittenQuery(rewritten_query="better", strategy="test")
        mock_generate.return_value = GenerationResult(answer="A", sources=[], has_answer=True)
        mock_hallucination.return_value = HallucinationCheck(
            is_grounded="yes", confidence="high", issues="None"
        )

        result = await run_rag_pipeline("original question")

        assert [c.args[0] for c in mock_decompose.call_args_list] == ["original question", "better"]
        assert mock_vectorstore.return_value.retrieve_for_question.call_count == 2
        mock_vectorstore.return_value.vectorstore.similarity_search.assert_not_called()
        assert result["sub_queries"] == ["b1", "b2", "b3"]
        assert result["iteration_count"] == 1

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.decompose_query")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.rewrite_query")
    async def test_complex_no_relevant_docs_after_max_rewrites(
        self, mock_rewrite, mock_grade, mock_decompose, mock_vectorstore, router
    ):
        """Test the no-relevant-docs exit on the complex path, at its 12-call budget."""
        from app.chains.grader import GradingResult
        from app.chains.rewriter import RewrittenQuery

        self.setup_complex(router, mock_vectorstore, [Document(page_content="x", metadata={})])
        mock_decompose.return_value = ["s1", "s2"]
        mock_grade.return_value = GradingResult(
            relevant_docs=[], irrelevant_count=1, has_relevant_docs=False
        )
        mock_rewrite.return_value = RewrittenQuery(rewritten_query="again", strategy="test")

        result = await run_rag_pipeline("unanswerable comparison")

        assert "couldn't find relevant information" in result["generation"]
        assert result["is_grounded"] is None
        # route 1 + decompose 4 + grade 4 + rewrite 3 = 12 LLM calls
        llm_calls = [router, mock_decompose, mock_grade, mock_rewrite]
        assert [m.call_count for m in llm_calls] == [1, 4, 4, 3]


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


class TestSelfCorrection:
    """An ungrounded answer is regenerated once; if still ungrounded it gets a caveat."""

    DOC = Document(page_content="ScaNN uses anisotropic quantization.", metadata={})

    @staticmethod
    def verdicts(*grounded):
        from app.chains.hallucination_checker import HallucinationCheck

        return [
            HallucinationCheck(
                is_grounded="yes" if g else "no",
                confidence="high",
                issues="None" if g else "The claim about pricing is unsupported",
            )
            for g in grounded
        ]

    def setup(self, mock_vectorstore, mock_grade, mock_generate):
        from app.chains.generator import GenerationResult
        from app.chains.grader import GradingResult

        mock_vectorstore.return_value.vectorstore.similarity_search.return_value = [self.DOC]
        mock_grade.return_value = GradingResult(
            relevant_docs=[self.DOC], irrelevant_count=0, has_relevant_docs=True
        )
        mock_generate.return_value = GenerationResult(
            answer="ScaNN costs $5 per month.", sources=[], has_answer=True
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.generate_strict_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_regeneration_fixes_the_answer(
        self, mock_check, mock_strict, mock_generate, mock_grade, mock_vectorstore, router
    ):
        self.setup(mock_vectorstore, mock_grade, mock_generate)
        mock_check.side_effect = self.verdicts(False, True)
        mock_strict.return_value = "ScaNN uses anisotropic quantization."

        result = await run_rag_pipeline("How does ScaNN work?")

        assert result["generation"] == "ScaNN uses anisotropic quantization."
        assert result["is_grounded"] is True
        assert result["regeneration_count"] == 1
        assert result["caveat"] is None
        mock_strict.assert_called_once_with(
            "How does ScaNN work?", [self.DOC], "The claim about pricing is unsupported"
        )
        # The re-check sees the regenerated answer
        assert mock_check.call_args_list[1].args[0] == "ScaNN uses anisotropic quantization."
        # Budget with regeneration: route, grade, generate, check, regenerate, check = 6
        llm_calls = [router, mock_grade, mock_generate, mock_check, mock_strict]
        assert [m.call_count for m in llm_calls] == [1, 1, 1, 2, 1]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.generate_strict_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_still_ungrounded_gets_a_caveat(
        self, mock_check, mock_strict, mock_generate, mock_grade, mock_vectorstore, router
    ):
        from app.agents.nodes import UNGROUNDED_CAVEAT

        self.setup(mock_vectorstore, mock_grade, mock_generate)
        mock_check.side_effect = self.verdicts(False, False)
        mock_strict.return_value = "Still unsupported answer."

        result = await run_rag_pipeline("How does ScaNN work?")

        assert result["generation"] == "Still unsupported answer."
        assert result["is_grounded"] is False
        assert result["caveat"] == UNGROUNDED_CAVEAT
        # Regenerated once only, then flagged
        assert mock_strict.call_count == 1
        assert mock_check.call_count == 2

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.generate_strict_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_failed_regeneration_fails_closed(
        self, mock_check, mock_strict, mock_generate, mock_grade, mock_vectorstore, router
    ):
        from app.errors import LLMError

        self.setup(mock_vectorstore, mock_grade, mock_generate)
        mock_check.side_effect = self.verdicts(False)
        mock_strict.side_effect = LLMError("Generation failed")

        with pytest.raises(LLMError):
            await run_rag_pipeline("How does ScaNN work?")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.agents.nodes.generate_answer")
    @patch("app.agents.nodes.generate_strict_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_failed_recheck_fails_closed(
        self, mock_check, mock_strict, mock_generate, mock_grade, mock_vectorstore, router
    ):
        from app.errors import LLMError

        self.setup(mock_vectorstore, mock_grade, mock_generate)
        mock_check.side_effect = [*self.verdicts(False), LLMError("Check failed")]
        mock_strict.return_value = "Regenerated."

        with pytest.raises(LLMError):
            await run_rag_pipeline("How does ScaNN work?")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @patch("app.agents.nodes.get_vectorstore_manager")
    @patch("app.agents.nodes.grade_documents")
    @patch("app.chains.generator.generate_answer_stream")
    @patch("app.agents.nodes.generate_strict_answer")
    @patch("app.agents.nodes.check_hallucination")
    async def test_stream_emits_regeneration_after_the_tokens(
        self, mock_check, mock_strict, mock_stream, mock_grade, mock_vectorstore, router
    ):
        from app.chains.grader import GradingResult

        mock_vectorstore.return_value.vectorstore.similarity_search.return_value = [self.DOC]
        mock_grade.return_value = GradingResult(
            relevant_docs=[self.DOC], irrelevant_count=0, has_relevant_docs=True
        )

        async def fake_stream(query, documents):
            yield "First answer."

        mock_stream.side_effect = fake_stream
        mock_check.side_effect = self.verdicts(False, True)
        mock_strict.return_value = "Grounded answer."

        events = [u async for u in run_rag_pipeline_stream_tokens("How does ScaNN work?")]

        kinds = [
            u["type"] if u["type"] != "state_update" else next(iter(u["data"])) for u in events
        ]
        token_at, regen_at = kinds.index("token"), kinds.index("regenerate")
        assert token_at < regen_at
        assert events[regen_at]["data"]["regenerate"]["generation"] == "Grounded answer."
        # The first check saw the streamed answer
        assert mock_check.call_args_list[0].args[0] == "First answer."
        assert kinds[-1] == "done"


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
