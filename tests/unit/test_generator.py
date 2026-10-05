"""
Unit tests for Generator Chain.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.chains.generator import (
    GENERATOR_SYSTEM_PROMPT,
    NO_CONTEXT_PROMPT,
    GenerationResult,
    extract_sources,
    format_documents,
    generate_answer,
    get_generator_chain,
    get_no_context_chain,
)
from app.errors import LLMError


class TestGenerationResult:
    """Tests for GenerationResult schema."""

    def test_generation_result_with_answer(self):
        """Test creating a generation result with answer."""
        result = GenerationResult(
            answer="AI is used for diagnosis.", sources=["doc1.pdf"], has_answer=True
        )
        assert result.has_answer is True
        assert len(result.sources) == 1

    def test_generation_result_no_answer(self):
        """Test creating a generation result without answer."""
        result = GenerationResult(
            answer="No relevant information found.", sources=[], has_answer=False
        )
        assert result.has_answer is False
        assert len(result.sources) == 0


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.unit
    def test_format_documents(self):
        """Test document formatting."""
        docs = [
            Document(page_content="Content 1", metadata={"source": "doc1.pdf"}),
            Document(page_content="Content 2", metadata={"source": "doc2.pdf"}),
        ]

        result = format_documents(docs)

        assert "Content 1" in result
        assert "Content 2" in result
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    @pytest.mark.unit
    def test_format_documents_empty(self):
        """Test formatting empty document list."""
        result = format_documents([])
        assert result == "No relevant documents found."

    @pytest.mark.unit
    def test_extract_sources(self):
        """Test extracting sources from documents."""
        docs = [
            Document(page_content="Content", metadata={"source": "doc1.pdf"}),
            Document(page_content="Content", metadata={"source": "doc2.pdf"}),
            Document(page_content="Content", metadata={}),  # No source
        ]

        sources = extract_sources(docs)

        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources
        assert "Unknown" in sources  # Third doc has no source, gets "Unknown"
        assert len(sources) == 3

    @pytest.mark.unit
    def test_extract_sources_empty(self):
        """Test extracting sources from empty list."""
        sources = extract_sources([])
        assert sources == []


class TestGeneratorChain:
    """Tests for generator chain functionality."""

    @pytest.mark.unit
    def test_generator_system_prompt_content(self):
        """Test that system prompt contains key instructions."""
        assert "answer" in GENERATOR_SYSTEM_PROMPT.lower()
        assert "context" in GENERATOR_SYSTEM_PROMPT.lower()

    @pytest.mark.unit
    def test_no_context_prompt_content(self):
        """Test that no-context prompt contains key instructions."""
        assert (
            "no relevant" in NO_CONTEXT_PROMPT.lower() or "don't have" in NO_CONTEXT_PROMPT.lower()
        )
        assert (
            "knowledge base" in NO_CONTEXT_PROMPT.lower()
            or "information" in NO_CONTEXT_PROMPT.lower()
        )

    @pytest.mark.unit
    @patch("app.chains.generator.get_llm")
    def test_get_generator_chain(self, mock_get_llm):
        """Test generator chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_generator_chain()

        mock_get_llm.assert_called_once()
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.generator.get_llm")
    def test_get_no_context_chain(self, mock_get_llm):
        """Test no-context chain creation."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        chain = get_no_context_chain()

        mock_get_llm.assert_called_once()
        assert chain is not None

    @pytest.mark.unit
    @patch("app.chains.generator.get_generator_chain")
    def test_generate_answer_with_documents(self, mock_get_chain):
        """Test generating answer with documents."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "AI is used for faster diagnosis."
        mock_get_chain.return_value = mock_chain

        docs = [
            Document(
                page_content="AI improves healthcare diagnostics",
                metadata={"source": "healthcare.pdf"},
            )
        ]

        result = generate_answer("How is AI used in healthcare?", docs)

        assert result.has_answer is True
        assert "AI" in result.answer or "diagnosis" in result.answer
        assert len(result.sources) == 1
        assert "healthcare.pdf" in result.sources
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.generator.get_no_context_chain")
    def test_generate_answer_no_documents(self, mock_get_chain):
        """Test generating answer without documents."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "I apologize, but I couldn't find relevant information."
        mock_get_chain.return_value = mock_chain

        result = generate_answer("Test query", [])

        assert result.has_answer is False
        assert len(result.sources) == 0
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.generator.get_generator_chain")
    def test_generate_answer_raises_on_error(self, mock_get_chain):
        """Test a failed generation raises instead of returning an apology answer."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        docs = [Document(page_content="Test", metadata={"source": "test.pdf"})]

        with pytest.raises(LLMError) as exc_info:
            generate_answer("Test query", docs)

        assert "no answer could be produced" in exc_info.value.message


class TestStrictRegeneration:
    """Tests for regenerating an ungrounded answer with the strict grounding prompt."""

    @pytest.mark.unit
    def test_strict_prompt_demands_grounding(self):
        from app.chains.generator import STRICT_GENERATOR_SYSTEM_PROMPT

        prompt = STRICT_GENERATOR_SYSTEM_PROMPT.lower()
        assert "only facts that the context directly supports" in prompt
        assert "don't cover" in prompt
        assert "synthesize" not in prompt

    @pytest.mark.unit
    @patch("app.chains.generator.get_strict_generator_chain")
    def test_passes_the_checker_issues(self, mock_get_chain):
        from app.chains.generator import generate_strict_answer

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Strictly grounded answer."
        mock_get_chain.return_value = mock_chain
        docs = [Document(page_content="Context text", metadata={"source": "w.pdf"})]

        answer = generate_strict_answer("Q?", docs, "Claims about pricing are unsupported")

        assert answer == "Strictly grounded answer."
        inputs = mock_chain.invoke.call_args.args[0]
        assert inputs["issues"] == "Claims about pricing are unsupported"
        assert inputs["query"] == "Q?"
        assert "Context text" in inputs["context"]
        mock_chain.invoke.assert_called_once()

    @pytest.mark.unit
    @patch("app.chains.generator.get_llm")
    def test_issues_reach_the_model_prompt(self, mock_get_llm):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        from app.chains.generator import generate_strict_answer

        seen = []

        def respond(prompt_value):
            seen.append(prompt_value.to_messages())
            return AIMessage(content="ok")

        mock_get_llm.return_value = RunnableLambda(respond)

        generate_strict_answer("Q?", [Document(page_content="C")], "The year 2019 is unsupported")

        system, human = seen[0]
        assert "directly supports" in system.content
        assert "The year 2019 is unsupported" in human.content

    @pytest.mark.unit
    @patch("app.chains.generator.get_strict_generator_chain")
    def test_fails_closed(self, mock_get_chain):
        from app.chains.generator import generate_strict_answer

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_chain.return_value = mock_chain

        with pytest.raises(LLMError):
            generate_strict_answer("Q?", [Document(page_content="C")], "issues")
