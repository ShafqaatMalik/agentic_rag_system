"""
End-to-end tests for FastAPI endpoints.
"""

# Set environment variables before importing app
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["GOOGLE_API_KEY"] = "test-key"

from app.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_sse_exit_event():
    """sse-starlette binds a module-level exit event to the first event loop; each
    TestClient runs its own loop, so reset it between streaming tests."""
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.e2e
    @patch("app.api.main.get_vectorstore_manager")
    def test_health_check_healthy(self, mock_vectorstore, client):
        """Test health check returns healthy status."""
        mock_vs = MagicMock()
        mock_vs.get_collection_stats.return_value = {"name": "documents", "count": 10}
        mock_vectorstore.return_value = mock_vs

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["document_count"] == 10

    @pytest.mark.e2e
    @patch("app.api.main.get_vectorstore_manager")
    def test_health_check_unhealthy(self, mock_vectorstore, client):
        """Test health check returns unhealthy on error."""
        mock_vectorstore.side_effect = Exception("DB connection failed")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    async def test_query_success(self, mock_pipeline, client):
        """Test successful query."""
        from langchain_core.documents import Document

        mock_pipeline.return_value = {
            "query": "Test query",
            "documents": [
                Document(page_content="Test content", metadata={"source": "test.pdf", "page": 1})
            ],
            "documents_relevant": True,
            "generation": "This is the answer.",
            "iteration_count": 0,
            "query_type": "simple",
            "rewrite_history": [],
            "is_grounded": False,
        }

        response = client.post("/query", json={"query": "What is the answer?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the answer."
        assert data["status"] == "success"
        assert len(data["sources"]) == 1
        assert data["is_grounded"] is False
        assert data["final_query"] == "Test query"

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    def test_query_no_relevant_docs_has_no_grounding_verdict(self, mock_pipeline, client):
        """Test is_grounded is None when no answer was checked, and final_query is the rewrite."""
        mock_pipeline.return_value = {
            "query": "rewritten query",
            "documents": [],
            "documents_relevant": False,
            "generation": "I couldn't find relevant information.",
            "iteration_count": 3,
            "is_grounded": None,
            "query_type": "complex",
            "sub_queries": ["part one", "part two"],
        }

        data = client.post("/query", json={"query": "original"}).json()

        assert data["is_grounded"] is None
        assert data["final_query"] == "rewritten query"
        assert data["query_type"] == "complex"
        assert data["sub_queries"] == ["part one", "part two"]

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    async def test_query_no_relevant_docs(self, mock_pipeline, client):
        """Test query with no relevant documents."""
        mock_pipeline.return_value = {
            "query": "Unknown topic",
            "documents": [],
            "documents_relevant": False,
            "generation": "I couldn't find relevant information.",
            "iteration_count": 3,
            "query_type": "simple",
            "rewrite_history": ["q1", "q2", "q3"],
        }

        response = client.post("/query", json={"query": "Unknown topic"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_relevant_docs"
        assert data["iterations"] == 3

    @pytest.mark.e2e
    def test_query_empty_string(self, client):
        """Test query with empty string is rejected."""
        response = client.post("/query", json={"query": ""})

        assert response.status_code == 422  # Validation error

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    async def test_query_without_sources(self, mock_pipeline, client):
        """Test query with include_sources=False."""
        mock_pipeline.return_value = {
            "query": "Test",
            "documents": [],
            "documents_relevant": True,
            "generation": "Answer",
            "iteration_count": 0,
            "query_type": "simple",
            "rewrite_history": [],
        }

        response = client.post("/query", json={"query": "Test", "include_sources": False})

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []


class TestQueryStreamEndpoint:
    """Tests for /query/stream endpoint."""

    @pytest.mark.e2e
    @patch("app.agents.graph.run_rag_pipeline_stream_tokens")
    def test_stream_timing_has_total(self, mock_stream, client):
        """Test the timing event reports a numeric total from partial state updates."""

        async def fake_stream(query):
            # Like LangGraph's astream: node updates only, no start_time
            yield {"type": "state_update", "data": {"generate": {"timing": {"retrieve": 0.1}}}}
            yield {"type": "token", "data": "Answer"}
            yield {
                "type": "done",
                "data": {"check_hallucination": {"timing": {"retrieve": 0.1, "generate": 0.2}}},
            }

        mock_stream.side_effect = fake_stream

        response = client.post("/query/stream", json={"query": "Test"})

        assert response.status_code == 200
        events = [
            json.loads(line[len("data:") :])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        timing = next(e for e in events if "total_ms" in e)
        assert isinstance(timing["total_ms"], float)
        assert timing["total_ms"] >= 0
        assert timing["breakdown"] == {"retrieve": 100.0, "generate": 200.0}


class TestStreamDoneEvent:
    """Tests for the stream's done event."""

    @pytest.mark.e2e
    @patch("app.agents.graph.run_rag_pipeline_stream_tokens")
    def test_done_event_has_grounding_and_final_query(self, mock_stream, client):
        """Test the done event carries is_grounded and the rewritten query."""

        async def fake_stream(query):
            yield {"type": "state_update", "data": {"rewrite": {"query": "rewritten"}}}
            yield {"type": "state_update", "data": {"decompose": {"sub_queries": ["s1", "s2"]}}}
            yield {"type": "token", "data": "Answer"}
            yield {
                "type": "state_update",
                "data": {"check_hallucination": {"is_grounded": False, "timing": {"x": 0.1}}},
            }
            yield {
                "type": "done",
                "data": {"check_hallucination": {"is_grounded": False, "timing": {"x": 0.1}}},
            }

        mock_stream.side_effect = fake_stream

        response = client.post("/query/stream", json={"query": "original"})

        events = [
            json.loads(line[len("data:") :])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events[-1] == {
            "status": "complete",
            "is_grounded": False,
            "final_query": "rewritten",
            "sub_queries": ["s1", "s2"],
            "revised": False,
            "caveat": None,
        }


class TestSelfCorrectionResponses:
    """Revised answers and caveats in /query and the stream."""

    @staticmethod
    def stream_events(response):
        return [
            json.loads(line[len("data:") :])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]

    @pytest.mark.e2e
    @patch("app.agents.graph.run_rag_pipeline_stream_tokens")
    def test_stream_sends_revised_answer_then_caveat(self, mock_stream, client):
        """Test a regeneration sends revised_answer, and done carries revised and caveat."""

        async def fake_stream(query):
            yield {"type": "token", "data": "First answer."}
            yield {"type": "state_update", "data": {"check_hallucination": {"is_grounded": False}}}
            yield {"type": "state_update", "data": {"regenerate": {"generation": "Revised."}}}
            yield {"type": "state_update", "data": {"check_hallucination": {"is_grounded": False}}}
            yield {"type": "state_update", "data": {"flag_ungrounded": {"caveat": "Unverified."}}}
            yield {"type": "done", "data": {"flag_ungrounded": {"caveat": "Unverified."}}}

        mock_stream.side_effect = fake_stream

        events = self.stream_events(client.post("/query/stream", json={"query": "q"}))

        assert events[0] == {"content": "First answer."}
        assert events[1] == {"revised_answer": "Revised."}
        done = events[-1]
        assert done["status"] == "complete"
        assert done["revised"] is True
        assert done["caveat"] == "Unverified."
        assert done["is_grounded"] is False

    @pytest.mark.e2e
    @patch("app.agents.graph.run_rag_pipeline_stream_tokens")
    def test_stream_without_regeneration_has_no_revised_event(self, mock_stream, client):
        async def fake_stream(query):
            yield {"type": "token", "data": "Answer."}
            yield {"type": "state_update", "data": {"check_hallucination": {"is_grounded": True}}}
            yield {"type": "done", "data": {"check_hallucination": {"is_grounded": True}}}

        mock_stream.side_effect = fake_stream

        events = self.stream_events(client.post("/query/stream", json={"query": "q"}))

        assert not any("revised_answer" in e for e in events)
        assert events[-1]["revised"] is False
        assert events[-1]["caveat"] is None

    @pytest.mark.e2e
    @patch("app.agents.graph.run_rag_pipeline_stream_tokens")
    def test_error_after_first_answer_sends_error_without_sources(self, mock_stream, client):
        """Test a failed regeneration or re-check ends the stream with an error event."""
        from langchain_core.documents import Document

        from app.errors import LLMError

        async def fake_stream(query):
            doc = Document(page_content="chunk", metadata={"source": "w.pdf"})
            yield {"type": "state_update", "data": {"grade": {"documents": [doc]}}}
            yield {"type": "token", "data": "First answer."}
            raise LLMError(message="Regeneration failed message", details={})

        mock_stream.side_effect = fake_stream

        events = self.stream_events(client.post("/query/stream", json={"query": "q"}))

        assert events == [{"content": "First answer."}, {"error": "Regeneration failed message"}]

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    def test_query_returns_revised_and_caveat(self, mock_pipeline, client):
        mock_pipeline.return_value = {
            "query": "q",
            "documents": [],
            "documents_relevant": True,
            "generation": "Regenerated answer.",
            "iteration_count": 0,
            "is_grounded": False,
            "regeneration_count": 1,
            "caveat": "Unverified.",
        }

        data = client.post("/query", json={"query": "q"}).json()

        assert data["answer"] == "Regenerated answer."
        assert data["revised"] is True
        assert data["caveat"] == "Unverified."
        assert data["is_grounded"] is False


class TestQueryErrors:
    """Tests that pipeline failures are reported as errors, never as answers."""

    @staticmethod
    def llm_error(rate_limited, retry_after=None, timed_out=False):
        from app.errors import LLMError

        if rate_limited:
            message = "Rate limit message"
        elif timed_out:
            message = "Timeout message"
        else:
            message = "LLM failure message"
        return LLMError(
            message=message,
            details={
                "rate_limited": rate_limited,
                "timed_out": timed_out,
                "retry_after": retry_after,
            },
        )

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    def test_query_rate_limited(self, mock_pipeline, client):
        """Test a rate limit returns 429, Retry-After and status error."""
        mock_pipeline.side_effect = self.llm_error(rate_limited=True, retry_after=37.2)

        response = client.post("/query", json={"query": "Test"})

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "38"
        data = response.json()
        assert data["status"] == "error"
        assert data["answer"] == "Rate limit message"
        assert data["sources"] == []

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    def test_query_timeout(self, mock_pipeline, client):
        """Test a timed-out LLM call returns 504 and status error."""
        mock_pipeline.side_effect = self.llm_error(rate_limited=False, timed_out=True)

        response = client.post("/query", json={"query": "Test"})

        assert response.status_code == 504
        data = response.json()
        assert data["status"] == "error"
        assert data["answer"] == "Timeout message"
        assert data["sources"] == []

    @pytest.mark.e2e
    @patch("app.agents.graph.run_rag_pipeline_stream_tokens")
    def test_stream_timeout_error_event(self, mock_stream, client):
        """Test a timed-out LLM call ends the stream with the timeout message."""
        error = self.llm_error(rate_limited=False, timed_out=True)

        async def failing_stream(query):
            raise error
            yield  # pragma: no cover

        mock_stream.side_effect = failing_stream

        response = client.post("/query/stream", json={"query": "Test"})

        events = [
            json.loads(line[len("data:") :])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events == [{"error": "Timeout message"}]

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    def test_query_llm_failure(self, mock_pipeline, client):
        """Test another LLM failure returns 502 and status error."""
        mock_pipeline.side_effect = self.llm_error(rate_limited=False)

        response = client.post("/query", json={"query": "Test"})

        assert response.status_code == 502
        assert response.json()["status"] == "error"
        assert response.json()["answer"] == "LLM failure message"

    @pytest.mark.e2e
    @patch("app.api.main.run_rag_pipeline")
    def test_query_unexpected_failure_hides_internals(self, mock_pipeline, client):
        """Test an unexpected failure returns 500 with a generic message."""
        mock_pipeline.side_effect = RuntimeError("chroma exploded at /secret/path")

        response = client.post("/query", json={"query": "Test"})

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "secret" not in data["answer"]

    @pytest.mark.e2e
    @patch("app.agents.graph.run_rag_pipeline_stream_tokens")
    def test_stream_error_event_without_sources(self, mock_stream, client):
        """Test a failure mid-pipeline sends one error event and no sources or timing."""
        from langchain_core.documents import Document

        error = self.llm_error(rate_limited=True, retry_after=30)

        async def failing_stream(query):
            doc = Document(page_content="Retrieved", metadata={"source": "doc.pdf"})
            yield {"type": "state_update", "data": {"retrieve": {"documents": [doc]}}}
            raise error

        mock_stream.side_effect = failing_stream

        response = client.post("/query/stream", json={"query": "Test"})

        events = [
            json.loads(line[len("data:") :])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events == [{"error": "Rate limit message"}]


class TestIngestEndpoint:
    """Tests for /ingest endpoints."""

    @pytest.mark.e2e
    def test_ingest_unsupported_file_type(self, client):
        """Test rejection of unsupported file types."""
        from io import BytesIO

        response = client.post(
            "/ingest/file",
            files={"file": ("test.exe", BytesIO(b"content"), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    @pytest.mark.e2e
    @patch("app.api.main.get_vectorstore_manager")
    def test_ingest_txt_file(self, mock_vectorstore, client):
        """Test ingestion of text file."""
        from io import BytesIO

        mock_vs = MagicMock()
        mock_vs.ingest_file = AsyncMock(return_value=["id1", "id2"])
        mock_vectorstore.return_value = mock_vs

        response = client.post(
            "/ingest/file", files={"file": ("test.txt", BytesIO(b"Test content"), "text/plain")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_added"] == 2


class TestCollectionEndpoints:
    """Tests for collection management endpoints."""

    @pytest.mark.e2e
    @patch("app.api.main.get_vectorstore_manager")
    def test_get_collection_stats(self, mock_vectorstore, client):
        """Test getting collection statistics."""
        mock_vs = MagicMock()
        mock_vs.get_collection_stats.return_value = {"name": "documents", "count": 100}
        mock_vectorstore.return_value = mock_vs

        response = client.get("/collection/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "documents"
        assert data["document_count"] == 100

    @pytest.mark.e2e
    @patch("app.api.main.get_vectorstore_manager")
    def test_clear_collection(self, mock_vectorstore, client):
        """Test clearing collection."""
        mock_vs = MagicMock()
        mock_vectorstore.return_value = mock_vs

        response = client.delete("/collection")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_vs.clear_collection.assert_called_once()


class TestGraphVisualization:
    """Tests for graph visualization endpoint."""

    @pytest.mark.e2e
    @patch("app.api.main.get_graph_visualization")
    def test_get_graph_visualization(self, mock_viz, client):
        """Test getting graph visualization."""
        mock_viz.return_value = "graph TD\n  A --> B"

        response = client.get("/graph/visualization")

        assert response.status_code == 200
        data = response.json()
        assert "mermaid" in data
        assert "graph" in data["mermaid"]
