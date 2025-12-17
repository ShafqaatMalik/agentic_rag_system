"""
End-to-end tests for FastAPI endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Set environment variables before importing app
import os
os.environ["GOOGLE_API_KEY"] = "test-key"

from app.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    @pytest.mark.e2e
    @patch("app.api.main.get_vectorstore_manager")
    def test_health_check_healthy(self, mock_vectorstore, client):
        """Test health check returns healthy status."""
        mock_vs = MagicMock()
        mock_vs.get_collection_stats.return_value = {
            "name": "documents",
            "count": 10
        }
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
                Document(
                    page_content="Test content",
                    metadata={"source": "test.pdf", "page": 1}
                )
            ],
            "documents_relevant": True,
            "generation": "This is the answer.",
            "iteration_count": 0,
            "query_type": "simple",
            "rewrite_history": []
        }
        
        response = client.post(
            "/query",
            json={"query": "What is the answer?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the answer."
        assert data["status"] == "success"
        assert len(data["sources"]) == 1
    
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
            "rewrite_history": ["q1", "q2", "q3"]
        }
        
        response = client.post(
            "/query",
            json={"query": "Unknown topic"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_relevant_docs"
        assert data["iterations"] == 3
    
    @pytest.mark.e2e
    def test_query_empty_string(self, client):
        """Test query with empty string is rejected."""
        response = client.post(
            "/query",
            json={"query": ""}
        )
        
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
            "rewrite_history": []
        }
        
        response = client.post(
            "/query",
            json={"query": "Test", "include_sources": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []


class TestIngestEndpoint:
    """Tests for /ingest endpoints."""
    
    @pytest.mark.e2e
    def test_ingest_unsupported_file_type(self, client):
        """Test rejection of unsupported file types."""
        from io import BytesIO
        
        response = client.post(
            "/ingest/file",
            files={"file": ("test.exe", BytesIO(b"content"), "application/octet-stream")}
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
            "/ingest/file",
            files={"file": ("test.txt", BytesIO(b"Test content"), "text/plain")}
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
        mock_vs.get_collection_stats.return_value = {
            "name": "documents",
            "count": 100
        }
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
