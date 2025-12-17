"""API module for FastAPI endpoints."""

from app.api.main import app
from app.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    IngestFileResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    CollectionStats,
    ClearCollectionResponse,
    StreamEvent
)

__all__ = [
    "app",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "IngestFileResponse",
    "QueryRequest",
    "QueryResponse",
    "SourceDocument",
    "CollectionStats",
    "ClearCollectionResponse",
    "StreamEvent"
]
