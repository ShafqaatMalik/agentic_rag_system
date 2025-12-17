"""
API schemas for request/response models.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# --- Health ---

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"]
    version: str
    vectorstore_status: Optional[str] = None
    document_count: Optional[int] = None


# --- Ingest ---

class IngestRequest(BaseModel):
    """Request to ingest documents from a directory."""
    directory_path: Optional[str] = Field(
        default=None,
        description="Path to directory containing documents to ingest"
    )


class IngestFileResponse(BaseModel):
    """Response for single file ingestion."""
    filename: str
    chunks_added: int
    status: Literal["success", "failed"]
    error: Optional[str] = None


class IngestResponse(BaseModel):
    """Response for document ingestion."""
    success: List[str]
    failed: List[dict]
    total_files: int
    message: str


# --- Query ---

class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source documents in response"
    )


class SourceDocument(BaseModel):
    """A source document used in the response."""
    content: str
    source: str
    page: Optional[int] = None


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    sources: List[SourceDocument]
    query_type: Optional[str] = None
    iterations: int
    status: Literal["success", "no_relevant_docs", "error"]
    latency_ms: Optional[float] = Field(None, description="Total latency in milliseconds")
    latency_breakdown: Optional[dict] = Field(None, description="Latency breakdown by component")


class StreamEvent(BaseModel):
    """Server-sent event for streaming responses."""
    event: Literal["token", "source", "done", "error", "timing"]
    data: str


# --- Collection ---

class CollectionStats(BaseModel):
    """Statistics about the document collection."""
    name: str
    document_count: int


class DocumentInfo(BaseModel):
    """Information about a single document."""
    name: str
    path: str
    page_count: int


class DocumentsListResponse(BaseModel):
    """Response containing list of documents."""
    documents: List[DocumentInfo]
    total_count: int


class ClearCollectionResponse(BaseModel):
    """Response for clearing collection."""
    status: Literal["success", "failed"]
    message: str


class DeleteDocumentResponse(BaseModel):
    """Response for deleting a document."""
    status: Literal["success", "failed"]
    document_name: str
    chunks_deleted: int
    message: str
