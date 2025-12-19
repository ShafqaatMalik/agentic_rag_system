"""
FastAPI application for Agentic RAG system.

Provides REST API endpoints for document ingestion, querying,
and system management.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app import __version__
from app.agents.graph import get_graph_visualization, run_rag_pipeline
from app.api.schemas import (
    ClearCollectionResponse,
    CollectionStats,
    DeleteDocumentResponse,
    DocumentsListResponse,
    HealthResponse,
    IngestFileResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from app.config import get_settings
from app.logging_config import setup_logging
from app.retrieval.vectorstore import get_vectorstore_manager

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    setup_logging()
    logger.info("Starting Agentic RAG API", version=__version__)

    # Initialize vectorstore
    try:
        vectorstore = get_vectorstore_manager()
        stats = vectorstore.get_collection_stats()
        logger.info("Vectorstore initialized", **stats)
    except Exception as e:
        logger.error("Failed to initialize vectorstore", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down Agentic RAG API")


# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="A lightweight, production-ready Retrieval-Augmented Generation system with self-correction capabilities.",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# --- Root Endpoint ---


@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Agentic RAG API", "docs": "/docs"}


# --- Health Endpoints ---


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health and vectorstore status.
    """
    try:
        vectorstore = get_vectorstore_manager()
        stats = vectorstore.get_collection_stats()

        return HealthResponse(
            status="healthy",
            version=__version__,
            vectorstore_status="connected",
            document_count=stats["count"],
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy", version=__version__, vectorstore_status=f"error: {str(e)}"
        )


# --- Ingest Endpoints ---


@app.post("/ingest/file", response_model=IngestFileResponse, tags=["Ingest"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a single document file.

    Supported formats: PDF, TXT, MD
    """
    import os
    import tempfile

    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Ingest with original filename
        vectorstore = get_vectorstore_manager()
        ids = await vectorstore.ingest_file(tmp_path, original_filename=file.filename)

        # Cleanup
        os.unlink(tmp_path)

        logger.info("File ingested", filename=file.filename, chunks=len(ids))

        return IngestFileResponse(filename=file.filename, chunks_added=len(ids), status="success")

    except Exception as e:
        logger.error("File ingestion failed", filename=file.filename, error=str(e))
        return IngestFileResponse(
            filename=file.filename, chunks_added=0, status="failed", error=str(e)
        )


@app.post("/ingest/directory", response_model=IngestResponse, tags=["Ingest"])
async def ingest_directory(request: IngestRequest):
    """
    Ingest all supported documents from a directory.
    """
    if not request.directory_path:
        raise HTTPException(status_code=400, detail="directory_path is required")

    try:
        vectorstore = get_vectorstore_manager()
        results = await vectorstore.ingest_directory(request.directory_path)

        return IngestResponse(
            success=results["success"],
            failed=results["failed"],
            total_files=len(results["success"]) + len(results["failed"]),
            message=f"Ingested {len(results['success'])} files successfully",
        )

    except NotADirectoryError:
        raise HTTPException(status_code=400, detail="Path is not a directory") from None
    except Exception as e:
        logger.error("Directory ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Query Endpoints ---


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the RAG system.

    Runs the complete agentic pipeline:
    1. Route query
    2. Retrieve documents
    3. Grade relevance
    4. Generate answer (with optional rewrite loop)
    5. Check for hallucinations
    """
    import time

    logger.info("Query received", query=request.query[:50])

    try:
        # Run the RAG pipeline
        final_state = await run_rag_pipeline(request.query)

        # Calculate total latency
        start_time = final_state.get("start_time")
        total_latency_ms = None
        if start_time:
            total_latency_ms = (time.time() - start_time) * 1000

        # Format timing breakdown (convert seconds to milliseconds)
        latency_breakdown = {}
        timing_data = final_state.get("timing", {})
        for node_name, duration_sec in timing_data.items():
            latency_breakdown[node_name] = round(duration_sec * 1000, 2)

        # Extract sources if requested
        sources = []
        if request.include_sources and final_state.get("documents"):
            for doc in final_state["documents"]:
                sources.append(
                    SourceDocument(
                        content=doc.page_content[:500],  # Truncate for response
                        source=doc.metadata.get("source", "Unknown"),
                        page=doc.metadata.get("page"),
                    )
                )

        # Determine status
        status = "success"
        if not final_state.get("documents_relevant", False):
            status = "no_relevant_docs"

        return QueryResponse(
            answer=final_state.get("generation", "No answer generated"),
            sources=sources,
            query_type=final_state.get("query_type"),
            iterations=final_state.get("iteration_count", 0),
            status=status,
            latency_ms=round(total_latency_ms, 2) if total_latency_ms else None,
            latency_breakdown=latency_breakdown,
        )

    except Exception as e:
        logger.error("Query failed", query=request.query[:50], error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response.

    Returns Server-Sent Events (SSE) with:
    - token: Generated answer tokens
    - source: Source documents used
    - done: Completion signal
    - error: Error information
    """
    logger.info("Streaming query received", query=request.query[:50])

    async def event_generator():
        import time

        from app.agents.graph import run_rag_pipeline_stream_tokens

        try:
            final_state = None
            documents = []
            full_answer = ""

            # Stream through the pipeline with token-by-token generation
            async for update in run_rag_pipeline_stream_tokens(request.query):
                update_type = update.get("type")
                update_data = update.get("data")

                if update_type == "token":
                    # Stream individual tokens
                    full_answer += update_data
                    yield {"event": "token", "data": json.dumps({"content": update_data})}
                elif update_type == "state_update":
                    # Capture documents and state for later use
                    for _node_name, node_state in update_data.items():
                        if isinstance(node_state, dict) and node_state.get("documents"):
                            documents = node_state["documents"]
                    final_state = update_data
                elif update_type == "done":
                    final_state = update_data

            # Send sources
            if request.include_sources and documents:
                for doc in documents:
                    yield {
                        "event": "source",
                        "data": json.dumps(
                            {
                                "content": doc.page_content[:200],
                                "source": doc.metadata.get("source", "Unknown"),
                            }
                        ),
                    }

            # Send timing information
            if final_state:
                # Extract timing from the final state
                for node_state in final_state.values():
                    if isinstance(node_state, dict):
                        start_time = node_state.get("start_time")
                        timing_data = node_state.get("timing", {})

                        if timing_data:
                            # Calculate total latency
                            total_latency_ms = None
                            if start_time:
                                total_latency_ms = round((time.time() - start_time) * 1000, 2)

                            # Format timing breakdown
                            latency_breakdown = {}
                            for node_name, duration_sec in timing_data.items():
                                latency_breakdown[node_name] = round(duration_sec * 1000, 2)

                            yield {
                                "event": "timing",
                                "data": json.dumps(
                                    {"total_ms": total_latency_ms, "breakdown": latency_breakdown}
                                ),
                            }
                            break

            # Send completion
            yield {"event": "done", "data": json.dumps({"status": "complete"})}

        except Exception as e:
            logger.error("Streaming query failed", error=str(e))
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())


# --- Collection Management ---


@app.get("/collection/stats", response_model=CollectionStats, tags=["Collection"])
async def get_collection_stats():
    """
    Get statistics about the document collection.
    """
    try:
        vectorstore = get_vectorstore_manager()
        stats = vectorstore.get_collection_stats()

        return CollectionStats(name=stats["name"], document_count=stats["count"])
    except Exception as e:
        logger.error("Failed to get collection stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collection/documents", response_model=DocumentsListResponse, tags=["Collection"])
async def get_documents_list():
    """
    Get list of all documents in the collection.

    Returns information about each unique document including:
    - Document name
    - File path
    - Number of chunks/pages
    """
    try:
        vectorstore = get_vectorstore_manager()
        documents = vectorstore.get_documents_list()

        return DocumentsListResponse(documents=documents, total_count=len(documents))
    except Exception as e:
        logger.error("Failed to get documents list", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete(
    "/collection/document/{document_name}",
    response_model=DeleteDocumentResponse,
    tags=["Collection"],
)
async def delete_document(document_name: str):
    """
    Delete a specific document from the collection.

    All chunks associated with the document will be removed.
    """
    try:
        vectorstore = get_vectorstore_manager()
        chunks_deleted = vectorstore.delete_document(document_name)

        if chunks_deleted == 0:
            return DeleteDocumentResponse(
                status="failed",
                document_name=document_name,
                chunks_deleted=0,
                message=f"Document '{document_name}' not found",
            )

        return DeleteDocumentResponse(
            status="success",
            document_name=document_name,
            chunks_deleted=chunks_deleted,
            message=f"Successfully deleted {chunks_deleted} chunks",
        )
    except Exception as e:
        logger.error("Failed to delete document", document=document_name, error=str(e))
        return DeleteDocumentResponse(
            status="failed", document_name=document_name, chunks_deleted=0, message=str(e)
        )


@app.delete("/collection", response_model=ClearCollectionResponse, tags=["Collection"])
async def clear_collection():
    """
    Clear all documents from the collection.

    ⚠️ This action is irreversible!
    """
    try:
        vectorstore = get_vectorstore_manager()
        vectorstore.clear_collection()

        return ClearCollectionResponse(status="success", message="Collection cleared successfully")
    except Exception as e:
        logger.error("Failed to clear collection", error=str(e))
        return ClearCollectionResponse(status="failed", message=str(e))


# --- Debug Endpoints ---


@app.get("/graph/visualization", tags=["Debug"])
async def get_graph_viz():
    """
    Get Mermaid diagram of the RAG workflow graph.
    """
    return {"mermaid": get_graph_visualization()}


# --- Main ---

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
