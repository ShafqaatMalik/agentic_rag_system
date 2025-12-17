# Agentic RAG System

A production-ready Agentic RAG system that autonomously improves retrieval through multi-iteration query rewriting, validates generated answers using hallucination detection, and orchestrates complex decision flows via LangGraph’s state-based architecture—going beyond traditional linear RAG pipelines.The system supports streaming responses, includes comprehensive evaluation metrics, and is fully containerized with Docker for scalable deployment.

## Features

- **Intelligent Query Routing**: Classifies queries and routes to appropriate processing paths
- **Self-Correcting Retrieval**: Grades document relevance and rewrites queries when needed (up to 3 attempts)
- **Hallucination Detection**: Validates generated answers against retrieved context
- **Streaming Responses**: Real-time response streaming via Server-Sent Events (SSE)
- **RAG Evaluation**: Built-in metrics for faithfulness, relevance, precision, and recall
- **Production Ready**: Comprehensive testing, Docker support, CI/CD pipeline, structured logging

## Architecture

```
┌──────────────────────────────────┐
│   LangGraph (Flow Control)       │  ← StateGraph, conditional edges, nodes
├──────────────────────────────────┤
│   LangChain (Logic Layer)        │  ← Prompts, LLM chains, retrievers
├──────────────────────────────────┤
│   Infrastructure (DB, LLM, APIs) │  ← ChromaDB, Gemini, FastAPI
└──────────────────────────────────┘
```

## Workflow

```
START → Router → Retriever → Grader → [Decision]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
              (docs relevant)                          (docs not relevant)
                    │                                         │
                    ▼                                         ▼
               Generator                              Query Rewriter
                    │                                         │
                    ▼                                         └──→ back to Retriever
           Hallucination Check                                    (max 3 times)
                    │
                    ▼
                  END
```

## Quick Start

### Prerequisites

- Python 3.11+
- Google API Key (for Gemini)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic-rag
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Running the Application

**Development:**
```bash
uvicorn app.api.main:app --reload
```

**Docker:**
```bash
docker-compose up --build
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ingest/file` | POST | Ingest single document (PDF, TXT, MD) |
| `/ingest/directory` | POST | Ingest all documents from directory |
| `/query` | POST | Query the system |
| `/query/stream` | POST | Query with streaming response (SSE) |
| `/collection/stats` | GET | Get collection statistics |
| `/collection/documents` | GET | Get list of documents in collection |
| `/collection/document/{document_name}` | DELETE | Delete specific document |
| `/collection` | DELETE | Clear all documents |
| `/graph/visualization` | GET | Get Mermaid diagram of workflow |

### API Usage Examples

**Query the system:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How is AI used in healthcare?"}'
```

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/ingest/file" \
  -F "file=@document.pdf"
```

**Stream response:**
```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the revenue trends"}'
```

## Testing

| Marker | Purpose | Run Command |
|--------|---------|---
----------|
| `unit` | Component-level tests | `pytest -m unit` |
| `integration` | Chain/graph flow tests | `pytest -m integration` |
| `e2e` | Full pipeline tests | `pytest -m e2e` |
| `evaluation` | RAG metrics tests | `pytest -m evaluation` |

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## Project Structure

```
agentic-rag/
├── app/
│   ├── agents/          # LangGraph workflow
│   │   ├── graph.py     # Workflow definition
│   │   ├── state.py     # State schema
│   │   └── nodes.py     # Node functions
│   ├── chains/          # LangChain components
│   │   ├── router.py    # Query routing
│   │   ├── grader.py    # Document grading
│   │   ├── generator.py # Answer generation
│   │   └── rewriter.py  # Query rewriting
│   ├── retrieval/       # Vector store
│   ├── api/             # FastAPI endpoints
│   └── config.py        # Configuration
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   ├── e2e/             # End-to-end tests
│   └── evaluation/      # RAG evaluation tests
├── evaluation/          # Evaluation datasets
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Google API key for Gemini |
| `LLM_MODEL` | `gemini-1.5-flash` | LLM model to use |
| `RETRIEVAL_K` | `4` | Number of documents to retrieve |
| `MAX_REWRITE_ITERATIONS` | `3` | Max query rewrite attempts |

## RAG Evaluation

The system includes built-in evaluation metrics:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevance** | Does the answer address the user's query? |
| **Context Precision** | Are the retrieved documents relevant? |
| **Context Recall** | Did we retrieve all important documents? |

## CI/CD Pipeline

The project includes a comprehensive GitHub Actions pipeline:

- ✅ Linting (Ruff, Black, isort)
- ✅ Unit Tests
- ✅ Integration Tests
- ✅ E2E Tests
- ✅ RAG Evaluation Tests
- ✅ Code Coverage
- ✅ Docker Build
- ✅ Security Scan

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangGraph |
| **LLM Framework** | LangChain |
| **Vector Store** | ChromaDB |
| **LLM** | Google Gemini |
| **API** | FastAPI |
| **Testing** | pytest, ragas |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |

## License

MIT
