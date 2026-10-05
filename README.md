# Agentic RAG System

A production-ready Agentic RAG system that autonomously improves retrieval through multi-iteration query rewriting, flags ungrounded answers with a hallucination check, and orchestrates complex decision flows via LangGraph’s state-based architecture—going beyond traditional linear RAG pipelines. The system supports streaming responses, includes comprehensive evaluation metrics, and is fully containerized with Docker for scalable deployment.

## Features

- **Query Routing**: A router labels a question complex only when it compares things or has clearly separate parts; everything else is simple. Complex questions are decomposed into 2–3 sub-queries, and the question itself plus each sub-query is retrieved separately; the results are merged, de-duplicated, capped at 12 and graded together
- **Self-Correcting Retrieval**: Grades all retrieved chunks for relevance in one LLM call and rewrites the query when needed (up to 3 attempts)
- **Self-Correcting Answers**: Checks each answer against the retrieved context. An ungrounded answer is regenerated once with a stricter grounding prompt and checked again; if it is still ungrounded it is returned with `is_grounded=false` and a visible caveat
- **Streaming Responses**: Real-time response streaming via Server-Sent Events (SSE)
- **RAG Evaluation**: Built-in metrics for faithfulness, relevance, precision, and recall
- **Production Ready**: Comprehensive testing, Docker support, CI/CD pipeline, structured logging
- **Containerised**: Runs locally via Docker; containerised and Cloud Run-deployable

## Screenshots

### Welcome Screen
![Welcome Screen](docs/screenshots/01-welcome.png)

### Document Upload
![Document Upload](docs/screenshots/02-upload.png)

### Successful Upload
![Successful Upload](docs/screenshots/03-successful%20upload.png)

### Query & Response
![Query Response](docs/screenshots/04-query%20response.png)

### Source Attribution
![Sources](docs/screenshots/05-sources.png)

### Response Latency
![Latency](docs/screenshots/06-latency.png)

### Want to Test It Yourself?

See the [Quick Start](#quick-start) section below to run it locally with Docker.

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
START → Router (simple or complex)
          ├── simple ──→ Retriever (k=4)
          └── complex ─→ Decomposer (2–3 sub-queries) → Retriever (k=4 for the question + each sub-query; merged, de-duplicated, ≤12)
                                        │
                                        ▼
                       Grader (one call for all chunks) → [Decision]
                                        │
          ┌─────────────────────────────┼──────────────────────────────┐
          ▼                             ▼                              ▼
   (docs relevant)          (not relevant, rewrites left)   (not relevant, max rewrites reached)
          │                             │                              │
          ▼                             ▼                              ▼
      Generator                  Query Rewriter               No Relevant Documents
          │                             │                       (fallback message)
          ▼                             └──→ back to the start of its path    │
  Hallucination Check                        (simple: Retriever,              ▼
  (with self-correction, below)               complex: Decomposer;           END
          │                                   up to 3 rewrites)
          ▼
         END
```

Self-correction after generation:

```
Generator → Hallucination Check
              ├── grounded ──────────────────────────────────────────→ END
              └── ungrounded → Regenerate (strict grounding prompt) → Hallucination Check
                                                                        ├── grounded ──────────→ END (revised)
                                                                        └── still ungrounded ──→ END (is_grounded=false + caveat)
```

The router labels a question complex only when it compares or contrasts named things, or asks clearly separate questions; single-topic questions are simple, even when they ask how something works. Simple questions get a single retrieval. Complex questions are decomposed into 2–3 self-contained sub-queries; the original question and each sub-query are retrieved separately (k=4), so decomposition can't lose the question's key terms. The results are merged by rank (the question's own hits first), de-duplicated and capped at 12 chunks: up to 16 candidates, keeping ranks 1–3 of every query. The grader judges all chunks in one call; on the complex path it also sees the sub-queries, and a chunk that helps answer any one of them counts as relevant. A rewritten query goes back to the start of its path, so complex questions are decomposed again.

The hallucination check runs on the answer the user actually receives. If the answer isn't grounded in the retrieved context, it is regenerated once with a stricter prompt that includes the checker's list of unsupported claims, and the new answer is checked again. If that is still ungrounded, it is returned with `is_grounded=false` and a `caveat`. When streaming, the first answer arrives as tokens and a regenerated answer arrives as one `revised` event (`{"revised_answer": "..."}`) that replaces it; the `done` event carries `is_grounded`, `revised` and `caveat`, and `/query` returns the same fields. Responses also include `query_type`, `sub_queries` (complex path) and `final_query`, the query used for the last retrieval after any rewrites. If the grader still finds no relevant documents after `MAX_REWRITE_ITERATIONS` rewrites, the pipeline ends with a "no relevant documents" message instead of generating an answer.

LLM calls per query:

| Path | Answered, grounded first time | With regeneration | Each rewrite adds | Worst case answered (3 rewrites + regeneration) | No relevant documents (3 rewrites) |
|---|---|---|---|---|---|
| Simple | 4 (route, grade, generate, check) | 6 (+ regenerate, check) | 2 (rewrite, grade) | 12 | 8 |
| Complex | 5 (route, decompose, grade, generate, check) | 7 | 3 (rewrite, decompose, grade) | 16 | 12 |

Embedding calls (separate model and quota, not rate-limited): 1 per simple retrieval; 3–4 per complex retrieval (the question plus each sub-query), and the same again after each rewrite.

## Rate Limits and Failures

The defaults suit the Gemini free tier (15 requests per minute per model). All LLM calls share a client-side rate limiter (0.2 requests/s with a burst of 3), so the first three calls of a query start immediately and later calls queue (about 5 seconds each) instead of hitting 429 errors. Each LLM request times out after 30 seconds. A 429, a 503 or a timeout is retried once (after the wait the server suggests for a 429, as long as that is 40 seconds or less); no other layer retries silently.

Failures are never hidden: a failed grading or hallucination check is never treated as relevant or grounded, and a failed regeneration never falls back to the first answer. If a call still fails, `/query` returns `status: "error"` with a clear message (HTTP 429 with `Retry-After` for rate limits, 504 for timeouts), and `/query/stream` sends an `error` event with no sources.

## Quick Start

The app runs locally via Docker; the image is containerised and Cloud Run-deployable.

**Prerequisites:**
- Docker Desktop (or Docker Engine with the Compose plugin)
- A Google API key for Gemini (the free tier works)

1. Clone the repository:
```bash
git clone https://github.com/ShafqaatMalik/agentic_rag_system.git
cd agentic_rag_system
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY
```

3. Start the app:
```bash
docker compose up --build
```

4. Ingest documents (PDF, TXT, MD), either by uploading them in the web UI, or via the API:
```bash
# A single file
curl -X POST "http://localhost:8000/ingest/file" -F "file=@document.pdf"

# Everything in ./data (mounted into the container at /app/data)
curl -X POST "http://localhost:8000/ingest/directory" \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/app/data"}'
```

5. Open [http://localhost:8000](http://localhost:8000) and start asking questions.

Ingested documents are stored in the `chroma_data` Docker volume and survive restarts. `docker compose down` keeps them; `docker compose down -v` deletes them.

**Running without Docker (development):**
```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
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

The suite has 203 tests; LLM and embedding calls are mocked, so no API key is needed.

| Marker | Purpose | Run Command |
|--------|---------|-------------|
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
agentic_rag_system/
├── .github/workflows/ci.yml     # CI: lint, tests, coverage, Docker build, security scan
├── app/
│   ├── agents/
│   │   ├── graph.py             # LangGraph workflow definition
│   │   ├── nodes.py             # Node functions and conditional edges
│   │   └── state.py             # Agent state schema
│   ├── api/
│   │   ├── main.py              # FastAPI app, endpoints, SSE streaming
│   │   └── schemas.py           # Request/response models
│   ├── chains/
│   │   ├── decomposer.py        # Splits complex questions into sub-queries
│   │   ├── generator.py         # Answer generation
│   │   ├── grader.py            # Batch document relevance grading
│   │   ├── hallucination_checker.py  # Groundedness and answer-relevance checks
│   │   ├── rewriter.py          # Query rewriting
│   │   └── router.py            # Query classification (simple/complex)
│   ├── retrieval/
│   │   └── vectorstore.py       # ChromaDB ingestion and retrieval
│   ├── config.py                # Settings from environment variables
│   ├── errors.py                # Custom exceptions and error handling
│   ├── llm.py                   # Gemini LLM setup
│   └── logging_config.py        # structlog configuration
├── frontend/
│   ├── index.html               # Chat UI served at /
│   ├── app.js                   # Chat, upload, streaming and latency display
│   └── styles.css
├── docs/screenshots/            # README screenshots
├── data/                        # Documents to ingest (mounted at /app/data)
├── evaluation/
│   ├── eval_dataset.json        # Small dataset used by the mocked evaluation tests
│   ├── whitepaper_eval.json     # Live evaluation set: 20 in-scope + 5 out-of-scope questions
│   └── run_eval.py              # Live evaluation against the running app (Gemini judge)
├── tests/
│   ├── unit/                    # Chain-level tests
│   ├── integration/             # Graph flow and rewrite-loop tests
│   ├── e2e/                     # API and full-pipeline tests
│   ├── evaluation/              # RAG metrics tests
│   └── conftest.py              # Shared fixtures
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml               # Ruff, Black, isort, pytest, coverage config
├── pytest.ini
└── requirements.txt
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Google API key for Gemini |
| `LLM_MODEL` | `gemini-flash-lite-latest` | LLM model to use |
| `LLM_TEMPERATURE` | `0.0` | LLM sampling temperature |
| `LLM_REQUESTS_PER_SECOND` | `0.2` | Client-side rate limit shared by all LLM calls |
| `LLM_MAX_BURST` | `3` | Calls allowed back to back before the rate limit applies |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-request timeout for LLM calls |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model for ChromaDB |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `RETRIEVAL_K` | `4` | Number of documents to retrieve |
| `MAX_REWRITE_ITERATIONS` | `3` | Max query rewrite attempts |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## RAG Evaluation

The system includes built-in evaluation metrics:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevance** | Does the answer address the user's query? |
| **Context Precision** | Are the retrieved documents relevant? |
| **Context Recall** | Did we retrieve all important documents? |

### Live evaluation

`evaluation/run_eval.py` evaluates the running app against `evaluation/whitepaper_eval.json` (20 questions with verbatim supporting quotes from the whitepaper, plus 5 out-of-scope questions that should end in "no relevant documents"):

```bash
docker compose up -d
python3 evaluation/run_eval.py --judge-model gemini-3.5-flash
```

It sends each question to `/query`, recreates each answer's final retrieval with the app's own code, and scores in-scope answers with a separate Gemini judge (temperature 0): correctness against the reference, faithfulness, answer relevance, context recall and precision, refusal accuracy, and agreement with the app's `is_grounded`. Out-of-scope questions need no judge request. A 503 from the judge (model overloaded) is retried up to 2 times, 60 seconds apart; a 429 stops judging at once. Every request actually sent counts toward `--max-judge-requests` (default 20, the judge model's free-tier daily quota), and questions left unjudged are judged on the next run with the same `--out`. Results go to `evaluation/results/<date>-<commit>/` (`results.jsonl`, `summary.md`).

To split a run across two days, run `--skip-judge` first: it asks the app all 25 questions and computes everything that doesn't need the judge (context recall, router labels, refusals, latency) with no judge requests; later, run again with the same `--out` to judge the saved answers without querying the app again.

The evaluation sends questions back to back, so its latency includes rate-limiter queueing. To measure single-user latency, run `python3 evaluation/run_eval.py --latency-probe --out <the run's directory>`: it sends 5 questions, each after 60 seconds of idle time, makes no judge requests, and writes `latency_probe.md` comparing single-user latency with the same questions under evaluation load.

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
