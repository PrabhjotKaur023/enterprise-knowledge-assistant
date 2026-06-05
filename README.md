# Enterprise Knowledge Assistant
### RAG + Agentic AI Platform

A production-ready document Q&A system built with FastAPI, LangChain, FAISS, and HuggingFace embeddings. Upload documents, ask questions, get cited answers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                                                                  │
│  POST /upload    POST /chat    GET /search    GET /history       │
└──────────┬───────────┬────────────┬───────────────┬─────────────┘
           │           │            │               │
           ▼           │            ▼               ▼
  ┌─────────────┐      │   ┌──────────────┐  ┌──────────┐
  │  Document   │      │   │ Embedding    │  │  SQLite  │
  │  Pipeline   │      │   │ Pipeline     │  │ Database │
  │             │      │   │              │  │          │
  │ • Validate  │      │   │ HuggingFace  │  │ Documents│
  │ • Extract   │      │   │ Sentence     │  │ Sessions │
  │ • Clean     │─────►│   │ Transformers │  │ Messages │
  │ • Chunk     │      │   │              │  └──────────┘
  │ • Metadata  │      │   │ FAISS Index  │
  └─────────────┘      │   │ (persisted)  │
                        │   └──────┬───────┘
                        │          │ top-k chunks
                        ▼          ▼
                  ┌─────────────────────┐
                  │     RAG Engine      │
                  │                     │
                  │  Prompt Builder ──► │──► OpenAI / Gemini
                  │  Citation Support   │
                  └─────────────────────┘
                        │
                        ▼
                  ┌─────────────────────┐
                  │   Agent Router      │
                  │                     │
                  │  DocumentAgent      │
                  │  SQLAgent           │
                  │  SummaryAgent       │
                  └─────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| Document ingestion | PDF, DOCX, TXT — up to 50MB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| Vector search | FAISS FlatIP with cosine similarity |
| LLM support | OpenAI GPT-3.5/4 or Google Gemini |
| Citation | Every answer includes source chunks + scores |
| Session history | Multi-session chat with persistent history |
| Agentic layer | Document, SQL, and Summary agents via LangChain |
| Containerized | Docker + docker-compose |
| CI/CD | GitHub Actions: test + lint + Docker build |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or GEMINI_API_KEY
```

### 2. Run locally (recommended for development)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Create required directories
mkdir -p data/uploads data/faiss_index logs

# Start the server
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

### 3. Run with Docker

```bash
docker compose up --build
```

---

## API Endpoints

### Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/document.pdf"
```

### Ask a question
```bash
# Create a session first
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/chat/sessions | jq -r .session_id)

# Ask
curl -X POST http://localhost:8000/api/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main topics covered?"}'
```

### Semantic search
```bash
curl "http://localhost:8000/api/v1/search?q=machine+learning&top_k=5"
```

### Health check
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
enterprise-knowledge-assistant/
├── app/
│   ├── api/endpoints/       # FastAPI route handlers
│   │   ├── upload.py        # Document upload + management
│   │   ├── chat.py          # Chat with session management
│   │   ├── search.py        # Semantic search
│   │   ├── history.py       # Chat history
│   │   └── health.py        # Health checks
│   ├── agents/
│   │   └── agent_router.py  # DocumentAgent, SQLAgent, SummaryAgent
│   ├── core/
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   └── logging_config.py
│   ├── db/
│   │   └── database.py      # SQLAlchemy models + session
│   ├── pipeline/
│   │   ├── document_processor.py  # Validate → Extract → Chunk
│   │   ├── embedding_pipeline.py  # HuggingFace + FAISS
│   │   └── rag_engine.py          # RAG orchestrator + LLM
│   ├── utils/
│   │   └── schemas.py       # Pydantic request/response models
│   └── main.py              # FastAPI app factory
├── tests/
│   ├── unit/                # Fast, no-API-key tests
│   └── integration/         # Full HTTP layer tests
├── data/
│   ├── sample_docs/         # Example documents for testing
│   ├── uploads/             # Uploaded files (gitignored)
│   └── faiss_index/         # FAISS index files (gitignored)
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── project_journey.md
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `LLM_MODEL` | `gpt-3.5-turbo` | Model name |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `GEMINI_API_KEY` | — | Required for Gemini |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model |
| `DATABASE_URL` | SQLite | SQLAlchemy connection string |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |

---

## Tech Stack

- **FastAPI** — async Python web framework
- **LangChain** — agent orchestration
- **FAISS** — local vector similarity search
- **sentence-transformers** — local HuggingFace embeddings
- **SQLAlchemy** — ORM for SQLite/Postgres
- **PyPDF2 + python-docx** — document parsing
- **Docker** — containerization
- **pytest** — testing
- **GitHub Actions** — CI/CD
