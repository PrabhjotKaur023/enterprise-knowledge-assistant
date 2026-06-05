# Interview Prep

## Resume Bullet Points

1. **Built a production-ready RAG platform** using FastAPI, LangChain, FAISS, and HuggingFace
   sentence-transformers; supports multi-format document ingestion (PDF/DOCX/TXT), semantic
   search, and cited Q&A with configurable OpenAI/Gemini backends.

2. **Designed and implemented a 4-stage document processing pipeline** — validation, multi-format
   text extraction, paragraph-aware chunking with overlap, and metadata enrichment — handling
   edge cases including scanned PDFs, mixed encodings, and oversized paragraphs.

3. **Engineered a persistent FAISS vector store** with cosine similarity retrieval; built document
   deletion support via index reconstruction and designed the metadata store to enable citation
   tracking back to source document and chunk position.

4. **Implemented a multi-agent AI system** with LangChain ReAct agents (DocumentAgent, SQLAgent,
   SummaryAgent) and an intent-based router; designed prompt engineering that reduces hallucination
   by grounding LLM responses strictly in retrieved document context.

5. **Containerized and CI/CD enabled** the full application with Docker multi-stage builds,
   docker-compose for local orchestration, and GitHub Actions pipeline covering unit tests,
   integration tests, linting, and Docker build verification.

---

## Interview Questions

### RAG & ML System Design

**Q: What is RAG and why is it useful?**
A: RAG (Retrieval-Augmented Generation) combines an LLM with a retrieval step over external documents.
It solves two LLM limitations: knowledge cutoff (training data is fixed in time) and hallucination
(models generate confident but wrong answers). By fetching relevant context at inference time, the
model can answer questions about current documents it was never trained on, with responses grounded
in verifiable sources.

**Q: How does your chunking strategy work and why did you choose it?**
A: I use paragraph-aware chunking: split on double newlines first, accumulate paragraphs until the
chunk size limit, then use overlap (64 chars) so context isn't lost at boundaries. For paragraphs
that exceed the chunk size, I fall back to word-boundary splitting. This beats fixed character
chunking because it respects natural document structure, which improves embedding quality.

**Q: How do you handle the FAISS deletion problem?**
A: FAISS FlatIndex doesn't support removing individual vectors. When a document is deleted, I
rebuild the entire index by extracting all remaining vectors with `reconstruct()`, creating a
new index, and adding them back. This is O(n) but correct. For large-scale production, the right
approach is to use a vector DB that supports deletion (Pinecone, Weaviate) or maintain a
"deleted IDs" filter list and exclude them during search.

**Q: Why use cosine similarity for semantic search?**
A: Cosine similarity measures the angle between vectors, ignoring magnitude. This is better than
Euclidean distance for text embeddings because the magnitude of an embedding often reflects text
length rather than semantic content. Two texts with similar meaning should have similar angles
regardless of their lengths. With L2-normalized vectors, inner product (IndexFlatIP) gives cosine
similarity efficiently.

**Q: How did you prevent hallucination in your RAG system?**
A: Three main techniques: (1) System prompt explicitly instructs the model to answer ONLY from
the provided context and say "I don't have information" when context is insufficient. (2) Low
temperature (0.2) for factual tasks reduces creative but inaccurate outputs. (3) Citations — the
model is asked to reference [Source N] so users can verify claims against the original chunks.

---

### System Design

**Q: How would you scale this system to handle 100 GB of documents?**
A: Several changes: (1) Move from FAISS FlatIndex to HNSW or IVFFlat for approximate nearest
neighbor search — exact search doesn't scale. (2) Move to a managed vector DB (Pinecone, Weaviate)
with horizontal scaling. (3) Add an async job queue (Celery + Redis) for document processing so
HTTP handlers don't block. (4) Cache frequently asked queries (Redis) to avoid repeated LLM calls.
(5) Consider PostgreSQL with pgvector for simpler ops if the scale doesn't require dedicated VDB.

**Q: Why is your document processing synchronous? What would you change?**
A: It's synchronous for simplicity — acceptable for this project's scope. In production, large
PDFs can take 10-30s to process, which would cause HTTP timeouts. The fix is: (1) accept the file,
save to disk, create a DB record with status=pending, return a 202 Accepted with a job ID.
(2) Process in a Celery worker. (3) Client polls `/documents/{id}` or receives a webhook when ready.

**Q: What's the difference between your search endpoint and chat endpoint?**
A: `/search` does retrieval only — embeds the query, finds similar chunks, returns ranked results.
No LLM involved, so it's fast (~50ms) and cheap. `/chat` does full RAG — retrieval + LLM generation.
Use `/search` when you want to browse relevant passages; use `/chat` when you want a synthesized answer.

---

### Python & Backend

**Q: Why use pydantic-settings for configuration?**
A: It validates env variables at startup with type checking — a missing required variable or wrong
type raises an error immediately, not when the first request uses it. The `lru_cache` on
`get_settings()` ensures the `.env` file is only read once.

**Q: What is a FastAPI dependency and how do you use it here?**
A: Dependencies are functions that FastAPI calls before your route handler, injecting their return
value. I use `get_db` as a dependency to provide a per-request database session, ensuring the
session is properly closed after each request even if the handler raises an exception.

**Q: Why use SQLAlchemy's `declarative_base` over raw SQL?**
A: ORM benefits: type-safe model definitions, Python-level relationships, database-agnostic code
(SQLite in dev → Postgres in prod = one config change), and automatic schema management via
`create_all()`. For complex analytics queries I'd drop to raw SQL via `db.execute()`, but CRUD
operations on simple tables are cleaner with the ORM.
