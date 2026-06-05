# Architecture & Design Decisions

## 1. Why FastAPI over Flask?

FastAPI's automatic request validation via Pydantic means bugs surface immediately with
clear error messages rather than at runtime deep in business logic. The auto-generated
OpenAPI docs (`/docs`) are a practical bonus — no manual documentation for API consumers.
Async support matters less here (the bottleneck is LLM latency, not I/O), but it's
available if needed.

## 2. Local embeddings (HuggingFace) instead of OpenAI embeddings

`all-MiniLM-L6-v2` runs on CPU in ~50ms per batch and costs nothing. For most document
Q&A use cases, the quality difference versus `text-embedding-ada-002` is small. Keeping
embeddings local also means the project works without any API key — useful for demos
and for avoiding per-call costs during development.

Tradeoff: embedding quality is lower for highly technical or multilingual text.
Switch to `all-mpnet-base-v2` or OpenAI embeddings if quality becomes an issue.

## 3. FAISS FlatIP index

FAISS IndexFlatIP (Inner Product) with L2-normalized vectors gives exact cosine similarity.
"Flat" means brute-force search — O(n) per query but perfectly accurate.

For this project's scale (thousands of documents, millions of chunks), this is fine.
At 10M+ vectors, switch to IVFFlat or HNSW for approximate search with much lower latency.

Deletion limitation: FlatIndex doesn't support removing individual vectors.
Our workaround — rebuilding the index without the deleted document's vectors — is correct
but O(n). Acceptable for infrequent deletes at this scale.

## 4. SQLite for default storage

One file, zero configuration, works everywhere. SQLAlchemy abstracts it, so switching
to Postgres is one config change (`DATABASE_URL`). Postgres is the right choice for
production (concurrent writes, better query planner, JSON operators).

## 5. Paragraph-aware chunking with word-boundary fallback

Fixed character chunking is the simplest approach but often cuts sentences mid-word,
degrading embedding quality. Paragraph-aware chunking respects natural document
structure. The word-boundary fallback handles paragraphs that exceed the chunk size.

Chunk size (512 chars) and overlap (64 chars) were chosen empirically. Larger chunks
= more context per retrieval but dilute the semantic signal. Smaller chunks = more
precise retrieval but less context. 512/64 is a reasonable starting point.

## 6. Synchronous document processing

Document processing happens inline in the HTTP request handler. This is simple but
means large PDFs block the request thread and can cause timeouts.

The right production solution is async job processing: store the file, return a job ID
immediately, process in a background worker (Celery + Redis), and let the client poll
for completion. This was a deliberate scope decision — out-of-scope for v1.

## 7. AgentRouter keyword matching instead of intent classification

Simple and fast. The keyword list covers the obvious cases (database queries vs.
document questions). A proper solution would train a small intent classifier or use
embedding similarity to route queries. Added as a TODO — it's a clear improvement
without being a blocker.

## 8. LLM provider abstraction

`LLMClient` wraps both OpenAI and Gemini behind a single `.complete()` method.
LangChain wrappers (`ChatOpenAI`, `ChatGoogleGenerativeAI`) are used in the agent layer.
The abstraction is thin on purpose — deep abstractions make debugging harder.

## 9. Module-level singletons for expensive objects

The embedding model (~80MB), FAISS index, and RAG engine are initialized once and
reused across requests via module-level `_instance` variables with `get_*()` functions.
This avoids re-loading models on every request (which would be extremely slow).

In production, these would be initialized during app startup via FastAPI's lifespan
event and injected via dependency injection.

## 10. Separate Pydantic schemas from SQLAlchemy models

DB models describe storage. API schemas describe the HTTP contract. Keeping them
separate prevents accidentally exposing internal fields (like `error_message` on
Document) and makes the API contract explicit and stable independent of DB changes.
