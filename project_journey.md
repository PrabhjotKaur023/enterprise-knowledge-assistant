# Project Journey: Enterprise Knowledge Assistant

## Week 1 — Setting up the foundation

**Day 1-2: Architecture planning**

Spent time sketching the overall system before writing code. Main decisions:
- FastAPI over Flask — async support is useful for I/O-bound tasks, and the automatic
  OpenAPI docs are great for a portfolio project
- FAISS over Pinecone — wanted to avoid cloud dependency for local development.
  FAISS is battle-tested and free.
- SQLite for dev — switching to Postgres in prod is one config line change with SQLAlchemy

**Day 3: Basic project scaffold**

Set up the folder structure and got a "hello world" FastAPI app running in Docker.
This took longer than expected because I needed to get the multi-stage Dockerfile right
— the first version was 1.2GB, the optimized version is ~600MB.

**Day 4-5: Database models**

Designed the Document, ChatSession, and ChatMessage tables. Initially had a single
`messages` table, but realized tracking sessions separately made the history endpoint
much cleaner.

---

## Week 2 — Document pipeline

**Day 6-7: Text extraction**

PDF extraction was straightforward with PyPDF2 but hit a snag with scanned PDFs
(no text layer — just images). Added a check for empty extraction and a clear error message.
DOCX tables were being skipped — fixed by adding explicit table parsing.

**Day 8-9: Chunking**

First attempt used fixed character-based chunking. The chunks often split sentences
mid-word which hurt embedding quality. Switched to paragraph-aware chunking with
word-boundary fallback for long paragraphs. Quality improved noticeably in smoke tests.

**Day 10: Metadata extraction**

Added word count, estimated page count, and file type to chunk metadata. The page
count estimate (word_count // 250) is rough but useful for display.

---

## Week 3 — Embeddings and RAG

**Day 11-12: Setting up FAISS**

FAISS documentation is thin on Python examples. Took time to figure out:
- IndexFlatIP with normalized vectors gives cosine similarity
- Metadata must be stored separately (FAISS only stores vectors)
- Deletion requires rebuilding the index (annoying but acceptable at this scale)

**Day 13-14: RAG pipeline**

First version called the LLM for every query regardless of retrieval quality.
Added a score threshold check — if no chunks score above 0.5, tell the user rather
than making up an answer.

Prompt engineering took most of this time. The key was being explicit about:
1. "Do not use outside knowledge" — reduces hallucination significantly
2. "Say you don't know if context is insufficient" — prevents confident wrong answers
3. Citation format — having the LLM reference [Source N] makes answers more trustworthy

**Day 15: LLM provider abstraction**

Added support for both OpenAI and Gemini with a simple provider switch. The LangChain
wrappers made this straightforward. Main gotcha: Gemini uses `system_instruction` in the
model constructor, not as a system message.

---

## Week 4 — Agents, API, and polish

**Day 16-17: Agentic layer**

LangChain agents are powerful but the ReAct format is fragile — the agent sometimes
outputs malformed responses that cause parsing errors. Added `handle_parsing_errors=True`
which helped a lot. The AgentRouter with keyword matching is a simple but effective
first pass; a proper intent classifier would be better.

**Day 18-19: API endpoints**

Upload endpoint was the trickiest — needed to handle:
- DB record creation before processing (for status tracking)
- Error handling that updates the DB record's status
- File cleanup on failure

**Day 20: Testing**

Added unit tests for chunking and RAG logic, integration tests for the API endpoints.
Mocking the embedding model and LLM was necessary to keep tests fast and free.
Coverage isn't 100% but the critical paths are covered.

**Day 21: Documentation and cleanup**

Wrote README, API docs, added comments. Removed a bunch of dead code and TODO comments
that were half-finished ideas. Realized I had three slightly different ways of handling
exceptions — consolidated to one pattern.

---

## Challenges and learnings

**Biggest technical challenge**: FAISS deletion. The FlatIndex doesn't support deletion,
so removing a document means rebuilding the entire index. Fine for hundreds of docs,
would be a real problem at millions. Solution at scale: use a filtered ANN index or
a managed vector DB.

**Biggest time sink**: Prompt engineering. Getting the LLM to stay grounded in context
and give useful answers with proper citations took way more iteration than expected.

**What I'd do differently**: Add background task processing from the start. Synchronous
document processing in the HTTP handler means large PDFs can cause timeout errors.
Celery + Redis would fix this.

**What I'm most proud of**: The chunking algorithm. It's simple, well-tested, and handles
edge cases (empty docs, oversized paragraphs, encoding issues) gracefully.
