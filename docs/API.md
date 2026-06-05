# API Documentation

Base URL: `http://localhost:8000`

Interactive docs available at `/docs` (Swagger UI) and `/redoc`.

---

## Health

### `GET /health`
Liveness check.

**Response 200:**
```json
{"status": "ok", "environment": "development"}
```

### `GET /health/ready`
Readiness check — verifies core dependencies.

**Response 200:**
```json
{
  "status": "ready",
  "checks": {
    "vector_store": {"status": "ok", "total_chunks": 42},
    "llm": {"status": "ok", "provider": "openai"}
  }
}
```

---

## Documents

### `POST /api/v1/documents/upload`
Upload a document for indexing.

**Request:** `multipart/form-data`
- `file`: File to upload (PDF, DOCX, TXT — max 50MB)

**Response 200:**
```json
{
  "document_id": "3f4a1b2c-...",
  "filename": "report.pdf",
  "status": "ready",
  "chunk_count": 28,
  "word_count": 4200,
  "message": "Document uploaded and indexed successfully."
}
```

**Errors:**
- `400` — Unsupported file type
- `422` — Processing error (empty/unreadable document)
- `500` — Internal error

---

### `GET /api/v1/documents`
List all uploaded documents.

**Query params:**
- `skip` (int, default 0)
- `limit` (int, default 50)

**Response 200:**
```json
[
  {
    "document_id": "3f4a1b2c-...",
    "filename": "report.pdf",
    "file_type": "pdf",
    "file_size_bytes": 245760,
    "chunk_count": 28,
    "status": "ready",
    "uploaded_at": "2024-06-05T10:30:00"
  }
]
```

---

### `DELETE /api/v1/documents/{document_id}`
Delete a document and remove its vectors.

**Response 200:**
```json
{"message": "Deleted document 'report.pdf' (28 chunks removed from index)."}
```

---

## Chat

### `POST /api/v1/chat/sessions`
Create a new chat session.

**Response 200:**
```json
{"session_id": "a1b2c3d4-..."}
```

---

### `POST /api/v1/chat/{session_id}`
Send a message and get a RAG-powered response.

**Request body:**
```json
{
  "message": "What are the key findings?",
  "top_k": 4,
  "document_ids": ["3f4a1b2c-..."]
}
```

- `top_k`: Number of chunks to retrieve (1-20, default 4)
- `document_ids`: Filter retrieval to specific documents (optional)

**Response 200:**
```json
{
  "session_id": "a1b2c3d4-...",
  "answer": "The key findings include...",
  "sources": [
    {
      "chunk_id": "3f4a1b2c-..._chunk_3",
      "document_id": "3f4a1b2c-...",
      "filename": "report.pdf",
      "excerpt": "Revenue grew by 20% year-over-year...",
      "relevance_score": 0.8821
    }
  ],
  "latency_ms": 1240.5,
  "model_used": "openai/gpt-3.5-turbo"
}
```

---

## Search

### `GET /api/v1/search`
Semantic search without LLM generation — faster and cheaper.

**Query params:**
- `q` (required): Search query
- `top_k` (int, default 5, max 20): Number of results

**Response 200:**
```json
{
  "query": "revenue growth",
  "results": [
    {
      "chunk_id": "..._chunk_3",
      "document_id": "3f4a1b2c-...",
      "filename": "report.pdf",
      "content": "Revenue grew by 20%...",
      "score": 0.8821,
      "chunk_index": 3
    }
  ],
  "total": 1
}
```

---

## History

### `GET /api/v1/chat/sessions`
List all chat sessions.

**Response 200:**
```json
[
  {
    "session_id": "a1b2c3d4-...",
    "title": "What are the key findings?",
    "created_at": "2024-06-05T10:35:00",
    "updated_at": "2024-06-05T10:38:00"
  }
]
```

---

### `GET /api/v1/chat/sessions/{session_id}/history`
Get all messages in a session.

**Response 200:**
```json
{
  "session_id": "a1b2c3d4-...",
  "title": "What are the key findings?",
  "messages": [
    {
      "message_id": "...",
      "role": "user",
      "content": "What are the key findings?",
      "sources": [],
      "created_at": "2024-06-05T10:35:00",
      "latency_ms": null
    },
    {
      "message_id": "...",
      "role": "assistant",
      "content": "The key findings include...",
      "sources": [...],
      "created_at": "2024-06-05T10:35:01",
      "latency_ms": 1240.5
    }
  ]
}
```

---

### `DELETE /api/v1/chat/sessions/{session_id}`
Delete a session and all its messages.
