"""
Pydantic schemas for API request/response validation.

Keeping these separate from DB models is a good habit — it avoids
accidentally exposing internal fields and makes API contracts explicit.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---- Upload ----

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    word_count: int
    message: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    status: str
    uploaded_at: Optional[str] = None


# ---- Chat ----

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: Optional[int] = Field(default=4, ge=1, le=20)
    document_ids: Optional[List[str]] = Field(default=None, description="Filter to specific documents")


class SourceCitation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    excerpt: str
    relevance_score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceCitation]
    latency_ms: float
    model_used: str


class NewSessionResponse(BaseModel):
    session_id: str


# ---- Search ----

class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content: str
    score: float
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int


# ---- History ----

class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    sources: List[Dict[str, Any]] = []
    created_at: Optional[str] = None
    latency_ms: Optional[float] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    title: str
    messages: List[MessageResponse]
