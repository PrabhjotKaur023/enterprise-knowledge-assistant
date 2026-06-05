"""
Chat endpoint.

Main entry point for Q&A. Manages sessions, calls the RAG engine,
persists messages, and returns structured responses with citations.
"""

import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import ChatMessage, ChatSession, get_db
from app.pipeline.rag_engine import get_rag_engine
from app.utils.schemas import (
    ChatRequest,
    ChatResponse,
    NewSessionResponse,
    SourceCitation,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/sessions", response_model=NewSessionResponse)
def create_session(db: Session = Depends(get_db)):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    session = ChatSession(id=session_id)
    db.add(session)
    db.commit()
    return NewSessionResponse(session_id=session_id)


@router.post("/chat/{session_id}", response_model=ChatResponse)
def chat(
    session_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Send a message and get a RAG-powered response.

    If the session doesn't exist, it's created automatically.
    This is more user-friendly than requiring an explicit session creation step.
    """
    # Ensure session exists
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        session = ChatSession(id=session_id)
        db.add(session)
        db.commit()

    # Auto-title the session from first message
    if session.title is None:
        session.title = request.message[:60] + ("..." if len(request.message) > 60 else "")
        db.commit()

    # Save user message
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    db.commit()

    # Get answer from RAG
    try:
        rag_engine = get_rag_engine()
        result = rag_engine.answer(
            query=request.message,
            top_k=request.top_k or settings.TOP_K_RESULTS,
            document_filter=request.document_ids or None,
        )
    except Exception as e:
        logger.error(f"RAG engine failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate response.")

    # Build citations
    citations = [
        SourceCitation(
            chunk_id=s.chunk_id,
            document_id=s.document_id,
            filename=s.filename,
            excerpt=s.content[:300] + "..." if len(s.content) > 300 else s.content,
            relevance_score=round(s.score, 4),
        )
        for s in result.sources
    ]

    # Save assistant message with source metadata
    assistant_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=result.answer,
        sources=json.dumps([c.dict() for c in citations]),
        document_ids=json.dumps([s.document_id for s in result.sources]),
        latency_ms=result.latency_ms,
    )
    db.add(assistant_msg)

    session.updated_at = datetime.utcnow()
    db.commit()

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        sources=citations,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )
