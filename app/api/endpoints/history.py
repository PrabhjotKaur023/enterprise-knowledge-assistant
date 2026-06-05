"""Chat history endpoint."""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import ChatMessage, ChatSession, get_db
from app.utils.schemas import SessionHistoryResponse, MessageResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/chat/sessions", response_model=list[dict])
def list_sessions(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    """List all chat sessions, most recent first."""
    sessions = (
        db.query(ChatSession)
        .order_by(ChatSession.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        {
            "session_id": s.id,
            "title": s.title or "Untitled",
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in sessions
    ]


@router.get("/chat/sessions/{session_id}/history", response_model=SessionHistoryResponse)
def get_session_history(session_id: str, db: Session = Depends(get_db)):
    """Get all messages in a chat session."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return SessionHistoryResponse(
        session_id=session_id,
        title=session.title or "Untitled",
        messages=[
            MessageResponse(
                message_id=m.id,
                role=m.role,
                content=m.content,
                sources=json.loads(m.sources) if m.sources else [],
                created_at=m.created_at.isoformat() if m.created_at else None,
                latency_ms=m.latency_ms,
            )
            for m in messages
        ],
    )


@router.delete("/chat/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a session and all its messages."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.delete(session)
    db.commit()
    return {"message": "Session deleted."}
