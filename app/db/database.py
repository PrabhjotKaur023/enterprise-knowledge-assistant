"""
Database setup using SQLAlchemy.

Chose SQLite for dev — zero config, single file, works everywhere.
Swap DATABASE_URL to postgres:// in prod; SQLAlchemy handles both.
"""

import logging
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.core.config import settings

logger = logging.getLogger(__name__)

# connect_args is SQLite-specific; not needed for Postgres
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,  # prints SQL in debug mode — helpful but verbose
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    """Tracks every uploaded document and its processing status."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # UUID
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending, processing, ready, failed
    error_message = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)


class ChatSession(Base):
    """Groups chat messages into sessions so history makes sense."""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)  # UUID
    title = Column(String, nullable=True)  # auto-generated from first message
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMessage(Base):
    """Individual messages within a session."""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    # Sources retrieved for this response (stored as JSON string)
    sources = Column(Text, nullable=True)
    # Track which document this was answered from
    document_ids = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create all tables. Safe to call multiple times — won't drop existing data."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for getting a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
