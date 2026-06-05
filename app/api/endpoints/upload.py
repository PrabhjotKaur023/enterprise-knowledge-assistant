"""
Document upload endpoint.

Handles multipart file uploads, runs the processing pipeline,
and indexes the document for search. Processing is synchronous here —
for large files in prod, this should be moved to a background task queue
(Celery + Redis) so the HTTP request doesn't time out.

TODO: Add background task processing with progress tracking via websocket
"""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import Document, get_db
from app.pipeline.document_processor import DocumentPipeline
from app.pipeline.embedding_pipeline import get_embedding_pipeline
from app.utils.schemas import DocumentResponse, UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

upload_dir = Path(settings.UPLOAD_DIR)
upload_dir.mkdir(parents=True, exist_ok=True)


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a document (PDF, DOCX, TXT) for indexing.

    The document is saved to disk, processed through the pipeline,
    embedded, and stored in the FAISS index. Processing time depends
    on document size (~2-10s for typical documents).
    """
    document_id = str(uuid.uuid4())
    original_name = file.filename or "unnamed_document"
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Save to disk first
    safe_filename = f"{document_id}.{ext}"
    file_path = upload_dir / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")

    file_size = file_path.stat().st_size

    # Create DB record immediately so we can track status
    db_doc = Document(
        id=document_id,
        filename=safe_filename,
        original_name=original_name,
        file_type=ext,
        file_size_bytes=file_size,
        status="processing",
    )
    db.add(db_doc)
    db.commit()

    # Process and embed
    try:
        pipeline = DocumentPipeline()
        processed = pipeline.process(file_path, original_name, document_id)

        embedding_pipeline = get_embedding_pipeline()
        embedding_pipeline.index_document(processed.chunks)

        db_doc.status = "ready"
        db_doc.chunk_count = len(processed.chunks)
        db_doc.processed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Document ready: {original_name} ({len(processed.chunks)} chunks)")

        return UploadResponse(
            document_id=document_id,
            filename=original_name,
            status="ready",
            chunk_count=len(processed.chunks),
            word_count=processed.word_count,
            message="Document uploaded and indexed successfully.",
        )

    except ValueError as e:
        db_doc.status = "failed"
        db_doc.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Processing failed for {original_name}: {e}", exc_info=True)
        db_doc.status = "failed"
        db_doc.error_message = str(e)[:500]
        db.commit()
        raise HTTPException(status_code=500, detail="Document processing failed.")


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """List all uploaded documents."""
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).offset(skip).limit(limit).all()
    return [
        DocumentResponse(
            document_id=d.id,
            filename=d.original_name,
            file_type=d.file_type,
            file_size_bytes=d.file_size_bytes,
            chunk_count=d.chunk_count,
            status=d.status,
            uploaded_at=d.uploaded_at.isoformat() if d.uploaded_at else None,
        )
        for d in docs
    ]


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and remove its vectors from the index."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove from vector store
    embedding_pipeline = get_embedding_pipeline()
    removed = embedding_pipeline.remove_document(document_id)

    # Remove file from disk
    file_path = upload_dir / doc.filename
    if file_path.exists():
        file_path.unlink()

    db.delete(doc)
    db.commit()

    return {"message": f"Deleted document '{doc.original_name}' ({removed} chunks removed from index)."}
