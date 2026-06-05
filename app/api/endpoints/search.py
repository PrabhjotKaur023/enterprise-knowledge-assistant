"""Search endpoint — semantic search without LLM generation."""

import logging
from fastapi import APIRouter, Query
from app.pipeline.embedding_pipeline import get_embedding_pipeline
from app.utils.schemas import SearchResponse, SearchResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search", response_model=SearchResponse)
def semantic_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """
    Perform semantic search over indexed documents.
    Returns ranked chunks without calling the LLM — faster and cheaper
    for cases where you just want to find relevant passages.
    """
    if not q.strip():
        return SearchResponse(query=q, results=[], total=0)

    pipeline = get_embedding_pipeline()
    raw_results = pipeline.search(q, top_k=top_k)

    results = [
        SearchResult(
            chunk_id=meta["chunk_id"],
            document_id=meta["document_id"],
            filename=meta["metadata"].get("filename", "Unknown"),
            content=meta["content"],
            score=round(score, 4),
            chunk_index=meta["metadata"].get("chunk_index", 0),
        )
        for meta, score in raw_results
    ]

    return SearchResponse(query=q, results=results, total=len(results))
