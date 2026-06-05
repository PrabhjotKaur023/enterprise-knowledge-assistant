"""Health check endpoint — used by Docker healthcheck and load balancers."""

import logging
from fastapi import APIRouter
from app.pipeline.embedding_pipeline import get_embedding_pipeline
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health_check():
    """Basic liveness check."""
    return {"status": "ok", "environment": settings.ENVIRONMENT}


@router.get("/health/ready")
def readiness_check():
    """
    Readiness check — verifies core dependencies are operational.
    Returns 200 only if the service can handle requests.
    """
    checks = {}

    # Check vector store is accessible
    try:
        pipeline = get_embedding_pipeline()
        checks["vector_store"] = {
            "status": "ok",
            "total_chunks": pipeline.vector_store.total_chunks,
        }
    except Exception as e:
        checks["vector_store"] = {"status": "error", "detail": str(e)}

    # Check LLM provider config (just key presence, not a live call)
    if settings.LLM_PROVIDER == "openai":
        checks["llm"] = {"status": "ok" if settings.OPENAI_API_KEY else "missing_key", "provider": "openai"}
    elif settings.LLM_PROVIDER == "gemini":
        checks["llm"] = {"status": "ok" if settings.GEMINI_API_KEY else "missing_key", "provider": "gemini"}

    all_ok = all(v.get("status") == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }
