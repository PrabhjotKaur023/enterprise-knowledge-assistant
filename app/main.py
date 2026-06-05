"""
Main FastAPI application entry point.

Started this project to learn RAG pipelines properly. The basic structure
here follows what I've seen in production codebases — lifespan for startup/shutdown,
routers organized by feature, middleware added as needed.
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import upload, chat, search, history, health
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.database import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    setup_logging()
    logger.info("Starting Enterprise Knowledge Assistant...")

    # Initialize database tables
    init_db()
    logger.info("Database initialized.")

    # TODO: pre-warm the embedding model on startup to reduce first-request latency
    # This matters in prod where cold starts are painful

    yield

    logger.info("Shutting down Knowledge Assistant.")


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG + Agentic AI Platform for document Q&A",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — tighten this in prod, localhost is fine for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Log request duration. Useful for catching slow endpoints early."""
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration)
    logger.debug(f"{request.method} {request.url.path} - {duration}ms")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all so we never leak stack traces to clients."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(history.router, prefix="/api/v1", tags=["History"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
