"""
Application configuration.

Using pydantic-settings for environment variable validation. This caught
several bugs during dev where I forgot to set a required variable — much
better than finding out at runtime when a real request comes in.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Enterprise Knowledge Assistant"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # API Keys — at least one LLM provider must be set
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Default LLM provider: "openai" or "gemini"
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-3.5-turbo"

    # Embeddings — using HuggingFace locally so we don't burn API credits
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector store
    FAISS_INDEX_PATH: str = "data/faiss_index"

    # Database
    DATABASE_URL: str = "sqlite:///./data/knowledge_assistant.db"

    # File upload limits
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt"]
    UPLOAD_DIR: str = "data/uploads"

    # RAG settings — tuned these through trial and error
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RESULTS: int = 4

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cache settings so we don't re-read .env on every request."""
    return Settings()


settings = get_settings()
