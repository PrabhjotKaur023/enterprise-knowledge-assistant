"""
Integration tests for FastAPI endpoints.

Using TestClient so we can test the full HTTP layer without spinning up
a real server. LLM and embeddings are mocked.
"""

import io
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import init_db


@pytest.fixture(scope="module")
def client():
    # Use in-memory SQLite for tests
    with patch("app.core.config.settings.DATABASE_URL", "sqlite:///./test.db"):
        init_db()
        with TestClient(app) as c:
            yield c
    import os
    if os.path.exists("test.db"):
        os.remove("test.db")


class TestHealthEndpoints:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readiness_returns_status(self, client):
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert "status" in r.json()


class TestUploadEndpoints:
    def test_upload_txt_file(self, client):
        with patch("app.api.endpoints.upload.DocumentPipeline") as MockPipeline, \
             patch("app.api.endpoints.upload.get_embedding_pipeline") as mock_ep:

            mock_processed = MagicMock()
            mock_processed.chunks = [MagicMock()] * 3
            mock_processed.word_count = 150
            MockPipeline.return_value.process.return_value = mock_processed
            mock_ep.return_value.index_document = MagicMock()

            content = b"This is a test document with some content."
            r = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test_doc.txt", io.BytesIO(content), "text/plain")},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ready"
            assert data["chunk_count"] == 3

    def test_upload_unsupported_format_rejected(self, client):
        content = b"binary content"
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("virus.exe", io.BytesIO(content), "application/octet-stream")},
        )
        assert r.status_code == 400
        assert "Unsupported" in r.json()["detail"]

    def test_list_documents(self, client):
        r = client.get("/api/v1/documents")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestChatEndpoints:
    def test_create_session(self, client):
        r = client.post("/api/v1/chat/sessions")
        assert r.status_code == 200
        assert "session_id" in r.json()

    def test_chat_returns_response(self, client):
        with patch("app.api.endpoints.chat.get_rag_engine") as mock_engine:
            mock_result = MagicMock()
            mock_result.answer = "The answer is 42."
            mock_result.sources = []
            mock_result.latency_ms = 120.0
            mock_result.model_used = "openai/gpt-3.5-turbo"
            mock_engine.return_value.answer.return_value = mock_result

            # Create session first
            session_r = client.post("/api/v1/chat/sessions")
            session_id = session_r.json()["session_id"]

            r = client.post(
                f"/api/v1/chat/{session_id}",
                json={"message": "What is the meaning of life?"},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["answer"] == "The answer is 42."
            assert data["session_id"] == session_id

    def test_empty_message_rejected(self, client):
        r = client.post(
            "/api/v1/chat/nonexistent-session",
            json={"message": ""},
        )
        assert r.status_code == 422  # pydantic validation error


class TestSearchEndpoints:
    def test_search_returns_results(self, client):
        with patch("app.api.endpoints.search.get_embedding_pipeline") as mock_ep:
            mock_ep.return_value.search.return_value = [
                (
                    {
                        "chunk_id": "c1",
                        "document_id": "d1",
                        "content": "Some content",
                        "metadata": {"filename": "doc.pdf", "chunk_index": 0},
                    },
                    0.87,
                )
            ]
            r = client.get("/api/v1/search?q=test+query")
            assert r.status_code == 200
            data = r.json()
            assert data["total"] == 1
            assert data["results"][0]["score"] == 0.87

    def test_empty_query_returns_empty(self, client):
        r = client.get("/api/v1/search?q=")
        assert r.status_code == 200
        assert r.json()["total"] == 0
