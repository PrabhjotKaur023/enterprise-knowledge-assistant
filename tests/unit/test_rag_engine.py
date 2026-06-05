"""
Tests for the RAG engine.

Mocking out the LLM and embedding pipeline so these run without
API keys or GPU. Testing the logic, not the models.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.pipeline.rag_engine import RAGEngine, PromptBuilder, Source, RAGResponse


class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder()

    def test_builds_prompt_with_sources(self):
        sources = [
            Source(
                chunk_id="c1",
                document_id="d1",
                filename="report.pdf",
                content="Revenue grew by 20% in Q3.",
                score=0.92,
                chunk_index=0,
            )
        ]
        system, user = self.builder.build_rag_prompt("What was the revenue growth?", sources)
        assert "report.pdf" in user
        assert "Revenue grew" in user
        assert "What was the revenue growth?" in user

    def test_builds_prompt_with_no_sources(self):
        system, user = self.builder.build_rag_prompt("Some question", [])
        assert "No relevant documents" in user

    def test_system_prompt_instructs_grounding(self):
        system, _ = self.builder.build_rag_prompt("q", [])
        # System prompt should enforce staying in context
        assert "context" in system.lower()
        assert "don't have enough information" in system.lower() or "not" in system.lower()

    def test_multiple_sources_all_included(self):
        sources = [
            Source("c1", "d1", "a.pdf", "Content A", 0.9, 0),
            Source("c2", "d2", "b.pdf", "Content B", 0.8, 0),
            Source("c3", "d3", "c.pdf", "Content C", 0.7, 0),
        ]
        _, user = self.builder.build_rag_prompt("question", sources)
        assert "[Source 1]" in user
        assert "[Source 2]" in user
        assert "[Source 3]" in user


class TestRAGEngine:
    def _make_engine_with_mocks(self):
        """Create RAG engine with mocked dependencies."""
        engine = RAGEngine.__new__(RAGEngine)

        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = [
            (
                {
                    "chunk_id": "chunk_1",
                    "document_id": "doc_1",
                    "content": "This is relevant content about the topic.",
                    "metadata": {"filename": "test.pdf", "chunk_index": 0},
                },
                0.88,
            )
        ]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Based on the documents, the answer is X."

        engine.embedding_pipeline = mock_pipeline
        engine.llm = mock_llm
        engine.prompt_builder = PromptBuilder()
        return engine

    def test_answer_returns_rag_response(self):
        engine = self._make_engine_with_mocks()
        result = engine.answer("What is the topic?")
        assert isinstance(result, RAGResponse)
        assert result.answer == "Based on the documents, the answer is X."
        assert len(result.sources) == 1

    def test_answer_includes_source_metadata(self):
        engine = self._make_engine_with_mocks()
        result = engine.answer("query")
        source = result.sources[0]
        assert source.document_id == "doc_1"
        assert source.filename == "test.pdf"
        assert source.score == 0.88

    def test_answer_with_document_filter(self):
        engine = self._make_engine_with_mocks()
        # Filter to a doc that doesn't match the mock result
        result = engine.answer("query", document_filter=["other_doc"])
        assert len(result.sources) == 0  # filtered out

    def test_answer_with_matching_document_filter(self):
        engine = self._make_engine_with_mocks()
        result = engine.answer("query", document_filter=["doc_1"])
        assert len(result.sources) == 1

    def test_llm_failure_returns_graceful_response(self):
        engine = self._make_engine_with_mocks()
        engine.llm.complete.side_effect = Exception("API error")
        result = engine.answer("query")
        assert result.answer  # Should not be empty
        assert "error" in result.answer.lower()

    def test_empty_vector_store_returns_response(self):
        engine = self._make_engine_with_mocks()
        engine.embedding_pipeline.search.return_value = []
        engine.llm.complete.return_value = "I don't have information on this."
        result = engine.answer("unknown query")
        assert result.sources == []
        assert result.answer

    def test_latency_is_tracked(self):
        engine = self._make_engine_with_mocks()
        result = engine.answer("test")
        assert result.latency_ms > 0

    def test_model_used_field_populated(self):
        engine = self._make_engine_with_mocks()
        result = engine.answer("test")
        assert "/" in result.model_used  # e.g. "openai/gpt-3.5-turbo"
