"""
Unit tests for the document processing pipeline.

Testing chunking logic and text cleaning — these are the parts that are
easiest to unit test and have caused the most bugs in practice.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.pipeline.document_processor import (
    DocumentValidator,
    TextChunker,
    TextExtractor,
    MetadataExtractor,
    DocumentChunk,
)


class TestDocumentValidator:
    def test_empty_file_rejected(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        validator = DocumentValidator()
        valid, error = validator.validate(f, "empty.txt")
        assert not valid
        assert "empty" in error.lower()

    def test_unsupported_extension_rejected(self, tmp_path):
        f = tmp_path / "file.exe"
        f.write_bytes(b"data" * 100)
        validator = DocumentValidator()
        valid, error = validator.validate(f, "file.exe")
        assert not valid
        assert "Unsupported" in error

    def test_valid_txt_accepted(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world content")
        validator = DocumentValidator()
        valid, error = validator.validate(f, "doc.txt")
        assert valid
        assert error == ""

    def test_missing_file_rejected(self, tmp_path):
        f = tmp_path / "nonexistent.txt"
        validator = DocumentValidator()
        valid, error = validator.validate(f, "nonexistent.txt")
        assert not valid


class TestTextChunker:
    def setup_method(self):
        # Small chunk size for test predictability
        self.chunker = TextChunker(chunk_size=100, overlap=20)

    def test_short_text_becomes_single_chunk(self):
        text = "This is a short document."
        chunks = self.chunker.chunk(text, "doc_001", {})
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_long_text_becomes_multiple_chunks(self):
        # 10 paragraphs of 50 chars each
        text = "\n\n".join(["word " * 10] * 10)
        chunks = self.chunker.chunk(text, "doc_002", {})
        assert len(chunks) > 1

    def test_chunks_have_correct_document_id(self):
        text = "Some text here.\n\nAnother paragraph."
        chunks = self.chunker.chunk(text, "my_doc_id", {})
        for chunk in chunks:
            assert chunk.document_id == "my_doc_id"

    def test_chunk_ids_are_unique(self):
        text = "\n\n".join(["sentence " * 20] * 5)
        chunks = self.chunker.chunk(text, "doc_003", {})
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_metadata_preserved_in_chunks(self):
        text = "Some content here."
        meta = {"filename": "test.txt", "file_type": "txt"}
        chunks = self.chunker.chunk(text, "doc_004", meta)
        for chunk in chunks:
            assert chunk.metadata["filename"] == "test.txt"

    def test_empty_text_returns_no_chunks(self):
        chunks = self.chunker.chunk("", "doc_005", {})
        assert len(chunks) == 0

    def test_whitespace_only_returns_no_chunks(self):
        chunks = self.chunker.chunk("   \n\n   \n\n   ", "doc_006", {})
        assert len(chunks) == 0


class TestTextExtractor:
    def setup_method(self):
        self.extractor = TextExtractor()

    def test_txt_extraction(self, tmp_path):
        content = "Hello World\nThis is test content."
        f = tmp_path / "test.txt"
        f.write_text(content, encoding="utf-8")
        result = self.extractor.extract(f, "txt")
        assert "Hello World" in result
        assert "test content" in result

    def test_txt_latin1_fallback(self, tmp_path):
        content = "Caf\xe9 content"  # latin-1 encoded
        f = tmp_path / "latin.txt"
        f.write_bytes(content.encode("latin-1"))
        result = self.extractor.extract(f, "txt")
        assert result  # Should not throw

    def test_unsupported_type_raises(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="No extractor"):
            self.extractor.extract(f, "xyz")

    def test_clean_text_removes_null_bytes(self):
        text = "Hello\x00World"
        result = self.extractor._clean_text(text)
        assert "\x00" not in result

    def test_clean_text_normalizes_line_endings(self):
        text = "Line1\r\nLine2\rLine3"
        result = self.extractor._clean_text(text)
        assert "\r" not in result


class TestMetadataExtractor:
    def test_word_count_accuracy(self):
        extractor = MetadataExtractor()
        text = "one two three four five"
        meta = extractor.extract(text, "test.txt", "txt")
        assert meta["word_count"] == 5

    def test_page_estimate_reasonable(self):
        extractor = MetadataExtractor()
        # 500 words should estimate ~2 pages
        text = " ".join(["word"] * 500)
        meta = extractor.extract(text, "doc.txt", "txt")
        assert meta["estimated_pages"] >= 1

    def test_metadata_contains_required_keys(self):
        extractor = MetadataExtractor()
        meta = extractor.extract("content", "file.pdf", "pdf")
        assert "filename" in meta
        assert "file_type" in meta
        assert "word_count" in meta
        assert "char_count" in meta
