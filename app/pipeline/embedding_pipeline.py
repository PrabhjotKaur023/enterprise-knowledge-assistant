"""
Embedding pipeline and FAISS vector store management.

Using HuggingFace sentence-transformers locally — no API cost, runs on CPU
fine for moderate workloads. The model loads once and stays in memory
(singleton pattern). FAISS index is persisted to disk so we don't re-embed
on every restart, which was the first performance fix I made.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.pipeline.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers.
    Keeps the model as a class variable so it's shared across instances
    (loading 80MB+ model per request would be painful).
    """

    _model = None
    _model_name: str = ""

    @classmethod
    def get_model(cls):
        if cls._model is None or cls._model_name != settings.EMBEDDING_MODEL:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            cls._model_name = settings.EMBEDDING_MODEL
            logger.info("Embedding model loaded.")
        return cls._model

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts to embedding vectors."""
        model = self.get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,  # progress bar only for large batches
            normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class FAISSVectorStore:
    """
    Manages a FAISS index with accompanying metadata store.

    FAISS only stores vectors — we keep a parallel list of chunk metadata
    (content, document_id, etc.) in a pickle file. The index position
    maps directly to the metadata list index.

    TODO: For >1M chunks, consider switching to FAISS IVFFlat or HNSW
    indexes for faster search. FlatL2/IndexFlatIP is exact but O(n).
    """

    def __init__(self, index_path: Optional[str] = None):
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.faiss_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.pkl"

        self._index = None
        self._metadata: List[dict] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.faiss_file.exists() and self.metadata_file.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self.faiss_file))
                with open(self.metadata_file, "rb") as f:
                    self._metadata = pickle.load(f)
                logger.info(
                    f"Loaded FAISS index: {self._index.ntotal} vectors, "
                    f"{len(self._metadata)} metadata entries."
                )
            except Exception as e:
                logger.warning(f"Could not load existing FAISS index: {e}. Starting fresh.")
                self._index = None
                self._metadata = []

    def _init_index(self, dimension: int) -> None:
        """Create a new flat inner-product index (works with normalized vectors = cosine sim)."""
        import faiss
        self._index = faiss.IndexFlatIP(dimension)
        logger.info(f"Created new FAISS index with dimension={dimension}")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add document chunks and their embeddings to the index."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        if self._index is None:
            self._init_index(embeddings.shape[1])

        self._index.add(embeddings)
        for chunk in chunks:
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            })

        self._save()
        logger.info(f"Added {len(chunks)} chunks. Total vectors: {self._index.ntotal}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = settings.TOP_K_RESULTS
    ) -> List[Tuple[dict, float]]:
        """Return top-k (metadata, score) pairs for a query embedding."""
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Vector store is empty — no results to return.")
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append((self._metadata[idx], float(score)))

        return results

    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks for a document.
        FAISS FlatIndex doesn't support deletion directly —
        we rebuild the index without the deleted doc's vectors.
        Expensive but correct. Fine for our scale.
        """
        if self._index is None:
            return 0

        keep_indices = [
            i for i, m in enumerate(self._metadata) if m["document_id"] != document_id
        ]
        removed = len(self._metadata) - len(keep_indices)

        if removed == 0:
            return 0

        # Rebuild index with remaining vectors
        import faiss
        dimension = self._index.d
        new_index = faiss.IndexFlatIP(dimension)

        if keep_indices:
            # Extract kept vectors using reconstruct
            kept_vectors = np.array([
                self._index.reconstruct(i) for i in keep_indices
            ], dtype=np.float32)
            new_index.add(kept_vectors)

        self._index = new_index
        self._metadata = [self._metadata[i] for i in keep_indices]
        self._save()

        logger.info(f"Deleted {removed} chunks for document {document_id}.")
        return removed

    def _save(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self.faiss_file))
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self._metadata, f)

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal if self._index else 0

    def get_document_chunks(self, document_id: str) -> List[dict]:
        return [m for m in self._metadata if m["document_id"] == document_id]


class EmbeddingPipeline:
    """Combines embedding model + vector store into a single interface."""

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = FAISSVectorStore()

    def index_document(self, chunks: List[DocumentChunk]) -> None:
        """Embed and store all chunks for a document."""
        if not chunks:
            logger.warning("No chunks to embed.")
            return

        texts = [c.content for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts)
        self.vector_store.add_chunks(chunks, embeddings)

    def search(self, query: str, top_k: int = settings.TOP_K_RESULTS) -> List[Tuple[dict, float]]:
        """Embed a query and return relevant chunks."""
        query_embedding = self.embedder.encode_single(query)
        return self.vector_store.search(query_embedding, top_k)

    def remove_document(self, document_id: str) -> int:
        return self.vector_store.delete_document(document_id)


# Module-level singleton
_embedding_pipeline: Optional[EmbeddingPipeline] = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    global _embedding_pipeline
    if _embedding_pipeline is None:
        _embedding_pipeline = EmbeddingPipeline()
    return _embedding_pipeline
