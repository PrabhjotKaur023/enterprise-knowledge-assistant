"""
RAG (Retrieval-Augmented Generation) system.

The core of the project. Takes a user query, retrieves relevant chunks,
builds a prompt, and calls the LLM. Sounds simple — the complexity is in
prompt engineering and handling edge cases like empty retrievals.

Supports OpenAI and Gemini. Add more providers by extending LLMClient.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.core.config import settings
from app.pipeline.embedding_pipeline import get_embedding_pipeline

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A retrieved source chunk used to answer a question."""
    chunk_id: str
    document_id: str
    filename: str
    content: str
    score: float
    chunk_index: int


@dataclass
class RAGResponse:
    """Complete response from the RAG system."""
    answer: str
    sources: List[Source]
    query: str
    latency_ms: float
    model_used: str
    retrieved_chunks: int


class LLMClient:
    """
    Abstraction over different LLM providers.
    Keeping it simple — no LangChain LLM wrapper here because I wanted
    to understand what's happening under the hood.
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL

    def complete(self, system_prompt: str, user_message: str) -> str:
        if self.provider == "openai":
            return self._openai_complete(system_prompt, user_message)
        elif self.provider == "gemini":
            return self._gemini_complete(system_prompt, user_message)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _openai_complete(self, system_prompt: str, user_message: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,  # Lower temp for factual Q&A — less hallucination
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _gemini_complete(self, system_prompt: str, user_message: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=self.model or "gemini-1.5-flash",
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_message)
        return response.text.strip()


class PromptBuilder:
    """
    Builds prompts for the RAG system.

    Prompt engineering is underrated. These prompts went through many
    iterations — the key insight was being very explicit about:
    1. Staying grounded in the context
    2. Citing sources
    3. What to say when the answer isn't in the docs
    """

    RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided document excerpts.

Your behavior:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context doesn't contain enough information, say "I don't have enough information in the uploaded documents to answer this."
- Be concise and direct. Avoid filler phrases.
- When relevant, cite the source document by referencing [Source N] tags.
- If multiple sources have conflicting information, mention the discrepancy.
- Format answers clearly. Use bullet points or numbered lists when appropriate.

You are assisting professionals who need accurate, sourced information from their documents."""

    def build_rag_prompt(self, query: str, sources: List[Source]) -> Tuple[str, str]:
        """Returns (system_prompt, user_message) tuple."""
        if not sources:
            context = "No relevant documents found for this query."
        else:
            context_parts = []
            for i, source in enumerate(sources, 1):
                context_parts.append(
                    f"[Source {i}] From: {source.filename} (relevance: {source.score:.2f})\n"
                    f"{source.content}"
                )
            context = "\n\n---\n\n".join(context_parts)

        user_message = f"""Context from uploaded documents:

{context}

---

Question: {query}

Answer based on the context above:"""

        return self.RAG_SYSTEM_PROMPT, user_message

    def build_summary_prompt(self, document_content: str, filename: str) -> Tuple[str, str]:
        system = "You are an expert at summarizing documents. Be concise but comprehensive."
        user = f"Summarize the following document '{filename}':\n\n{document_content[:6000]}"
        return system, user


class RAGEngine:
    """
    Main RAG orchestrator.

    Flow: query → embed → retrieve → build prompt → LLM → return response + sources
    """

    def __init__(self):
        self.embedding_pipeline = get_embedding_pipeline()
        self.llm = LLMClient()
        self.prompt_builder = PromptBuilder()

    def answer(
        self,
        query: str,
        top_k: int = settings.TOP_K_RESULTS,
        document_filter: Optional[List[str]] = None,
    ) -> RAGResponse:
        """
        Answer a query using RAG.

        document_filter: if provided, only retrieve from these document IDs.
        Useful when user wants to query a specific document.
        """
        start_time = time.time()
        logger.info(f"RAG query: '{query[:80]}...' (top_k={top_k})")

        # 1. Retrieve relevant chunks
        raw_results = self.embedding_pipeline.search(query, top_k=top_k * 2)  # fetch extra, filter down

        # 2. Convert to Source objects and apply optional document filter
        sources = []
        for meta, score in raw_results:
            if document_filter and meta["document_id"] not in document_filter:
                continue
            sources.append(Source(
                chunk_id=meta["chunk_id"],
                document_id=meta["document_id"],
                filename=meta["metadata"].get("filename", "Unknown"),
                content=meta["content"],
                score=score,
                chunk_index=meta["metadata"].get("chunk_index", 0),
            ))

        # Limit to top_k after filtering
        sources = sources[:top_k]

        # 3. Build prompt
        system_prompt, user_message = self.prompt_builder.build_rag_prompt(query, sources)

        # 4. Call LLM
        try:
            answer_text = self.llm.complete(system_prompt, user_message)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Return graceful degradation instead of crashing
            answer_text = "I encountered an error generating a response. Please check the server logs."

        latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"RAG response generated in {latency}ms with {len(sources)} sources.")

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            query=query,
            latency_ms=latency,
            model_used=f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
            retrieved_chunks=len(sources),
        )

    def summarize_document(self, document_id: str, filename: str) -> str:
        """Generate a summary for a specific document."""
        chunks = self.embedding_pipeline.vector_store.get_document_chunks(document_id)
        if not chunks:
            return "No content found for this document."

        # Combine first N chunks for summary (keeps it fast and cheap)
        combined = "\n\n".join(c["content"] for c in chunks[:10])
        system, user = self.prompt_builder.build_summary_prompt(combined, filename)
        return self.llm.complete(system, user)


# Module-level singleton
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
