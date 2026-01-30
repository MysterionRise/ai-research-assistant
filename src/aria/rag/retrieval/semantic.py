"""Semantic retrieval using vector similarity."""

from typing import Any

import structlog

from aria.rag.embedding.openai import OpenAIEmbedder
from aria.rag.retrieval.base import BaseRetriever, RetrievalResult
from aria.storage.vector.pgvector import PgVectorStore

logger = structlog.get_logger(__name__)


class SemanticRetriever(BaseRetriever):
    """Vector similarity-based semantic retriever.

    Uses embeddings to find semantically similar documents.
    """

    def __init__(
        self,
        embedder: OpenAIEmbedder | None = None,
        vector_store: PgVectorStore | None = None,
    ) -> None:
        """Initialize semantic retriever.

        Args:
            embedder: Embedding model. Creates default if not provided.
            vector_store: Vector store. Creates default if not provided.
        """
        self.embedder = embedder or OpenAIEmbedder()
        self.vector_store = vector_store or PgVectorStore()

        logger.info("semantic_retriever_initialized")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using semantic similarity.

        Args:
            query: Search query.
            top_k: Number of results to return.
            filters: Optional filters (document_id, section, etc.).

        Returns:
            List of retrieval results sorted by similarity.
        """
        logger.info("semantic_retrieval", query=query[:100], top_k=top_k)

        # Generate query embedding
        query_embedding = await self.embedder.embed(query)

        # Search vector store
        vector_results = await self.vector_store.search_with_document_info(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        # Convert to retrieval results
        results = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                section=r.section,
                page_number=r.page_number,
                document_title=r.metadata.get("document_title") if r.metadata else None,
                metadata=r.metadata or {},
            )
            for r in vector_results
        ]

        logger.info(
            "semantic_retrieval_completed",
            results_count=len(results),
        )

        return results
