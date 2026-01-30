"""Base vector store interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from aria.types import Embedding


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    section: str | None = None
    page_number: int | None = None
    metadata: dict | None = None


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.

    Vector stores provide storage and similarity search for embeddings.
    """

    @abstractmethod
    async def search(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            min_score: Minimum similarity score threshold.

        Returns:
            List of search results sorted by similarity.
        """
        pass

    @abstractmethod
    async def insert(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        embedding: Embedding,
        metadata: dict | None = None,
    ) -> None:
        """Insert a vector into the store.

        Args:
            chunk_id: Unique chunk identifier.
            document_id: Parent document identifier.
            content: Text content of the chunk.
            embedding: Embedding vector.
            metadata: Optional metadata.
        """
        pass

    @abstractmethod
    async def delete(self, chunk_id: str) -> None:
        """Delete a vector from the store.

        Args:
            chunk_id: Chunk identifier to delete.
        """
        pass

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> int:
        """Delete all vectors for a document.

        Args:
            document_id: Document identifier.

        Returns:
            Number of vectors deleted.
        """
        pass
