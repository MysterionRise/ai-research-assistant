"""Base retriever interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """Result from document retrieval."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    section: str | None = None
    page_number: int | None = None
    document_title: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract base class for document retrievers.

    Retrievers find relevant document chunks for a given query.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of retrieval results sorted by relevance.
        """
        pass
