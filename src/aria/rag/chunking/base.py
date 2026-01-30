"""Base chunking interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk from a document."""

    content: str
    chunk_index: int
    token_count: int
    section: str | None = None
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict | None = None


class BaseChunker(ABC):
    """Abstract base class for document chunkers.

    Chunkers split documents into smaller pieces suitable for embedding
    and retrieval.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of Chunk objects.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        pass
