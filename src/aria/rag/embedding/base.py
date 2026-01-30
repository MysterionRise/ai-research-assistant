"""Base embedding interface."""

from abc import ABC, abstractmethod

from aria.types import Embedding


class BaseEmbedder(ABC):
    """Abstract base class for embedding generators.

    Embedders convert text into vector representations for similarity search.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> Embedding:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        pass
