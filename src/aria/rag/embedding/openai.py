"""OpenAI embedding service."""

import asyncio

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from aria.config.settings import settings
from aria.exceptions import EmbeddingError
from aria.rag.embedding.base import BaseEmbedder
from aria.types import Embedding

logger = structlog.get_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text embedding service.

    Uses OpenAI's text-embedding-3-small model by default.
    Supports batch processing for efficiency.
    """

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Batch size limits
    MAX_BATCH_SIZE = 100
    MAX_TOKENS_PER_BATCH = 8191

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize OpenAI embedder.

        Args:
            model: Model name (default: from settings).
            api_key: OpenAI API key (default: from settings).
        """
        self._model = model or settings.openai_embedding_model
        api_key = api_key or (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )

        if not api_key:
            raise EmbeddingError("OpenAI API key not configured")

        self.client = AsyncOpenAI(api_key=api_key)

        logger.info(
            "openai_embedder_initialized",
            model=self._model,
            dimension=self.dimension,
        )

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.MODEL_DIMENSIONS.get(self._model, settings.embedding_dimension)

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed(self, text: str) -> Embedding:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            # Truncate if too long
            text = self._truncate_text(text)

            response = await self.client.embeddings.create(
                input=text,
                model=self._model,
            )

            return list(response.data[0].embedding)

        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Generate embeddings for multiple texts.

        Handles batching automatically for efficiency.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        try:
            # Truncate all texts
            texts = [self._truncate_text(t) for t in texts]

            # Process in batches
            all_embeddings: list[Embedding] = []
            for i in range(0, len(texts), self.MAX_BATCH_SIZE):
                batch = texts[i : i + self.MAX_BATCH_SIZE]

                response = await self.client.embeddings.create(
                    input=batch,
                    model=self._model,
                )

                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [list(d.embedding) for d in sorted_data]
                all_embeddings.extend(batch_embeddings)

                # Rate limiting between batches
                if i + self.MAX_BATCH_SIZE < len(texts):
                    await asyncio.sleep(0.1)

            logger.info(
                "batch_embedding_completed",
                text_count=len(texts),
                embedding_count=len(all_embeddings),
            )

            return all_embeddings

        except Exception as e:
            logger.error("batch_embedding_failed", error=str(e))
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    def _truncate_text(self, text: str, max_tokens: int = 8191) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated text.
        """
        # Simple character-based truncation (approximate)
        # 1 token ≈ 4 characters for English text
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text
