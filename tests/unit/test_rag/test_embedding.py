"""Unit tests for embedding components."""

import pytest

from aria.rag.embedding.base import BaseEmbedder


class TestBaseEmbedder:
    """Tests for BaseEmbedder abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseEmbedder cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_subclass_must_implement_dimension(self) -> None:
        """Test that subclass must implement dimension property."""

        class IncompleteEmbedder(BaseEmbedder):
            @property
            def model_name(self) -> str:
                return "test"

            async def embed(self, text: str):
                return [0.0] * 10

            async def embed_batch(self, texts: list[str]):
                return [[0.0] * 10 for _ in texts]

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_subclass_must_implement_model_name(self) -> None:
        """Test that subclass must implement model_name property."""

        class IncompleteEmbedder(BaseEmbedder):
            @property
            def dimension(self) -> int:
                return 10

            async def embed(self, text: str):
                return [0.0] * 10

            async def embed_batch(self, texts: list[str]):
                return [[0.0] * 10 for _ in texts]

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_subclass_must_implement_embed(self) -> None:
        """Test that subclass must implement embed method."""

        class IncompleteEmbedder(BaseEmbedder):
            @property
            def dimension(self) -> int:
                return 10

            @property
            def model_name(self) -> str:
                return "test"

            async def embed_batch(self, texts: list[str]):
                return [[0.0] * 10 for _ in texts]

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_subclass_must_implement_embed_batch(self) -> None:
        """Test that subclass must implement embed_batch method."""

        class IncompleteEmbedder(BaseEmbedder):
            @property
            def dimension(self) -> int:
                return 10

            @property
            def model_name(self) -> str:
                return "test"

            async def embed(self, text: str):
                return [0.0] * 10

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that complete implementation can be instantiated."""

        class MockEmbedder(BaseEmbedder):
            @property
            def dimension(self) -> int:
                return 384

            @property
            def model_name(self) -> str:
                return "mock-embedder"

            async def embed(self, text: str):
                return [0.1] * self.dimension

            async def embed_batch(self, texts: list[str]):
                return [[0.1] * self.dimension for _ in texts]

        embedder = MockEmbedder()
        assert isinstance(embedder, BaseEmbedder)
        assert embedder.dimension == 384
        assert embedder.model_name == "mock-embedder"

    @pytest.mark.asyncio
    async def test_mock_embedder_embed(self) -> None:
        """Test mock embedder embed method."""

        class MockEmbedder(BaseEmbedder):
            @property
            def dimension(self) -> int:
                return 10

            @property
            def model_name(self) -> str:
                return "mock"

            async def embed(self, text: str):
                return [0.1] * self.dimension

            async def embed_batch(self, texts: list[str]):
                return [[0.1] * self.dimension for _ in texts]

        embedder = MockEmbedder()
        embedding = await embedder.embed("test text")

        assert len(embedding) == 10

    @pytest.mark.asyncio
    async def test_mock_embedder_embed_batch(self) -> None:
        """Test mock embedder embed_batch method."""

        class MockEmbedder(BaseEmbedder):
            @property
            def dimension(self) -> int:
                return 10

            @property
            def model_name(self) -> str:
                return "mock"

            async def embed(self, text: str):
                return [0.1] * self.dimension

            async def embed_batch(self, texts: list[str]):
                return [[0.1] * self.dimension for _ in texts]

        embedder = MockEmbedder()
        embeddings = await embedder.embed_batch(["text 1", "text 2", "text 3"])

        assert len(embeddings) == 3
        assert all(len(e) == 10 for e in embeddings)


class TestOpenAIEmbedderConstants:
    """Tests for OpenAIEmbedder constants."""

    def test_model_dimensions(self) -> None:
        """Test MODEL_DIMENSIONS constant."""
        from aria.rag.embedding.openai import OpenAIEmbedder

        assert "text-embedding-3-small" in OpenAIEmbedder.MODEL_DIMENSIONS
        assert OpenAIEmbedder.MODEL_DIMENSIONS["text-embedding-3-small"] == 1536
        assert "text-embedding-3-large" in OpenAIEmbedder.MODEL_DIMENSIONS
        assert OpenAIEmbedder.MODEL_DIMENSIONS["text-embedding-3-large"] == 3072
        assert "text-embedding-ada-002" in OpenAIEmbedder.MODEL_DIMENSIONS
        assert OpenAIEmbedder.MODEL_DIMENSIONS["text-embedding-ada-002"] == 1536

    def test_batch_size_limits(self) -> None:
        """Test batch size limits."""
        from aria.rag.embedding.openai import OpenAIEmbedder

        assert OpenAIEmbedder.MAX_BATCH_SIZE == 100
        assert OpenAIEmbedder.MAX_TOKENS_PER_BATCH == 8191


class TestOpenAIEmbedderInit:
    """Tests for OpenAIEmbedder initialization."""

    def test_init_raises_without_api_key(self) -> None:
        """Test that initialization raises without API key."""
        from unittest.mock import patch

        from aria.exceptions import EmbeddingError
        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = None

            with pytest.raises(EmbeddingError) as exc_info:
                OpenAIEmbedder()

            assert "API key not configured" in str(exc_info.value)

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        from unittest.mock import patch

        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = None
            mock_settings.embedding_dimension = 1536

            embedder = OpenAIEmbedder(api_key="test-api-key")

            assert embedder._model == "text-embedding-3-small"
            assert embedder.client is not None


class TestOpenAIEmbedderProperties:
    """Tests for OpenAIEmbedder properties."""

    def test_dimension_property(self) -> None:
        """Test dimension property returns correct value."""
        from unittest.mock import patch

        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = None
            mock_settings.embedding_dimension = 1536

            embedder = OpenAIEmbedder(api_key="test-api-key")

            assert embedder.dimension == 1536

    def test_dimension_property_large_model(self) -> None:
        """Test dimension property for large model."""
        from unittest.mock import patch

        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-large"
            mock_settings.openai_api_key = None
            mock_settings.embedding_dimension = 1536

            embedder = OpenAIEmbedder(
                model="text-embedding-3-large",
                api_key="test-api-key",
            )

            assert embedder.dimension == 3072

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        from unittest.mock import patch

        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = None
            mock_settings.embedding_dimension = 1536

            embedder = OpenAIEmbedder(api_key="test-api-key")

            assert embedder.model_name == "text-embedding-3-small"


class TestOpenAIEmbedderTruncateText:
    """Tests for OpenAIEmbedder._truncate_text method."""

    def test_truncate_short_text(self) -> None:
        """Test that short text is not truncated."""
        from unittest.mock import patch

        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = None
            mock_settings.embedding_dimension = 1536

            embedder = OpenAIEmbedder(api_key="test-api-key")

            short_text = "This is a short text."
            result = embedder._truncate_text(short_text)

            assert result == short_text

    def test_truncate_long_text(self) -> None:
        """Test that long text is truncated."""
        from unittest.mock import patch

        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = None
            mock_settings.embedding_dimension = 1536

            embedder = OpenAIEmbedder(api_key="test-api-key")

            # Create a very long text (more than 8191 * 4 characters)
            long_text = "a" * 40000
            result = embedder._truncate_text(long_text, max_tokens=8191)

            # Should be truncated to max_tokens * 4 characters
            assert len(result) == 8191 * 4

    def test_truncate_custom_limit(self) -> None:
        """Test truncation with custom token limit."""
        from unittest.mock import patch

        from aria.rag.embedding.openai import OpenAIEmbedder

        with patch("aria.rag.embedding.openai.settings") as mock_settings:
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = None
            mock_settings.embedding_dimension = 1536

            embedder = OpenAIEmbedder(api_key="test-api-key")

            long_text = "a" * 500
            result = embedder._truncate_text(long_text, max_tokens=100)

            # Should be truncated to 100 * 4 = 400 characters
            assert len(result) == 400
