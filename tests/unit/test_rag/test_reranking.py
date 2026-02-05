"""Unit tests for reranking components."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aria.rag.retrieval.base import RetrievalResult


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        with patch("aria.rag.reranking.cross_encoder.settings") as mock_settings:
            mock_settings.rag_rerank_top_k = 5

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()

            assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert reranker.top_k == 5

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        with patch("aria.rag.reranking.cross_encoder.settings"):
            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                model_name="custom-model",
                top_k=10,
            )

            assert reranker.model_name == "custom-model"
            assert reranker.top_k == 10

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self) -> None:
        """Test reranking empty results returns empty list."""
        with patch("aria.rag.reranking.cross_encoder.settings") as mock_settings:
            mock_settings.rag_rerank_top_k = 5

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()
            results = await reranker.rerank("test query", [])

            assert results == []

    @pytest.mark.asyncio
    async def test_rerank_fallback_no_sentence_transformers(self) -> None:
        """Test reranking falls back when sentence_transformers not available."""
        with (
            patch("aria.rag.reranking.cross_encoder.settings") as mock_settings,
            patch.dict("sys.modules", {"sentence_transformers": None}),
        ):
            mock_settings.rag_rerank_top_k = 5

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()
            reranker._model = "fallback"  # Simulate fallback mode

            results = [
                RetrievalResult(
                    chunk_id="1",
                    document_id="doc1",
                    content="Low score content",
                    score=0.5,
                ),
                RetrievalResult(
                    chunk_id="2",
                    document_id="doc1",
                    content="High score content",
                    score=0.9,
                ),
            ]

            reranked = await reranker.rerank("test query", results)

            # Should be sorted by original score in fallback mode
            assert reranked[0].chunk_id == "2"  # Higher score first
            assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self) -> None:
        """Test that rerank respects top_k parameter."""
        with patch("aria.rag.reranking.cross_encoder.settings") as mock_settings:
            mock_settings.rag_rerank_top_k = 5

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()
            reranker._model = "fallback"

            results = [
                RetrievalResult(
                    chunk_id=str(i),
                    document_id="doc1",
                    content=f"Content {i}",
                    score=0.9 - i * 0.1,
                )
                for i in range(10)
            ]

            reranked = await reranker.rerank("test query", results, top_k=3)

            assert len(reranked) == 3

    @pytest.mark.asyncio
    async def test_rerank_with_model(self) -> None:
        """Test reranking with mocked cross-encoder model."""
        with patch("aria.rag.reranking.cross_encoder.settings") as mock_settings:
            mock_settings.rag_rerank_top_k = 5

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            # Create mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])

            reranker = CrossEncoderReranker()
            reranker._model = mock_model

            results = [
                RetrievalResult(
                    chunk_id="1",
                    document_id="doc1",
                    content="Content 1",
                    score=0.8,
                ),
                RetrievalResult(
                    chunk_id="2",
                    document_id="doc1",
                    content="Content 2",
                    score=0.7,
                ),
                RetrievalResult(
                    chunk_id="3",
                    document_id="doc1",
                    content="Content 3",
                    score=0.6,
                ),
            ]

            reranked = await reranker.rerank("test query", results, top_k=2)

            # Should be reordered by cross-encoder scores
            assert len(reranked) == 2
            assert reranked[0].chunk_id == "2"  # Highest cross-encoder score (0.9)
            mock_model.predict.assert_called_once()


class TestLoadModel:
    """Tests for model loading."""

    def test_lazy_loading(self) -> None:
        """Test that model is not loaded until needed."""
        with patch("aria.rag.reranking.cross_encoder.settings") as mock_settings:
            mock_settings.rag_rerank_top_k = 5

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()

            assert reranker._model is None

    def test_load_model_fallback(self) -> None:
        """Test that model falls back when sentence_transformers not available."""
        with patch("aria.rag.reranking.cross_encoder.settings") as mock_settings:
            mock_settings.rag_rerank_top_k = 5

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()

            # Patch the import to fail
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                # Force model loading
                original_import = __builtins__["__import__"]

                def mock_import(name, *args, **kwargs):
                    if name == "sentence_transformers":
                        raise ImportError("Mocked ImportError")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    reranker._model = None  # Reset
                    reranker._load_model()

                    # Should have fallback
                    assert reranker._model == "fallback"


class TestRerankerSorting:
    """Tests for reranker sorting behavior."""

    @pytest.mark.asyncio
    async def test_fallback_sorting_order(self) -> None:
        """Test that fallback mode sorts by score in descending order."""
        with patch("aria.rag.reranking.cross_encoder.settings") as mock_settings:
            mock_settings.rag_rerank_top_k = 10

            from aria.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker()
            reranker._model = "fallback"

            results = [
                RetrievalResult(
                    chunk_id="1",
                    document_id="doc1",
                    content="Medium score",
                    score=0.5,
                ),
                RetrievalResult(
                    chunk_id="2",
                    document_id="doc1",
                    content="Highest score",
                    score=0.9,
                ),
                RetrievalResult(
                    chunk_id="3",
                    document_id="doc1",
                    content="Lowest score",
                    score=0.1,
                ),
            ]

            reranked = await reranker.rerank("test query", results)

            # Should be sorted by score descending
            assert reranked[0].chunk_id == "2"  # 0.9
            assert reranked[1].chunk_id == "1"  # 0.5
            assert reranked[2].chunk_id == "3"  # 0.1
