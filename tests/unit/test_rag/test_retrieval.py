"""Tests for retrieval components."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.rag.retrieval.base import BaseRetriever, RetrievalResult
from aria.rag.retrieval.hybrid import HybridRetriever
from aria.rag.retrieval.keyword import KeywordRetriever


class TestKeywordRetriever:
    """Tests for KeywordRetriever."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        retriever = KeywordRetriever()
        tokens = retriever._tokenize("The quick brown fox jumps over the lazy dog")

        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        # Stopwords should be removed
        assert "the" not in tokens
        # "over" is kept (not a stopword)
        assert "over" in tokens

    def test_tokenize_removes_short_tokens(self):
        """Test that short tokens are removed."""
        retriever = KeywordRetriever()
        tokens = retriever._tokenize("A I O AI ML NLP RNA DNA")

        # Single letter tokens should be removed
        assert "a" not in tokens
        assert "i" not in tokens
        # 2-letter tokens are also removed (filter is len > 2)
        assert "ai" not in tokens
        assert "ml" not in tokens
        # 3+ letter tokens kept
        assert "nlp" in tokens
        assert "rna" in tokens
        assert "dna" in tokens

    def test_tokenize_lowercase(self):
        """Test that tokens are lowercased."""
        retriever = KeywordRetriever()
        tokens = retriever._tokenize("CRISPR DNA RNA mRNA")

        assert "crispr" in tokens
        assert "dna" in tokens
        assert "rna" in tokens
        assert "mrna" in tokens


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_rrf_fusion_single_list(self):
        """Test RRF fusion with single result list."""
        # Create with mock retrievers to avoid requiring API keys
        mock_semantic = MagicMock()
        mock_keyword = MagicMock()
        retriever = HybridRetriever(
            semantic_weight=0.7,
            keyword_weight=0.3,
            semantic_retriever=mock_semantic,
            keyword_retriever=mock_keyword,
        )

        semantic_results = [
            RetrievalResult(
                chunk_id="1",
                document_id="doc1",
                content="test content 1",
                score=0.9,
            ),
            RetrievalResult(
                chunk_id="2",
                document_id="doc1",
                content="test content 2",
                score=0.8,
            ),
        ]

        fused = retriever._rrf_fusion(semantic_results, [])

        assert len(fused) == 2
        # First result should have higher score
        assert fused[0].score >= fused[1].score

    def test_rrf_fusion_combines_results(self):
        """Test RRF fusion combines results from both lists."""
        mock_semantic = MagicMock()
        mock_keyword = MagicMock()
        retriever = HybridRetriever(
            semantic_weight=0.5,
            keyword_weight=0.5,
            semantic_retriever=mock_semantic,
            keyword_retriever=mock_keyword,
        )

        semantic_results = [
            RetrievalResult(
                chunk_id="1",
                document_id="doc1",
                content="semantic result",
                score=0.9,
            ),
        ]

        keyword_results = [
            RetrievalResult(
                chunk_id="2",
                document_id="doc2",
                content="keyword result",
                score=0.8,
            ),
        ]

        fused = retriever._rrf_fusion(semantic_results, keyword_results)

        assert len(fused) == 2
        chunk_ids = {r.chunk_id for r in fused}
        assert "1" in chunk_ids
        assert "2" in chunk_ids

    def test_rrf_fusion_boosts_duplicates(self):
        """Test RRF fusion boosts results appearing in both lists."""
        mock_semantic = MagicMock()
        mock_keyword = MagicMock()
        retriever = HybridRetriever(
            semantic_weight=0.5,
            keyword_weight=0.5,
            semantic_retriever=mock_semantic,
            keyword_retriever=mock_keyword,
        )

        # Same chunk appears in both lists
        semantic_results = [
            RetrievalResult(
                chunk_id="1",
                document_id="doc1",
                content="common result",
                score=0.9,
            ),
            RetrievalResult(
                chunk_id="2",
                document_id="doc2",
                content="semantic only",
                score=0.8,
            ),
        ]

        keyword_results = [
            RetrievalResult(
                chunk_id="1",
                document_id="doc1",
                content="common result",
                score=0.7,
            ),
            RetrievalResult(
                chunk_id="3",
                document_id="doc3",
                content="keyword only",
                score=0.6,
            ),
        ]

        fused = retriever._rrf_fusion(semantic_results, keyword_results)

        # Chunk 1 should be first (appears in both)
        assert fused[0].chunk_id == "1"

    def test_rrf_fusion_empty_lists(self):
        """Test RRF fusion with empty lists."""
        mock_semantic = MagicMock()
        mock_keyword = MagicMock()
        retriever = HybridRetriever(
            semantic_weight=0.7,
            keyword_weight=0.3,
            semantic_retriever=mock_semantic,
            keyword_retriever=mock_keyword,
        )

        fused = retriever._rrf_fusion([], [])
        assert len(fused) == 0

    @pytest.mark.asyncio
    async def test_retrieve_calls_both_retrievers(self):
        """Test that retrieve() calls both semantic and keyword retrievers."""
        mock_semantic = MagicMock()
        mock_semantic.retrieve = AsyncMock(
            return_value=[
                RetrievalResult(
                    chunk_id="1",
                    document_id="doc1",
                    content="semantic content",
                    score=0.9,
                )
            ]
        )

        mock_keyword = MagicMock()
        mock_keyword.retrieve = AsyncMock(
            return_value=[
                RetrievalResult(
                    chunk_id="2",
                    document_id="doc2",
                    content="keyword content",
                    score=0.8,
                )
            ]
        )

        retriever = HybridRetriever(
            semantic_retriever=mock_semantic,
            keyword_retriever=mock_keyword,
        )

        results = await retriever.retrieve("test query", top_k=5)

        mock_semantic.retrieve.assert_called_once()
        mock_keyword.retrieve.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        """Test that retrieve() respects top_k parameter."""
        # Create many results
        semantic_results = [
            RetrievalResult(
                chunk_id=f"s{i}",
                document_id="doc1",
                content=f"semantic {i}",
                score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]

        keyword_results = [
            RetrievalResult(
                chunk_id=f"k{i}",
                document_id="doc2",
                content=f"keyword {i}",
                score=0.85 - i * 0.1,
            )
            for i in range(5)
        ]

        mock_semantic = MagicMock()
        mock_semantic.retrieve = AsyncMock(return_value=semantic_results)

        mock_keyword = MagicMock()
        mock_keyword.retrieve = AsyncMock(return_value=keyword_results)

        retriever = HybridRetriever(
            semantic_retriever=mock_semantic,
            keyword_retriever=mock_keyword,
        )

        results = await retriever.retrieve("test query", top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self):
        """Test that retrieve() passes filters to retrievers."""
        mock_semantic = MagicMock()
        mock_semantic.retrieve = AsyncMock(return_value=[])

        mock_keyword = MagicMock()
        mock_keyword.retrieve = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            semantic_retriever=mock_semantic,
            keyword_retriever=mock_keyword,
        )

        filters = {"year": 2024}
        await retriever.retrieve("test query", top_k=5, filters=filters)

        # Check filters were passed
        mock_semantic.retrieve.assert_called_once()
        call_args = mock_semantic.retrieve.call_args
        assert call_args[0][2] == filters  # filters is 3rd positional arg


class TestBaseRetriever:
    """Tests for BaseRetriever abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseRetriever cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseRetriever()

    def test_subclass_must_implement_retrieve(self) -> None:
        """Test that subclass must implement retrieve method."""

        class IncompleteRetriever(BaseRetriever):
            pass

        with pytest.raises(TypeError):
            IncompleteRetriever()


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self) -> None:
        """Test creating RetrievalResult."""
        result = RetrievalResult(
            chunk_id="c1",
            document_id="d1",
            content="Test content",
            score=0.9,
        )

        assert result.chunk_id == "c1"
        assert result.document_id == "d1"
        assert result.content == "Test content"
        assert result.score == 0.9

    def test_retrieval_result_with_all_fields(self) -> None:
        """Test RetrievalResult with all fields."""
        result = RetrievalResult(
            chunk_id="c1",
            document_id="d1",
            content="Test content",
            score=0.85,
            section="Introduction",
            page_number=5,
            document_title="Test Paper",
            metadata={"key": "value"},
        )

        assert result.section == "Introduction"
        assert result.page_number == 5
        assert result.document_title == "Test Paper"
        assert result.metadata == {"key": "value"}


class TestSemanticRetriever:
    """Tests for SemanticRetriever."""

    @pytest.mark.asyncio
    async def test_retrieve_with_mocked_dependencies(self) -> None:
        """Test semantic retrieval with mocked embedder and vector store."""
        # Create mock results
        @dataclass
        class MockVectorResult:
            chunk_id: str
            document_id: str
            content: str
            score: float
            section: str | None = None
            page_number: int | None = None
            metadata: dict | None = None

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_vector_store = MagicMock()
        mock_vector_store.search_with_document_info = AsyncMock(
            return_value=[
                MockVectorResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="Test content",
                    score=0.95,
                    section="Methods",
                    page_number=3,
                    metadata={"document_title": "Test Paper"},
                )
            ]
        )

        from aria.rag.retrieval.semantic import SemanticRetriever

        retriever = SemanticRetriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )

        results = await retriever.retrieve("test query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].document_title == "Test Paper"
        mock_embedder.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_retrieve_passes_filters(self) -> None:
        """Test that filters are passed to vector store."""
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_vector_store = MagicMock()
        mock_vector_store.search_with_document_info = AsyncMock(return_value=[])

        from aria.rag.retrieval.semantic import SemanticRetriever

        retriever = SemanticRetriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )

        filters = {"document_id": "doc123"}
        await retriever.retrieve("test query", top_k=5, filters=filters)

        # Check filters were passed
        call_kwargs = mock_vector_store.search_with_document_info.call_args[1]
        assert call_kwargs["filters"] == filters
