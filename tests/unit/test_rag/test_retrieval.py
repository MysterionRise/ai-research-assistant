"""Tests for retrieval components."""

from unittest.mock import MagicMock

from aria.rag.retrieval.base import RetrievalResult
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
