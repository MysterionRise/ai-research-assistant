"""Unit tests for RAG pipeline."""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.rag.pipeline import RAGPipeline, RAGPipelineResult
from aria.rag.retrieval.base import RetrievalResult
from aria.types import Citation, RAGResponse


class TestRAGPipelineResult:
    """Tests for RAGPipelineResult dataclass."""

    def test_create_pipeline_result(self) -> None:
        """Test creating RAGPipelineResult."""
        citations = [
            Citation(
                citation_id=1,
                document_id="doc-1",
                title="Paper 1",
                excerpt="Text...",
                confidence=0.9,
            )
        ]
        retrieved = [
            RetrievalResult(
                chunk_id="c1",
                document_id="doc-1",
                content="Content",
                score=0.85,
            )
        ]
        result = RAGPipelineResult(
            answer="The answer is...",
            citations=citations,
            retrieved_chunks=retrieved,
            reranked_chunks=retrieved,
            confidence=0.9,
            latency_ms=150,
        )

        assert result.answer == "The answer is..."
        assert len(result.citations) == 1
        assert result.latency_ms == 150
        assert result.metadata == {}

    def test_pipeline_result_with_metadata(self) -> None:
        """Test RAGPipelineResult with metadata."""
        result = RAGPipelineResult(
            answer="Answer",
            citations=[],
            retrieved_chunks=[],
            reranked_chunks=[],
            confidence=0.5,
            latency_ms=100,
            metadata={"model": "claude-3-sonnet", "tokens": 500},
        )

        assert result.metadata["model"] == "claude-3-sonnet"
        assert result.metadata["tokens"] == 500


class TestRAGPipelineInitialization:
    """Tests for RAGPipeline initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default components."""
        with (
            patch("aria.rag.pipeline.HybridRetriever") as mock_retriever,
            patch("aria.rag.pipeline.CrossEncoderReranker") as mock_reranker,
            patch("aria.rag.pipeline.CitationAwareSynthesizer") as mock_synthesizer,
        ):
            RAGPipeline()

            mock_retriever.assert_called_once()
            mock_reranker.assert_called_once()
            mock_synthesizer.assert_called_once()

    def test_init_with_custom_components(self) -> None:
        """Test initialization with custom components."""
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()
        mock_synthesizer = MagicMock()

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            reranker=mock_reranker,
            synthesizer=mock_synthesizer,
        )

        assert pipeline.retriever is mock_retriever
        assert pipeline.reranker is mock_reranker
        assert pipeline.synthesizer is mock_synthesizer

    def test_init_with_custom_top_k(self) -> None:
        """Test initialization with custom top_k values."""
        with (
            patch("aria.rag.pipeline.HybridRetriever"),
            patch("aria.rag.pipeline.CrossEncoderReranker"),
            patch("aria.rag.pipeline.CitationAwareSynthesizer"),
        ):
            pipeline = RAGPipeline(
                retrieval_top_k=50,
                rerank_top_k=10,
            )

            assert pipeline.retrieval_top_k == 50
            assert pipeline.rerank_top_k == 10


class TestRAGPipelineQuery:
    """Tests for RAGPipeline.query method."""

    @pytest.fixture
    def mock_pipeline(self) -> RAGPipeline:
        """Create pipeline with mocked components."""
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()
        mock_synthesizer = MagicMock()

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            reranker=mock_reranker,
            synthesizer=mock_synthesizer,
        )
        return pipeline

    @pytest.mark.asyncio
    async def test_query_full_pipeline(self, mock_pipeline: RAGPipeline) -> None:
        """Test full pipeline execution."""
        # Setup mock retrieval results
        retrieved_chunks = [
            RetrievalResult(
                chunk_id="c1",
                document_id="doc-1",
                content="Content about topic",
                score=0.9,
                document_title="Paper 1",
            ),
            RetrievalResult(
                chunk_id="c2",
                document_id="doc-2",
                content="More content",
                score=0.85,
                document_title="Paper 2",
            ),
        ]

        # Setup mock reranked results
        reranked_chunks = [retrieved_chunks[0]]

        # Setup mock synthesis result
        @dataclass
        class MockSynthesisResult:
            answer: str = "Based on [1], the answer is..."
            citations: list = field(default_factory=list)
            confidence: float = 0.9
            tokens_used: int = 100
            metadata: dict = field(default_factory=lambda: {"model": "claude-3"})

        mock_synthesis = MockSynthesisResult()
        mock_synthesis.citations = [
            Citation(
                citation_id=1,
                document_id="doc-1",
                title="Paper 1",
                excerpt="Content...",
                confidence=0.9,
            )
        ]

        mock_pipeline.retriever.retrieve = AsyncMock(return_value=retrieved_chunks)
        mock_pipeline.reranker.rerank = AsyncMock(return_value=reranked_chunks)
        mock_pipeline.synthesizer.synthesize = AsyncMock(return_value=mock_synthesis)

        result = await mock_pipeline.query("What is the topic?")

        assert isinstance(result, RAGPipelineResult)
        assert result.answer == "Based on [1], the answer is..."
        assert len(result.citations) == 1
        assert len(result.retrieved_chunks) == 2
        assert len(result.reranked_chunks) == 1
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_query_calls_retriever(self, mock_pipeline: RAGPipeline) -> None:
        """Test that query calls retriever correctly."""
        mock_pipeline.retriever.retrieve = AsyncMock(return_value=[])
        mock_pipeline.reranker.rerank = AsyncMock(return_value=[])

        @dataclass
        class MockSynthesisResult:
            answer: str = "No info"
            citations: list = field(default_factory=list)
            confidence: float = 0.0
            tokens_used: int = 10
            metadata: dict = field(default_factory=dict)

        mock_pipeline.synthesizer.synthesize = AsyncMock(return_value=MockSynthesisResult())

        await mock_pipeline.query("Test query", filters={"year": 2024})

        mock_pipeline.retriever.retrieve.assert_called_once()
        call_kwargs = mock_pipeline.retriever.retrieve.call_args[1]
        assert call_kwargs["query"] == "Test query"
        assert call_kwargs["filters"] == {"year": 2024}

    @pytest.mark.asyncio
    async def test_query_with_empty_retrieval(self, mock_pipeline: RAGPipeline) -> None:
        """Test pipeline with no retrieval results."""
        mock_pipeline.retriever.retrieve = AsyncMock(return_value=[])
        mock_pipeline.reranker.rerank = AsyncMock(return_value=[])

        @dataclass
        class MockSynthesisResult:
            answer: str = "I don't have enough information."
            citations: list = field(default_factory=list)
            confidence: float = 0.0
            tokens_used: int = 10
            metadata: dict = field(default_factory=dict)

        mock_pipeline.synthesizer.synthesize = AsyncMock(return_value=MockSynthesisResult())

        result = await mock_pipeline.query("Unknown topic?")

        # Reranker should not be called with empty retrieval
        mock_pipeline.reranker.rerank.assert_not_called()
        assert result.retrieved_chunks == []
        assert result.reranked_chunks == []

    @pytest.mark.asyncio
    async def test_query_includes_latency(self, mock_pipeline: RAGPipeline) -> None:
        """Test that query includes latency measurement."""
        mock_pipeline.retriever.retrieve = AsyncMock(return_value=[])
        mock_pipeline.reranker.rerank = AsyncMock(return_value=[])

        @dataclass
        class MockSynthesisResult:
            answer: str = "Answer"
            citations: list = field(default_factory=list)
            confidence: float = 0.5
            tokens_used: int = 50
            metadata: dict = field(default_factory=dict)

        mock_pipeline.synthesizer.synthesize = AsyncMock(return_value=MockSynthesisResult())

        result = await mock_pipeline.query("Test?")

        # Latency should be a non-negative integer
        assert isinstance(result.latency_ms, int)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_query_metadata_includes_stats(self, mock_pipeline: RAGPipeline) -> None:
        """Test that query metadata includes statistics."""
        retrieved = [RetrievalResult(chunk_id="c1", document_id="d1", content="Content", score=0.9)]
        reranked = [retrieved[0]]

        mock_pipeline.retriever.retrieve = AsyncMock(return_value=retrieved)
        mock_pipeline.reranker.rerank = AsyncMock(return_value=reranked)

        @dataclass
        class MockSynthesisResult:
            answer: str = "Answer"
            citations: list = field(default_factory=list)
            confidence: float = 0.8
            tokens_used: int = 100
            metadata: dict = field(default_factory=lambda: {"model": "claude-3"})

        mock_pipeline.synthesizer.synthesize = AsyncMock(return_value=MockSynthesisResult())

        result = await mock_pipeline.query("Test?")

        assert "retrieval_count" in result.metadata
        assert "rerank_count" in result.metadata
        assert "tokens_used" in result.metadata
        assert result.metadata["retrieval_count"] == 1
        assert result.metadata["rerank_count"] == 1


class TestRAGPipelineToRAGResponse:
    """Tests for RAGPipeline.to_rag_response method."""

    def test_to_rag_response_conversion(self) -> None:
        """Test converting RAGPipelineResult to RAGResponse."""
        with (
            patch("aria.rag.pipeline.HybridRetriever"),
            patch("aria.rag.pipeline.CrossEncoderReranker"),
            patch("aria.rag.pipeline.CitationAwareSynthesizer"),
        ):
            pipeline = RAGPipeline()

        citations = [
            Citation(
                citation_id=1,
                document_id="doc-1",
                title="Paper 1",
                excerpt="Text",
                confidence=0.9,
            )
        ]
        pipeline_result = RAGPipelineResult(
            answer="The answer based on [1]",
            citations=citations,
            retrieved_chunks=[],
            reranked_chunks=[],
            confidence=0.85,
            latency_ms=200,
            metadata={"model": "claude-3", "tokens_used": 150},
        )

        response = pipeline.to_rag_response(pipeline_result, "What is the question?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "The answer based on [1]"
        assert response.query == "What is the question?"
        assert response.confidence == 0.85
        assert response.sources_used == 1
        assert response.metadata["latency_ms"] == 200

    def test_to_rag_response_preserves_citations(self) -> None:
        """Test that conversion preserves citations."""
        with (
            patch("aria.rag.pipeline.HybridRetriever"),
            patch("aria.rag.pipeline.CrossEncoderReranker"),
            patch("aria.rag.pipeline.CitationAwareSynthesizer"),
        ):
            pipeline = RAGPipeline()

        citations = [
            Citation(
                citation_id=1,
                document_id="doc-1",
                title="Paper 1",
                excerpt="Excerpt 1",
                confidence=0.9,
            ),
            Citation(
                citation_id=2,
                document_id="doc-2",
                title="Paper 2",
                excerpt="Excerpt 2",
                confidence=0.8,
            ),
        ]
        pipeline_result = RAGPipelineResult(
            answer="Answer with [1] and [2]",
            citations=citations,
            retrieved_chunks=[],
            reranked_chunks=[],
            confidence=0.9,
            latency_ms=100,
        )

        response = pipeline.to_rag_response(pipeline_result, "Query?")

        assert len(response.citations) == 2
        assert response.citations[0].document_id == "doc-1"
        assert response.citations[1].document_id == "doc-2"

    def test_to_rag_response_with_empty_citations(self) -> None:
        """Test conversion with no citations."""
        with (
            patch("aria.rag.pipeline.HybridRetriever"),
            patch("aria.rag.pipeline.CrossEncoderReranker"),
            patch("aria.rag.pipeline.CitationAwareSynthesizer"),
        ):
            pipeline = RAGPipeline()

        pipeline_result = RAGPipelineResult(
            answer="I don't have enough information.",
            citations=[],
            retrieved_chunks=[],
            reranked_chunks=[],
            confidence=0.0,
            latency_ms=50,
        )

        response = pipeline.to_rag_response(pipeline_result, "Unknown?")

        assert response.citations == []
        assert response.sources_used == 0
        assert response.confidence == 0.0


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_retrieval_result(self) -> None:
        """Test creating RetrievalResult."""
        result = RetrievalResult(
            chunk_id="chunk-123",
            document_id="doc-456",
            content="This is the chunk content.",
            score=0.92,
        )

        assert result.chunk_id == "chunk-123"
        assert result.document_id == "doc-456"
        assert result.content == "This is the chunk content."
        assert result.score == 0.92

    def test_retrieval_result_with_optional_fields(self) -> None:
        """Test RetrievalResult with all fields."""
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Content here",
            score=0.88,
            section="Methods",
            page_number=15,
            document_title="Research Paper",
            metadata={"embedding_model": "text-embedding-3-large"},
        )

        assert result.section == "Methods"
        assert result.page_number == 15
        assert result.document_title == "Research Paper"
        assert result.metadata["embedding_model"] == "text-embedding-3-large"

    def test_retrieval_result_default_optional_fields(self) -> None:
        """Test RetrievalResult default values."""
        result = RetrievalResult(
            chunk_id="c1",
            document_id="d1",
            content="Content",
            score=0.5,
        )

        assert result.section is None
        assert result.page_number is None
        assert result.document_title is None
        assert result.metadata == {}
