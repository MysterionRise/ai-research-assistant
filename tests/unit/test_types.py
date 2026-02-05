"""Unit tests for shared type definitions."""

import pytest
from pydantic import ValidationError

from aria.types import (
    ChatContext,
    ChunkMetadata,
    ChunkType,
    Citation,
    DocumentMetadata,
    DocumentSource,
    EvaluationResult,
    ProcessingResult,
    RAGContext,
    RAGResponse,
    RetrievedChunk,
    SearchResult,
    SearchSource,
)


class TestEnums:
    """Tests for enum types."""

    def test_document_source_values(self) -> None:
        """Test DocumentSource enum values."""
        assert DocumentSource.INTERNAL == "internal"
        assert DocumentSource.PUBMED == "pubmed"
        assert DocumentSource.ARXIV == "arxiv"
        assert DocumentSource.SEMANTIC_SCHOLAR == "semantic_scholar"

    def test_search_source_values(self) -> None:
        """Test SearchSource enum values."""
        assert SearchSource.INTERNAL == "internal"
        assert SearchSource.PUBMED == "pubmed"
        assert SearchSource.ARXIV == "arxiv"
        assert SearchSource.SEMANTIC_SCHOLAR == "semantic_scholar"

    def test_chunk_type_values(self) -> None:
        """Test ChunkType enum values."""
        assert ChunkType.TEXT == "text"
        assert ChunkType.TABLE == "table"
        assert ChunkType.FIGURE_CAPTION == "figure_caption"
        assert ChunkType.EQUATION == "equation"
        assert ChunkType.CODE == "code"

    def test_document_source_is_str(self) -> None:
        """Test that DocumentSource values are strings."""
        for source in DocumentSource:
            assert isinstance(source.value, str)

    def test_chunk_type_is_str(self) -> None:
        """Test that ChunkType values are strings."""
        for chunk_type in ChunkType:
            assert isinstance(chunk_type.value, str)


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_minimal_document_metadata(self) -> None:
        """Test creating DocumentMetadata with only required field."""
        meta = DocumentMetadata(title="Test Paper")
        assert meta.title == "Test Paper"
        assert meta.authors == []
        assert meta.year is None
        assert meta.source == DocumentSource.INTERNAL

    def test_full_document_metadata(self) -> None:
        """Test creating DocumentMetadata with all fields."""
        meta = DocumentMetadata(
            title="Advanced Materials Research",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            journal="Nature Materials",
            doi="10.1038/nature12345",
            abstract="This paper discusses...",
            keywords=["materials", "science"],
            source=DocumentSource.PUBMED,
            custom_fields={"impact_factor": 20.5},
        )
        assert meta.title == "Advanced Materials Research"
        assert len(meta.authors) == 2
        assert meta.year == 2024
        assert meta.source == DocumentSource.PUBMED
        assert meta.custom_fields["impact_factor"] == 20.5

    def test_document_metadata_serialization(self) -> None:
        """Test DocumentMetadata JSON serialization."""
        meta = DocumentMetadata(title="Test", year=2024)
        data = meta.model_dump()
        assert data["title"] == "Test"
        assert data["year"] == 2024
        assert "authors" in data


class TestChunkMetadata:
    """Tests for ChunkMetadata model."""

    def test_minimal_chunk_metadata(self) -> None:
        """Test creating ChunkMetadata with required fields."""
        chunk = ChunkMetadata(
            document_id="doc-123",
            chunk_index=0,
            token_count=100,
        )
        assert chunk.document_id == "doc-123"
        assert chunk.chunk_index == 0
        assert chunk.token_count == 100
        assert chunk.chunk_type == ChunkType.TEXT

    def test_full_chunk_metadata(self) -> None:
        """Test creating ChunkMetadata with all fields."""
        chunk = ChunkMetadata(
            document_id="doc-456",
            chunk_index=5,
            section="Methods",
            page_number=12,
            chunk_type=ChunkType.TABLE,
            token_count=250,
        )
        assert chunk.section == "Methods"
        assert chunk.page_number == 12
        assert chunk.chunk_type == ChunkType.TABLE

    def test_chunk_metadata_requires_document_id(self) -> None:
        """Test that document_id is required."""
        with pytest.raises(ValidationError):
            ChunkMetadata(chunk_index=0, token_count=100)


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_minimal_search_result(self) -> None:
        """Test creating SearchResult with required fields."""
        result = SearchResult(
            id="result-123",
            title="Sample Paper",
            source=SearchSource.PUBMED,
            score=0.95,
        )
        assert result.id == "result-123"
        assert result.title == "Sample Paper"
        assert result.score == 0.95
        assert result.authors == []

    def test_search_result_score_validation(self) -> None:
        """Test that score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            SearchResult(
                id="test",
                title="Test",
                source=SearchSource.INTERNAL,
                score=1.5,  # Invalid: > 1
            )

        with pytest.raises(ValidationError):
            SearchResult(
                id="test",
                title="Test",
                source=SearchSource.INTERNAL,
                score=-0.1,  # Invalid: < 0
            )

    def test_search_result_boundary_scores(self) -> None:
        """Test score boundary values."""
        # Score = 0 should be valid
        result_zero = SearchResult(
            id="test",
            title="Test",
            source=SearchSource.INTERNAL,
            score=0.0,
        )
        assert result_zero.score == 0.0

        # Score = 1 should be valid
        result_one = SearchResult(
            id="test",
            title="Test",
            source=SearchSource.INTERNAL,
            score=1.0,
        )
        assert result_one.score == 1.0


class TestRetrievedChunk:
    """Tests for RetrievedChunk model."""

    def test_minimal_retrieved_chunk(self) -> None:
        """Test creating RetrievedChunk with required fields."""
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Sample chunk content",
            score=0.85,
        )
        assert chunk.chunk_id == "chunk-1"
        assert chunk.content == "Sample chunk content"
        assert chunk.metadata == {}

    def test_full_retrieved_chunk(self) -> None:
        """Test creating RetrievedChunk with all fields."""
        chunk = RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            content="Full chunk content here",
            score=0.92,
            section="Results",
            page_number=5,
            document_title="Research Paper",
            metadata={"embedding_model": "text-embedding-3-large"},
        )
        assert chunk.section == "Results"
        assert chunk.page_number == 5
        assert chunk.document_title == "Research Paper"


class TestCitation:
    """Tests for Citation model."""

    def test_minimal_citation(self) -> None:
        """Test creating Citation with required fields."""
        citation = Citation(
            citation_id=1,
            document_id="doc-123",
            title="Test Paper",
            excerpt="Relevant text here",
            confidence=0.9,
        )
        assert citation.citation_id == 1
        assert citation.chunk_id is None

    def test_citation_confidence_validation(self) -> None:
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Citation(
                citation_id=1,
                document_id="doc-1",
                title="Test",
                excerpt="Text",
                confidence=1.5,
            )


class TestRAGContext:
    """Tests for RAGContext model."""

    def test_rag_context_creation(self) -> None:
        """Test creating RAGContext."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                content="Content 1",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id="c2",
                document_id="d2",
                content="Content 2",
                score=0.8,
            ),
        ]
        context = RAGContext(
            query="What is the answer?",
            chunks=chunks,
            total_tokens=500,
        )
        assert context.query == "What is the answer?"
        assert len(context.chunks) == 2
        assert context.total_tokens == 500


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_minimal_rag_response(self) -> None:
        """Test creating RAGResponse with required fields."""
        response = RAGResponse(
            answer="The answer is 42.",
            confidence=0.95,
            query="What is the answer?",
            sources_used=3,
        )
        assert response.answer == "The answer is 42."
        assert response.citations == []
        assert response.sources_used == 3

    def test_rag_response_with_citations(self) -> None:
        """Test RAGResponse with citations."""
        citations = [
            Citation(
                citation_id=1,
                document_id="doc-1",
                title="Paper 1",
                excerpt="Quote 1",
                confidence=0.9,
            ),
        ]
        response = RAGResponse(
            answer="Based on [1], the answer is...",
            citations=citations,
            confidence=0.85,
            query="Question?",
            sources_used=1,
        )
        assert len(response.citations) == 1
        assert response.citations[0].document_id == "doc-1"


class TestChatContext:
    """Tests for ChatContext model."""

    def test_minimal_chat_context(self) -> None:
        """Test creating ChatContext with required fields."""
        context = ChatContext(conversation_id="conv-123")
        assert context.conversation_id == "conv-123"
        assert context.message_history == []
        assert context.rag_context is None

    def test_chat_context_with_history(self) -> None:
        """Test ChatContext with message history."""
        context = ChatContext(
            conversation_id="conv-456",
            message_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )
        assert len(context.message_history) == 2


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_evaluation_result_creation(self) -> None:
        """Test creating EvaluationResult."""
        result = EvaluationResult(
            query="Test query",
            generated_answer="Generated response",
            faithfulness=0.85,
            relevancy=0.90,
            context_precision=0.88,
            latency_ms=150,
        )
        assert result.faithfulness == 0.85
        assert result.expected_answer is None
        assert result.latency_ms == 150

    def test_evaluation_metrics_validation(self) -> None:
        """Test that evaluation metrics are validated."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                query="Test",
                generated_answer="Answer",
                faithfulness=1.5,  # Invalid
                relevancy=0.9,
                context_precision=0.8,
                latency_ms=100,
            )


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_successful_processing_result(self) -> None:
        """Test creating successful ProcessingResult."""
        result = ProcessingResult(
            document_id="doc-123",
            success=True,
            chunk_count=25,
            processing_time_ms=1500,
            metadata=DocumentMetadata(title="Test Paper"),
        )
        assert result.success is True
        assert result.chunk_count == 25
        assert result.error is None
        assert result.metadata is not None

    def test_failed_processing_result(self) -> None:
        """Test creating failed ProcessingResult."""
        result = ProcessingResult(
            document_id="doc-456",
            success=False,
            error="Failed to parse PDF",
            processing_time_ms=500,
        )
        assert result.success is False
        assert result.error == "Failed to parse PDF"
        assert result.chunk_count == 0
