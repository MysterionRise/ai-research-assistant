"""Unit tests for document endpoints and models."""

from datetime import UTC, datetime
from uuid import uuid4


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_document_metadata_creation(self) -> None:
        """Test creating DocumentMetadata."""
        from aria.api.routes.documents import DocumentMetadata

        metadata = DocumentMetadata(
            title="Test Document",
            authors=["Author One", "Author Two"],
            year=2024,
            journal="Nature",
            doi="10.1000/test",
            tags=["science", "research"],
        )

        assert metadata.title == "Test Document"
        assert len(metadata.authors) == 2
        assert metadata.year == 2024
        assert metadata.journal == "Nature"

    def test_document_metadata_defaults(self) -> None:
        """Test DocumentMetadata default values."""
        from aria.api.routes.documents import DocumentMetadata

        metadata = DocumentMetadata(title="Test")

        assert metadata.authors == []
        assert metadata.year is None
        assert metadata.journal is None
        assert metadata.doi is None
        assert metadata.tags == []
        assert metadata.custom_fields == {}


class TestDocumentResponse:
    """Tests for DocumentResponse model."""

    def test_document_response_creation(self) -> None:
        """Test creating DocumentResponse."""
        from aria.api.routes.documents import DocumentMetadata, DocumentResponse

        doc_id = uuid4()
        response = DocumentResponse(
            id=doc_id,
            metadata=DocumentMetadata(title="Test"),
            file_type="application/pdf",
            file_size=12345,
            chunk_count=10,
            status="completed",
            created_at=datetime.now(UTC),
        )

        assert response.id == doc_id
        assert response.file_type == "application/pdf"
        assert response.status == "completed"


class TestDocumentCreateRequest:
    """Tests for DocumentCreateRequest model."""

    def test_document_create_request(self) -> None:
        """Test creating DocumentCreateRequest."""
        from aria.api.routes.documents import DocumentCreateRequest, DocumentMetadata

        request = DocumentCreateRequest(
            metadata=DocumentMetadata(
                title="New Document",
                authors=["Author"],
            )
        )

        assert request.metadata.title == "New Document"


class TestDocumentListResponse:
    """Tests for DocumentListResponse model."""

    def test_document_list_response_empty(self) -> None:
        """Test empty DocumentListResponse."""
        from aria.api.routes.documents import DocumentListResponse

        response = DocumentListResponse(
            total=0,
            documents=[],
            page=1,
            page_size=20,
        )

        assert response.total == 0
        assert response.documents == []
        assert response.page == 1


class TestIngestResponse:
    """Tests for IngestResponse model."""

    def test_ingest_response_creation(self) -> None:
        """Test creating IngestResponse."""
        from aria.api.routes.documents import IngestResponse

        doc_id = uuid4()
        response = IngestResponse(
            document_id=doc_id,
            status="queued",
            message="Document queued for processing",
        )

        assert response.document_id == doc_id
        assert response.status == "queued"


class TestChunkResponse:
    """Tests for ChunkResponse model."""

    def test_chunk_response_creation(self) -> None:
        """Test creating ChunkResponse."""
        from aria.api.routes.documents import ChunkResponse

        response = ChunkResponse(
            id="chunk-123",
            content="This is the chunk content",
            chunk_index=0,
            section="Introduction",
            page_number=1,
            token_count=50,
        )

        assert response.id == "chunk-123"
        assert response.chunk_index == 0
        assert response.section == "Introduction"
