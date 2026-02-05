"""Unit tests for database models."""

from unittest.mock import MagicMock


class TestDocumentStatus:
    """Tests for DocumentStatus enum."""

    def test_document_status_values(self) -> None:
        """Test DocumentStatus enum values."""
        from aria.db.models.document import DocumentStatus

        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

    def test_document_status_is_str(self) -> None:
        """Test that DocumentStatus values are strings."""
        from aria.db.models.document import DocumentStatus

        for status in DocumentStatus:
            assert isinstance(status.value, str)


class TestDocumentModel:
    """Tests for Document model."""

    def test_document_repr(self) -> None:
        """Test Document __repr__ method."""
        from aria.db.models.document import Document

        doc = Document()
        doc.id = "test-id-123"
        doc.title = "Test Document Title That Is Long Enough"

        repr_str = repr(doc)
        assert "Document" in repr_str
        assert "test-id-123" in repr_str

    def test_document_is_processing(self) -> None:
        """Test is_processing property."""
        from aria.db.models.document import Document, DocumentStatus

        doc = Document()
        doc.status = DocumentStatus.PROCESSING.value
        assert doc.is_processing is True

        doc.status = DocumentStatus.COMPLETED.value
        assert doc.is_processing is False

    def test_document_is_completed(self) -> None:
        """Test is_completed property."""
        from aria.db.models.document import Document, DocumentStatus

        doc = Document()
        doc.status = DocumentStatus.COMPLETED.value
        assert doc.is_completed is True

        doc.status = DocumentStatus.PENDING.value
        assert doc.is_completed is False

    def test_document_is_failed(self) -> None:
        """Test is_failed property."""
        from aria.db.models.document import Document, DocumentStatus

        doc = Document()
        doc.status = DocumentStatus.FAILED.value
        assert doc.is_failed is True

        doc.status = DocumentStatus.COMPLETED.value
        assert doc.is_failed is False

    def test_mark_processing(self) -> None:
        """Test mark_processing method."""
        from aria.db.models.document import Document, DocumentStatus

        doc = Document()
        doc.status = DocumentStatus.PENDING.value
        doc.mark_processing()

        assert doc.status == DocumentStatus.PROCESSING.value

    def test_mark_completed(self) -> None:
        """Test mark_completed method."""
        from aria.db.models.document import Document, DocumentStatus

        doc = Document()
        doc.status = DocumentStatus.PROCESSING.value
        doc.error_message = "previous error"

        doc.mark_completed(chunk_count=42)

        assert doc.status == DocumentStatus.COMPLETED.value
        assert doc.chunk_count == 42
        assert doc.error_message is None

    def test_mark_failed(self) -> None:
        """Test mark_failed method."""
        from aria.db.models.document import Document, DocumentStatus

        doc = Document()
        doc.status = DocumentStatus.PROCESSING.value

        doc.mark_failed("Processing failed: timeout")

        assert doc.status == DocumentStatus.FAILED.value
        assert doc.error_message == "Processing failed: timeout"


class TestChunkModel:
    """Tests for Chunk model."""

    def test_chunk_repr(self) -> None:
        """Test Chunk __repr__ method."""
        from aria.db.models.chunk import Chunk

        chunk = Chunk()
        chunk.id = "chunk-id-123"
        chunk.chunk_index = 5
        chunk.content = "This is some test content that is long enough for the preview."

        repr_str = repr(chunk)
        assert "Chunk" in repr_str
        assert "chunk-id-123" in repr_str
        assert "5" in repr_str

    def test_chunk_repr_with_short_content(self) -> None:
        """Test Chunk __repr__ with content shorter than preview."""
        from aria.db.models.chunk import Chunk

        chunk = Chunk()
        chunk.id = "chunk-1"
        chunk.chunk_index = 0
        chunk.content = "Short"

        repr_str = repr(chunk)
        assert "Short" in repr_str

    def test_chunk_has_embedding_true(self) -> None:
        """Test has_embedding returns True when embedding exists."""
        from aria.db.models.chunk import Chunk

        chunk = Chunk()
        chunk.embedding = [0.1, 0.2, 0.3]

        assert chunk.has_embedding is True

    def test_chunk_has_embedding_false(self) -> None:
        """Test has_embedding returns False when no embedding."""
        from aria.db.models.chunk import Chunk

        chunk = Chunk()
        chunk.embedding = None

        assert chunk.has_embedding is False


class TestConversationModel:
    """Tests for Conversation model."""

    def test_conversation_repr(self) -> None:
        """Test Conversation __repr__ method."""
        from aria.db.models.conversation import Conversation

        conv = Conversation()
        conv.id = "conv-123"
        conv.title = "Test Conversation Title"

        repr_str = repr(conv)
        assert "Conversation" in repr_str
        assert "conv-123" in repr_str

    def test_conversation_message_count_property(self) -> None:
        """Test message_count property."""
        from aria.db.models.conversation import Conversation

        conv = Conversation()
        # SQLAlchemy initializes this as an empty list
        # We can test with an empty conversation
        assert conv.message_count == 0


class TestMessageModel:
    """Tests for Message model."""

    def test_message_role_values(self) -> None:
        """Test MessageRole enum values."""
        from aria.db.models.conversation import MessageRole

        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"

    def test_message_repr(self) -> None:
        """Test Message __repr__ method."""
        from aria.db.models.conversation import Message, MessageRole

        msg = Message()
        msg.id = "msg-123"
        msg.role = MessageRole.USER.value
        msg.content = "Hello, how are you doing today?"

        repr_str = repr(msg)
        assert "Message" in repr_str
        assert "msg-123" in repr_str
        assert "user" in repr_str

    def test_message_is_user_message(self) -> None:
        """Test is_user_message property."""
        from aria.db.models.conversation import Message, MessageRole

        msg = Message()
        msg.role = MessageRole.USER.value
        assert msg.is_user_message is True

        msg.role = MessageRole.ASSISTANT.value
        assert msg.is_user_message is False

    def test_message_is_assistant_message(self) -> None:
        """Test is_assistant_message property."""
        from aria.db.models.conversation import Message, MessageRole

        msg = Message()
        msg.role = MessageRole.ASSISTANT.value
        assert msg.is_assistant_message is True

        msg.role = MessageRole.USER.value
        assert msg.is_assistant_message is False

    def test_message_has_citations(self) -> None:
        """Test has_citations property."""
        from aria.db.models.conversation import Message

        msg = Message()
        msg.citation_ids = None
        assert msg.has_citations is False

        msg.citation_ids = []
        assert msg.has_citations is False

        msg.citation_ids = ["doc-1", "doc-2"]
        assert msg.has_citations is True


class TestModelExports:
    """Tests for model module exports."""

    def test_document_exported(self) -> None:
        """Test that Document is exported."""
        from aria.db.models import Document

        assert Document is not None

    def test_chunk_exported(self) -> None:
        """Test that Chunk is exported."""
        from aria.db.models import Chunk

        assert Chunk is not None

    def test_conversation_exported(self) -> None:
        """Test that Conversation is exported."""
        from aria.db.models import Conversation

        assert Conversation is not None

    def test_message_exported(self) -> None:
        """Test that Message is exported."""
        from aria.db.models import Message

        assert Message is not None
