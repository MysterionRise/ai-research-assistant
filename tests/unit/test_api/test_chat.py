"""Unit tests for chat endpoints and models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_chat_message_creation(self) -> None:
        """Test creating ChatMessage."""
        from aria.api.routes.chat import ChatMessage

        message = ChatMessage(
            role="user",
            content="Hello, how are you?",
        )
        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert message.timestamp is not None

    def test_chat_message_with_timestamp(self) -> None:
        """Test ChatMessage with explicit timestamp."""
        from aria.api.routes.chat import ChatMessage

        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        message = ChatMessage(
            role="assistant",
            content="I'm doing well!",
            timestamp=ts,
        )
        assert message.timestamp == ts


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_minimal_chat_request(self) -> None:
        """Test minimal ChatRequest."""
        from aria.api.routes.chat import ChatRequest

        request = ChatRequest(message="What is the meaning of life?")
        assert request.message == "What is the meaning of life?"
        assert request.conversation_id is None
        assert request.temperature == 0.7
        assert request.max_tokens == 4096

    def test_chat_request_with_all_options(self) -> None:
        """Test ChatRequest with all options."""
        from aria.api.routes.chat import ChatRequest

        conv_id = uuid4()
        request = ChatRequest(
            message="Hello!",
            conversation_id=conv_id,
            model="claude-3-opus",
            temperature=0.5,
            max_tokens=1000,
        )
        assert request.conversation_id == conv_id
        assert request.model == "claude-3-opus"
        assert request.temperature == 0.5
        assert request.max_tokens == 1000

    def test_chat_request_empty_message_fails(self) -> None:
        """Test that empty message is rejected."""
        from aria.api.routes.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_request_temperature_bounds(self) -> None:
        """Test temperature validation bounds."""
        from aria.api.routes.chat import ChatRequest

        # Valid temperature
        request = ChatRequest(message="test", temperature=2.0)
        assert request.temperature == 2.0

        # Invalid: > 2.0
        with pytest.raises(ValidationError):
            ChatRequest(message="test", temperature=2.5)

        # Invalid: < 0
        with pytest.raises(ValidationError):
            ChatRequest(message="test", temperature=-0.1)

    def test_chat_request_max_tokens_bounds(self) -> None:
        """Test max_tokens validation bounds."""
        from aria.api.routes.chat import ChatRequest

        # Valid max_tokens
        request = ChatRequest(message="test", max_tokens=100000)
        assert request.max_tokens == 100000

        # Invalid: > 100000
        with pytest.raises(ValidationError):
            ChatRequest(message="test", max_tokens=100001)

        # Invalid: < 1
        with pytest.raises(ValidationError):
            ChatRequest(message="test", max_tokens=0)


class TestCitation:
    """Tests for Citation model."""

    def test_citation_creation(self) -> None:
        """Test creating Citation."""
        from aria.api.routes.chat import Citation

        citation = Citation(
            document_id="doc-123",
            title="Important Paper",
            excerpt="This is a relevant excerpt from the paper.",
            confidence=0.95,
        )
        assert citation.document_id == "doc-123"
        assert citation.page is None
        assert citation.confidence == 0.95

    def test_citation_with_page(self) -> None:
        """Test Citation with page number."""
        from aria.api.routes.chat import Citation

        citation = Citation(
            document_id="doc-456",
            title="Research Paper",
            excerpt="Quote from page 42.",
            page=42,
            confidence=0.88,
        )
        assert citation.page == 42

    def test_citation_confidence_validation(self) -> None:
        """Test confidence validation bounds."""
        from aria.api.routes.chat import Citation

        # Valid confidence
        citation = Citation(
            document_id="doc",
            title="Title",
            excerpt="Text",
            confidence=1.0,
        )
        assert citation.confidence == 1.0

        # Invalid: > 1
        with pytest.raises(ValidationError):
            Citation(
                document_id="doc",
                title="Title",
                excerpt="Text",
                confidence=1.5,
            )


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_chat_response_creation(self) -> None:
        """Test creating ChatResponse."""
        from aria.api.routes.chat import ChatMessage, ChatResponse

        conv_id = uuid4()
        message = ChatMessage(role="assistant", content="Here's my response.")

        response = ChatResponse(
            conversation_id=conv_id,
            message=message,
        )
        assert response.conversation_id == conv_id
        assert response.citations == []
        assert response.metadata == {}

    def test_chat_response_with_citations(self) -> None:
        """Test ChatResponse with citations."""
        from aria.api.routes.chat import ChatMessage, ChatResponse, Citation

        conv_id = uuid4()
        message = ChatMessage(
            role="assistant",
            content="Based on [1], the answer is...",
        )
        citations = [
            Citation(
                document_id="doc-1",
                title="Source Paper",
                excerpt="Relevant text.",
                confidence=0.9,
            ),
        ]

        response = ChatResponse(
            conversation_id=conv_id,
            message=message,
            citations=citations,
            metadata={"tokens_used": 150, "latency_ms": 500},
        )
        assert len(response.citations) == 1
        assert response.metadata["tokens_used"] == 150
