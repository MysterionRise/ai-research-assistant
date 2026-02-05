"""Unit tests for custom exception hierarchy."""

import pytest

from aria.exceptions import (
    ARIAError,
    ConfigurationError,
    ConnectorError,
    ContextLengthExceededError,
    ConversationError,
    ConversationNotFoundError,
    DocumentError,
    DocumentNotFoundError,
    DocumentParsingError,
    EmbeddingError,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    RAGError,
    RateLimitError,
    RetrievalError,
    SearchError,
    SynthesisError,
    UnsupportedFileTypeError,
    ValidationError,
)


class TestARIAError:
    """Tests for base ARIAError class."""

    def test_basic_initialization(self) -> None:
        """Test basic error initialization."""
        error = ARIAError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.code is None
        assert error.details == {}

    def test_initialization_with_code(self) -> None:
        """Test error initialization with code."""
        error = ARIAError("Error occurred", code="ERR_001")
        assert error.message == "Error occurred"
        assert error.code == "ERR_001"

    def test_initialization_with_details(self) -> None:
        """Test error initialization with details."""
        error = ARIAError(
            "Error with context",
            code="ERR_002",
            details={"key": "value", "count": 42},
        )
        assert error.details["key"] == "value"
        assert error.details["count"] == 42

    def test_str_representation(self) -> None:
        """Test string representation."""
        error = ARIAError("Test message")
        assert str(error) == "Test message"

    def test_inherits_from_exception(self) -> None:
        """Test that ARIAError inherits from Exception."""
        error = ARIAError("Test")
        assert isinstance(error, Exception)


class TestDocumentErrors:
    """Tests for document-related errors."""

    def test_document_error_is_aria_error(self) -> None:
        """Test that DocumentError inherits from ARIAError."""
        error = DocumentError("Doc error")
        assert isinstance(error, ARIAError)

    def test_document_not_found_error(self) -> None:
        """Test DocumentNotFoundError initialization."""
        error = DocumentNotFoundError("doc-123")
        assert "doc-123" in error.message
        assert error.code == "DOCUMENT_NOT_FOUND"
        assert error.details["document_id"] == "doc-123"

    def test_document_not_found_inherits_document_error(self) -> None:
        """Test inheritance hierarchy."""
        error = DocumentNotFoundError("doc-1")
        assert isinstance(error, DocumentError)
        assert isinstance(error, ARIAError)

    def test_document_parsing_error(self) -> None:
        """Test DocumentParsingError initialization."""
        error = DocumentParsingError("paper.pdf", "Corrupted file")
        assert "paper.pdf" in error.message
        assert "Corrupted file" in error.message
        assert error.code == "DOCUMENT_PARSING_ERROR"
        assert error.details["filename"] == "paper.pdf"
        assert error.details["reason"] == "Corrupted file"

    def test_unsupported_file_type_error(self) -> None:
        """Test UnsupportedFileTypeError initialization."""
        supported = ["pdf", "docx", "txt"]
        error = UnsupportedFileTypeError("exe", supported)
        assert "exe" in error.message
        assert error.code == "UNSUPPORTED_FILE_TYPE"
        assert error.details["file_type"] == "exe"
        assert error.details["supported_types"] == supported


class TestRAGErrors:
    """Tests for RAG pipeline errors."""

    def test_rag_error_is_aria_error(self) -> None:
        """Test that RAGError inherits from ARIAError."""
        error = RAGError("RAG error")
        assert isinstance(error, ARIAError)

    def test_embedding_error(self) -> None:
        """Test EmbeddingError initialization."""
        error = EmbeddingError("Model not available")
        assert "Model not available" in error.message
        assert error.code == "EMBEDDING_ERROR"
        assert error.details["reason"] == "Model not available"

    def test_embedding_error_inherits_rag_error(self) -> None:
        """Test inheritance hierarchy."""
        error = EmbeddingError("Test")
        assert isinstance(error, RAGError)

    def test_retrieval_error(self) -> None:
        """Test RetrievalError initialization."""
        error = RetrievalError("Vector store unavailable")
        assert "Vector store unavailable" in error.message
        assert error.code == "RETRIEVAL_ERROR"
        assert error.details["reason"] == "Vector store unavailable"

    def test_synthesis_error(self) -> None:
        """Test SynthesisError initialization."""
        error = SynthesisError("LLM timeout")
        assert "LLM timeout" in error.message
        assert error.code == "SYNTHESIS_ERROR"
        assert error.details["reason"] == "LLM timeout"


class TestSearchErrors:
    """Tests for search-related errors."""

    def test_search_error_is_aria_error(self) -> None:
        """Test that SearchError inherits from ARIAError."""
        error = SearchError("Search failed")
        assert isinstance(error, ARIAError)

    def test_connector_error(self) -> None:
        """Test ConnectorError initialization."""
        error = ConnectorError("PubMed", "API rate limited")
        assert "PubMed" in error.message
        assert "API rate limited" in error.message
        assert error.code == "CONNECTOR_ERROR"
        assert error.details["connector"] == "PubMed"
        assert error.details["reason"] == "API rate limited"

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError initialization."""
        error = RateLimitError("Semantic Scholar", retry_after=60)
        assert "Semantic Scholar" in error.message
        assert error.code == "RATE_LIMIT_ERROR"
        assert error.details["service"] == "Semantic Scholar"
        assert error.details["retry_after"] == 60

    def test_rate_limit_error_without_retry(self) -> None:
        """Test RateLimitError without retry_after."""
        error = RateLimitError("PubMed")
        assert error.details["retry_after"] is None


class TestLLMErrors:
    """Tests for LLM-related errors."""

    def test_llm_error_is_aria_error(self) -> None:
        """Test that LLMError inherits from ARIAError."""
        error = LLMError("LLM error")
        assert isinstance(error, ARIAError)

    def test_llm_connection_error(self) -> None:
        """Test LLMConnectionError initialization."""
        error = LLMConnectionError("anthropic", "Connection refused")
        assert "anthropic" in error.message
        assert "Connection refused" in error.message
        assert error.code == "LLM_CONNECTION_ERROR"
        assert error.details["provider"] == "anthropic"
        assert error.details["reason"] == "Connection refused"

    def test_llm_response_error(self) -> None:
        """Test LLMResponseError initialization."""
        error = LLMResponseError("Empty content block")
        assert "Empty content block" in error.message
        assert error.code == "LLM_RESPONSE_ERROR"
        assert error.details["reason"] == "Empty content block"

    def test_context_length_exceeded_error(self) -> None:
        """Test ContextLengthExceededError initialization."""
        error = ContextLengthExceededError(max_tokens=100000, actual_tokens=150000)
        assert "150000" in error.message
        assert "100000" in error.message
        assert error.code == "CONTEXT_LENGTH_EXCEEDED"
        assert error.details["max_tokens"] == 100000
        assert error.details["actual_tokens"] == 150000


class TestConversationErrors:
    """Tests for conversation-related errors."""

    def test_conversation_error_is_aria_error(self) -> None:
        """Test that ConversationError inherits from ARIAError."""
        error = ConversationError("Conversation error")
        assert isinstance(error, ARIAError)

    def test_conversation_not_found_error(self) -> None:
        """Test ConversationNotFoundError initialization."""
        error = ConversationNotFoundError("conv-123")
        assert "conv-123" in error.message
        assert error.code == "CONVERSATION_NOT_FOUND"
        assert error.details["conversation_id"] == "conv-123"


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error(self) -> None:
        """Test ValidationError initialization."""
        error = ValidationError("email", "Invalid email format")
        assert "email" in error.message
        assert "Invalid email format" in error.message
        assert error.code == "VALIDATION_ERROR"
        assert error.details["field"] == "email"
        assert error.details["reason"] == "Invalid email format"

    def test_validation_error_is_aria_error(self) -> None:
        """Test that ValidationError inherits from ARIAError."""
        error = ValidationError("field", "reason")
        assert isinstance(error, ARIAError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError initialization."""
        error = ConfigurationError("DATABASE_URL", "Missing required setting")
        assert "DATABASE_URL" in error.message
        assert "Missing required setting" in error.message
        assert error.code == "CONFIGURATION_ERROR"
        assert error.details["setting"] == "DATABASE_URL"
        assert error.details["reason"] == "Missing required setting"

    def test_configuration_error_is_aria_error(self) -> None:
        """Test that ConfigurationError inherits from ARIAError."""
        error = ConfigurationError("setting", "reason")
        assert isinstance(error, ARIAError)


class TestExceptionHierarchy:
    """Tests for the complete exception hierarchy."""

    def test_all_errors_can_be_caught_as_aria_error(self) -> None:
        """Test that all custom errors can be caught as ARIAError."""
        errors = [
            DocumentNotFoundError("doc-1"),
            DocumentParsingError("file.pdf", "reason"),
            UnsupportedFileTypeError("exe", ["pdf"]),
            EmbeddingError("reason"),
            RetrievalError("reason"),
            SynthesisError("reason"),
            ConnectorError("conn", "reason"),
            RateLimitError("service"),
            LLMConnectionError("provider", "reason"),
            LLMResponseError("reason"),
            ContextLengthExceededError(100, 200),
            ConversationNotFoundError("conv-1"),
            ValidationError("field", "reason"),
            ConfigurationError("setting", "reason"),
        ]

        for error in errors:
            assert isinstance(error, ARIAError)

    def test_document_errors_hierarchy(self) -> None:
        """Test document error hierarchy."""
        doc_errors = [
            DocumentNotFoundError("doc-1"),
            DocumentParsingError("file.pdf", "reason"),
            UnsupportedFileTypeError("exe", ["pdf"]),
        ]
        for error in doc_errors:
            assert isinstance(error, DocumentError)
            assert isinstance(error, ARIAError)

    def test_rag_errors_hierarchy(self) -> None:
        """Test RAG error hierarchy."""
        rag_errors = [
            EmbeddingError("reason"),
            RetrievalError("reason"),
            SynthesisError("reason"),
        ]
        for error in rag_errors:
            assert isinstance(error, RAGError)
            assert isinstance(error, ARIAError)

    def test_search_errors_hierarchy(self) -> None:
        """Test search error hierarchy."""
        search_errors = [
            ConnectorError("conn", "reason"),
            RateLimitError("service"),
        ]
        for error in search_errors:
            assert isinstance(error, SearchError)
            assert isinstance(error, ARIAError)

    def test_llm_errors_hierarchy(self) -> None:
        """Test LLM error hierarchy."""
        llm_errors = [
            LLMConnectionError("provider", "reason"),
            LLMResponseError("reason"),
            ContextLengthExceededError(100, 200),
        ]
        for error in llm_errors:
            assert isinstance(error, LLMError)
            assert isinstance(error, ARIAError)

    def test_errors_can_be_raised_and_caught(self) -> None:
        """Test that errors can be raised and caught properly."""
        with pytest.raises(DocumentNotFoundError) as exc_info:
            raise DocumentNotFoundError("test-doc")

        assert exc_info.value.details["document_id"] == "test-doc"

        # Can also be caught as parent type
        with pytest.raises(ARIAError):
            raise DocumentNotFoundError("test-doc")
