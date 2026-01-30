"""Custom exception hierarchy for ARIA.

All application-specific exceptions inherit from ARIAError.
"""


class ARIAError(Exception):
    """Base exception for all ARIA errors.

    Attributes:
        message: Human-readable error message.
        code: Optional error code for programmatic handling.
        details: Optional additional details about the error.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize ARIA error.

        Args:
            message: Human-readable error message.
            code: Optional error code for programmatic handling.
            details: Optional additional details about the error.
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


# =========================
# Document Processing Errors
# =========================


class DocumentError(ARIAError):
    """Base exception for document-related errors."""

    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a document is not found."""

    def __init__(self, document_id: str) -> None:
        """Initialize error.

        Args:
            document_id: ID of the document that was not found.
        """
        super().__init__(
            message=f"Document not found: {document_id}",
            code="DOCUMENT_NOT_FOUND",
            details={"document_id": document_id},
        )


class DocumentParsingError(DocumentError):
    """Raised when document parsing fails."""

    def __init__(self, filename: str, reason: str) -> None:
        """Initialize error.

        Args:
            filename: Name of the file that failed to parse.
            reason: Reason for parsing failure.
        """
        super().__init__(
            message=f"Failed to parse document '{filename}': {reason}",
            code="DOCUMENT_PARSING_ERROR",
            details={"filename": filename, "reason": reason},
        )


class UnsupportedFileTypeError(DocumentError):
    """Raised when file type is not supported."""

    def __init__(self, file_type: str, supported_types: list[str]) -> None:
        """Initialize error.

        Args:
            file_type: The unsupported file type.
            supported_types: List of supported file types.
        """
        super().__init__(
            message=f"Unsupported file type: {file_type}. Supported: {', '.join(supported_types)}",
            code="UNSUPPORTED_FILE_TYPE",
            details={"file_type": file_type, "supported_types": supported_types},
        )


# =========================
# RAG Pipeline Errors
# =========================


class RAGError(ARIAError):
    """Base exception for RAG pipeline errors."""

    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""

    def __init__(self, reason: str) -> None:
        """Initialize error.

        Args:
            reason: Reason for embedding failure.
        """
        super().__init__(
            message=f"Embedding generation failed: {reason}",
            code="EMBEDDING_ERROR",
            details={"reason": reason},
        )


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""

    def __init__(self, reason: str) -> None:
        """Initialize error.

        Args:
            reason: Reason for retrieval failure.
        """
        super().__init__(
            message=f"Document retrieval failed: {reason}",
            code="RETRIEVAL_ERROR",
            details={"reason": reason},
        )


class SynthesisError(RAGError):
    """Raised when response synthesis fails."""

    def __init__(self, reason: str) -> None:
        """Initialize error.

        Args:
            reason: Reason for synthesis failure.
        """
        super().__init__(
            message=f"Response synthesis failed: {reason}",
            code="SYNTHESIS_ERROR",
            details={"reason": reason},
        )


# =========================
# Search/Connector Errors
# =========================


class SearchError(ARIAError):
    """Base exception for search-related errors."""

    pass


class ConnectorError(SearchError):
    """Raised when an external connector fails."""

    def __init__(self, connector: str, reason: str) -> None:
        """Initialize error.

        Args:
            connector: Name of the connector that failed.
            reason: Reason for failure.
        """
        super().__init__(
            message=f"Connector '{connector}' failed: {reason}",
            code="CONNECTOR_ERROR",
            details={"connector": connector, "reason": reason},
        )


class RateLimitError(SearchError):
    """Raised when rate limit is exceeded."""

    def __init__(self, service: str, retry_after: int | None = None) -> None:
        """Initialize error.

        Args:
            service: Name of the rate-limited service.
            retry_after: Seconds to wait before retrying.
        """
        super().__init__(
            message=f"Rate limit exceeded for {service}",
            code="RATE_LIMIT_ERROR",
            details={"service": service, "retry_after": retry_after},
        )


# =========================
# LLM Errors
# =========================


class LLMError(ARIAError):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when LLM API connection fails."""

    def __init__(self, provider: str, reason: str) -> None:
        """Initialize error.

        Args:
            provider: LLM provider name.
            reason: Reason for connection failure.
        """
        super().__init__(
            message=f"Failed to connect to {provider}: {reason}",
            code="LLM_CONNECTION_ERROR",
            details={"provider": provider, "reason": reason},
        )


class LLMResponseError(LLMError):
    """Raised when LLM returns an invalid response."""

    def __init__(self, reason: str) -> None:
        """Initialize error.

        Args:
            reason: Reason for invalid response.
        """
        super().__init__(
            message=f"Invalid LLM response: {reason}",
            code="LLM_RESPONSE_ERROR",
            details={"reason": reason},
        )


class ContextLengthExceededError(LLMError):
    """Raised when context length is exceeded."""

    def __init__(self, max_tokens: int, actual_tokens: int) -> None:
        """Initialize error.

        Args:
            max_tokens: Maximum allowed tokens.
            actual_tokens: Actual token count.
        """
        super().__init__(
            message=f"Context length exceeded: {actual_tokens} tokens (max: {max_tokens})",
            code="CONTEXT_LENGTH_EXCEEDED",
            details={"max_tokens": max_tokens, "actual_tokens": actual_tokens},
        )


# =========================
# Conversation Errors
# =========================


class ConversationError(ARIAError):
    """Base exception for conversation-related errors."""

    pass


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation is not found."""

    def __init__(self, conversation_id: str) -> None:
        """Initialize error.

        Args:
            conversation_id: ID of the conversation that was not found.
        """
        super().__init__(
            message=f"Conversation not found: {conversation_id}",
            code="CONVERSATION_NOT_FOUND",
            details={"conversation_id": conversation_id},
        )


# =========================
# Validation Errors
# =========================


class ValidationError(ARIAError):
    """Raised when validation fails."""

    def __init__(self, field: str, reason: str) -> None:
        """Initialize error.

        Args:
            field: Field that failed validation.
            reason: Reason for validation failure.
        """
        super().__init__(
            message=f"Validation failed for '{field}': {reason}",
            code="VALIDATION_ERROR",
            details={"field": field, "reason": reason},
        )


# =========================
# Configuration Errors
# =========================


class ConfigurationError(ARIAError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, setting: str, reason: str) -> None:
        """Initialize error.

        Args:
            setting: Configuration setting name.
            reason: Reason for configuration error.
        """
        super().__init__(
            message=f"Configuration error for '{setting}': {reason}",
            code="CONFIGURATION_ERROR",
            details={"setting": setting, "reason": reason},
        )
