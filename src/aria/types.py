"""Shared type definitions for ARIA.

Contains Pydantic models, TypedDicts, and type aliases used across the application.
"""

from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

# =========================
# Type Aliases
# =========================

# Vector embedding type
Embedding = list[float]

# Generic type for models
T = TypeVar("T")


# =========================
# Enums
# =========================


class DocumentSource(StrEnum):
    """Source of a document."""

    INTERNAL = "internal"
    PUBMED = "pubmed"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class SearchSource(StrEnum):
    """Search source identifiers."""

    INTERNAL = "internal"
    PUBMED = "pubmed"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class ChunkType(StrEnum):
    """Type of document chunk."""

    TEXT = "text"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    EQUATION = "equation"
    CODE = "code"


# =========================
# Document Types
# =========================


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    title: str = Field(..., description="Document title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    year: int | None = Field(default=None, description="Publication year")
    journal: str | None = Field(default=None, description="Journal or venue")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    abstract: str | None = Field(default=None, description="Abstract text")
    keywords: list[str] = Field(default_factory=list, description="Keywords")
    source: DocumentSource = Field(
        default=DocumentSource.INTERNAL,
        description="Document source",
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom fields",
    )


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    document_id: str = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., description="Chunk position in document")
    section: str | None = Field(default=None, description="Section name")
    page_number: int | None = Field(default=None, description="Page number")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT, description="Chunk type")
    token_count: int = Field(..., description="Token count")


# =========================
# Search Types
# =========================


class SearchResult(BaseModel):
    """A single search result from any source."""

    id: str = Field(..., description="Result identifier")
    title: str = Field(..., description="Result title")
    abstract: str | None = Field(default=None, description="Abstract or summary")
    authors: list[str] = Field(default_factory=list, description="Authors")
    year: int | None = Field(default=None, description="Publication year")
    journal: str | None = Field(default=None, description="Journal or source")
    doi: str | None = Field(default=None, description="DOI")
    url: str | None = Field(default=None, description="Source URL")
    source: SearchSource = Field(..., description="Search source")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class RetrievedChunk(BaseModel):
    """A retrieved chunk with relevance score."""

    chunk_id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Relevance score")
    section: str | None = Field(default=None, description="Section name")
    page_number: int | None = Field(default=None, description="Page number")
    document_title: str | None = Field(default=None, description="Document title")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


# =========================
# Citation Types
# =========================


class Citation(BaseModel):
    """A citation reference in a response."""

    citation_id: int = Field(..., description="Citation number [1], [2], etc.")
    document_id: str = Field(..., description="Source document ID")
    chunk_id: str | None = Field(default=None, description="Source chunk ID")
    title: str = Field(..., description="Document title")
    excerpt: str = Field(..., description="Relevant excerpt")
    page: int | None = Field(default=None, description="Page number")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relevance confidence")


# =========================
# RAG Types
# =========================


class RAGContext(BaseModel):
    """Context assembled for RAG generation."""

    query: str = Field(..., description="Original user query")
    chunks: list[RetrievedChunk] = Field(..., description="Retrieved chunks")
    total_tokens: int = Field(..., description="Total context tokens")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata",
    )


class RAGResponse(BaseModel):
    """Response from the RAG pipeline."""

    answer: str = Field(..., description="Generated answer")
    citations: list[Citation] = Field(default_factory=list, description="Citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence")
    query: str = Field(..., description="Original query")
    sources_used: int = Field(..., description="Number of sources used")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata",
    )


# =========================
# Chat Types
# =========================


class ChatContext(BaseModel):
    """Context for a chat interaction."""

    conversation_id: str = Field(..., description="Conversation ID")
    message_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous messages in conversation",
    )
    rag_context: RAGContext | None = Field(
        default=None,
        description="RAG context if retrieval was performed",
    )


# =========================
# Evaluation Types
# =========================


class EvaluationResult(BaseModel):
    """Result from RAG evaluation."""

    query: str = Field(..., description="Test query")
    expected_answer: str | None = Field(default=None, description="Expected answer")
    generated_answer: str = Field(..., description="Generated answer")
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score")
    relevancy: float = Field(..., ge=0.0, le=1.0, description="Answer relevancy")
    context_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Context precision",
    )
    latency_ms: int = Field(..., description="Response latency in milliseconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation data",
    )


# =========================
# Processing Types
# =========================


class ProcessingResult(BaseModel):
    """Result from document processing."""

    document_id: str = Field(..., description="Document ID")
    success: bool = Field(..., description="Whether processing succeeded")
    chunk_count: int = Field(default=0, description="Number of chunks created")
    error: str | None = Field(default=None, description="Error message if failed")
    processing_time_ms: int = Field(..., description="Processing time")
    metadata: DocumentMetadata | None = Field(
        default=None,
        description="Extracted metadata",
    )
