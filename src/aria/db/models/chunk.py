"""Chunk model for storing document chunks with embeddings."""

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aria.config.settings import settings
from aria.db.base import Base, TimestampMixin, UUIDMixin

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for type checking or when pgvector is not installed
    Vector = None  # type: ignore[assignment, misc]


class Chunk(Base, UUIDMixin, TimestampMixin):
    """Model for storing document chunks with vector embeddings.

    Each chunk represents a semantically coherent piece of text from a document,
    along with its vector embedding for similarity search.

    Attributes:
        id: Unique chunk identifier (UUID).
        document_id: Foreign key to parent document.
        content: The text content of the chunk.
        chunk_index: Position of chunk in document (0-indexed).
        token_count: Number of tokens in the chunk.
        section: Section name if detected (e.g., "Abstract", "Methods").
        page_number: Page number in original document.
        embedding: Vector embedding for similarity search.
        metadata_: Additional chunk metadata.
        document: Parent document relationship.
    """

    __tablename__ = "chunks"

    # Foreign key to document
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Context information
    section: Mapped[str | None] = mapped_column(String(100), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Start and end positions in original text
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Vector embedding - using pgvector
    # Dimension matches OpenAI text-embedding-3-small (1536)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension) if Vector else None,  # type: ignore[misc]
        nullable=True,
    )

    # Flexible metadata storage
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    # Relationships
    document: Mapped["Document"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Document",
        back_populates="chunks",
    )

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_section", "section"),
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
        # Vector similarity index - IVFFlat for approximate nearest neighbor
        # This will be created via migration after data is loaded
    )

    def __repr__(self) -> str:
        """Return string representation."""
        content_preview = self.content[:50] if self.content else ""
        return f"<Chunk(id={self.id}, index={self.chunk_index}, content='{content_preview}...')>"

    @property
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding."""
        return self.embedding is not None
