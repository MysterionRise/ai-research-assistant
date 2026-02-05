"""Document model for storing ingested documents."""

from enum import StrEnum

from sqlalchemy import Index, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aria.db.base import Base, TimestampMixin, UUIDMixin


class DocumentStatus(StrEnum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base, UUIDMixin, TimestampMixin):
    """Model for storing document metadata and processing status.

    Attributes:
        id: Unique document identifier (UUID).
        title: Document title.
        filename: Original filename.
        file_type: MIME type of the document.
        file_size: File size in bytes.
        file_path: Path to stored file.
        status: Processing status.
        chunk_count: Number of chunks created.
        metadata_: Additional metadata (title, authors, DOI, etc.).
        error_message: Error message if processing failed.
        chunks: Related chunks.
    """

    __tablename__ = "documents"

    # Core fields
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Processing status
    status: Mapped[str] = mapped_column(
        String(50),
        default=DocumentStatus.PENDING.value,
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Rich metadata
    authors: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    year: Mapped[int | None] = mapped_column(nullable=True)
    journal: Mapped[str | None] = mapped_column(String(500), nullable=True)
    doi: Mapped[str | None] = mapped_column(String(100), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)

    # Flexible metadata storage
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_documents_status", "status"),
        Index("ix_documents_title", "title"),
        Index("ix_documents_doi", "doi"),
        Index("ix_documents_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Document(id={self.id}, title='{self.title[:30]}...')>"

    @property
    def is_processing(self) -> bool:
        """Check if document is being processed."""
        return self.status == DocumentStatus.PROCESSING.value

    @property
    def is_completed(self) -> bool:
        """Check if document processing is complete."""
        return self.status == DocumentStatus.COMPLETED.value

    @property
    def is_failed(self) -> bool:
        """Check if document processing failed."""
        return self.status == DocumentStatus.FAILED.value

    def mark_processing(self) -> None:
        """Mark document as processing."""
        self.status = DocumentStatus.PROCESSING.value

    def mark_completed(self, chunk_count: int) -> None:
        """Mark document as completed.

        Args:
            chunk_count: Number of chunks created.
        """
        self.status = DocumentStatus.COMPLETED.value
        self.chunk_count = chunk_count
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Mark document as failed.

        Args:
            error: Error message describing the failure.
        """
        self.status = DocumentStatus.FAILED.value
        self.error_message = error
