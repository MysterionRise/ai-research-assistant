"""Conversation and Message models for chat history."""

from enum import StrEnum

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aria.db.base import Base, TimestampMixin, UUIDMixin


class MessageRole(StrEnum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base, UUIDMixin, TimestampMixin):
    """Model for storing chat conversations.

    Attributes:
        id: Unique conversation identifier (UUID).
        title: Optional conversation title.
        user_id: Optional user identifier for multi-user support.
        metadata_: Additional conversation metadata.
        messages: List of messages in the conversation.
    """

    __tablename__ = "conversations"

    # Optional title (auto-generated or user-provided)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # User identifier for multi-user support
    user_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Conversation state
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    # Flexible metadata storage
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    # Relationships
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
        Index("ix_conversations_created_at", "created_at"),
        Index("ix_conversations_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        title_preview = self.title[:30] if self.title else "Untitled"
        return f"<Conversation(id={self.id}, title='{title_preview}')>"

    @property
    def message_count(self) -> int:
        """Get the number of messages in this conversation."""
        return len(self.messages)

    def get_last_message(self) -> "Message | None":
        """Get the most recent message.

        Returns:
            Most recent message or None if no messages.
        """
        return self.messages[-1] if self.messages else None


class Message(Base, UUIDMixin, TimestampMixin):
    """Model for storing individual chat messages.

    Attributes:
        id: Unique message identifier (UUID).
        conversation_id: Foreign key to parent conversation.
        role: Message role (user, assistant, system).
        content: Message text content.
        citations: List of citation references.
        metadata_: Additional message metadata (tokens, latency, model).
        conversation: Parent conversation relationship.
    """

    __tablename__ = "messages"

    # Foreign key to conversation
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Message content
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Citation references (list of document/chunk IDs)
    citation_ids: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )

    # Model information
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(nullable=True)

    # Flexible metadata storage (citations detail, confidence, etc.)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_role", "role"),
        Index("ix_messages_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        content_preview = self.content[:50] if self.content else ""
        return f"<Message(id={self.id}, role={self.role}, content='{content_preview}...')>"

    @property
    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.role == MessageRole.USER.value

    @property
    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message."""
        return self.role == MessageRole.ASSISTANT.value

    @property
    def has_citations(self) -> bool:
        """Check if message has citations."""
        return bool(self.citation_ids)
