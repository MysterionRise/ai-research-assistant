"""Initial schema with documents, chunks, conversations, and messages.

Revision ID: 001
Revises:
Create Date: 2025-01-29

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(100), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.String(1000), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("authors", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("journal", sa.String(500), nullable=True),
        sa.Column("doi", sa.String(100), nullable=True),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_title", "documents", ["title"])
    op.create_index("ix_documents_doi", "documents", ["doi"])
    op.create_index("ix_documents_created_at", "documents", ["created_at"])

    # Create chunks table with vector column
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(100), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("start_char", sa.Integer(), nullable=True),
        sa.Column("end_char", sa.Integer(), nullable=True),
        sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_section", "chunks", ["section"])
    op.create_index("ix_chunks_document_index", "chunks", ["document_id", "chunk_index"])

    # Update chunks table to use pgvector type
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536)")

    # Create IVFFlat index for vector similarity search
    # Note: This requires data to be present for optimal list calculation
    # For initial setup, we use a smaller lists value
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("user_id", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index("ix_conversations_created_at", "conversations", ["created_at"])
    op.create_index("ix_conversations_is_active", "conversations", ["is_active"])

    # Create messages table
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citation_ids", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_role", "messages", ["role"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_index("ix_chunks_embedding", "chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
