"""Chat endpoints.

Provides conversational AI interface for research queries.
"""

import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from aria.api.dependencies import DBSession, RAGPipelineDep
from aria.db.models import Conversation, Message
from aria.db.models.conversation import MessageRole

router = APIRouter(prefix="/chat")
logger = structlog.get_logger(__name__)


# =========================
# Request/Response Models
# =========================


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Message timestamp",
    )


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_id: UUID | None = Field(
        default=None,
        description="Existing conversation ID for context",
    )
    model: str | None = Field(
        default=None,
        description="Model override (default: claude-sonnet-4-20250514)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Response temperature",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=100000,
        description="Maximum response tokens",
    )


class Citation(BaseModel):
    """A citation reference."""

    document_id: str = Field(..., description="Source document ID")
    title: str = Field(..., description="Document title")
    excerpt: str = Field(..., description="Relevant excerpt")
    page: int | None = Field(default=None, description="Page number if applicable")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relevance confidence")


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    conversation_id: UUID = Field(..., description="Conversation ID")
    message: ChatMessage = Field(..., description="Assistant response")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata",
    )


# =========================
# Endpoints
# =========================


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Send a message to the AI research assistant and receive a response.",
)
async def send_message(
    request: ChatRequest,
    session: DBSession,
    rag_pipeline: RAGPipelineDep,
) -> ChatResponse:
    """Send a message to the AI assistant.

    Args:
        request: Chat request with message and options.
        session: Database session.
        rag_pipeline: RAG pipeline for retrieval and synthesis.

    Returns:
        ChatResponse: AI response with citations.
    """
    start_time = time.time()

    # Get or create conversation
    if request.conversation_id:
        result = await session.execute(
            select(Conversation).where(Conversation.id == str(request.conversation_id))
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {request.conversation_id} not found",
            )
    else:
        conversation = Conversation(
            title=request.message[:100],  # Use first 100 chars as title
        )
        session.add(conversation)
        await session.flush()

    # Store user message
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER.value,
        content=request.message,
    )
    session.add(user_message)

    # Run RAG pipeline
    logger.info("processing_chat_request", conversation_id=conversation.id)

    rag_result = await rag_pipeline.query(
        question=request.message,
        max_tokens=request.max_tokens,
    )

    # Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)

    # Store assistant message
    assistant_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT.value,
        content=rag_result.answer,
        citation_ids=[c.document_id for c in rag_result.citations],
        tokens_used=rag_result.metadata.get("tokens_used", 0),
        latency_ms=latency_ms,
        metadata_={
            "confidence": rag_result.confidence,
            "citations": [
                {
                    "document_id": c.document_id,
                    "title": c.title,
                    "page": c.page,
                }
                for c in rag_result.citations
            ],
        },
    )
    session.add(assistant_message)

    await session.commit()

    # Build response
    response_message = ChatMessage(
        role="assistant",
        content=rag_result.answer,
    )

    citations = [
        Citation(
            document_id=c.document_id,
            title=c.title,
            excerpt=c.excerpt,
            page=c.page,
            confidence=c.confidence,
        )
        for c in rag_result.citations
    ]

    logger.info(
        "chat_request_completed",
        conversation_id=conversation.id,
        latency_ms=latency_ms,
        citations_count=len(citations),
    )

    return ChatResponse(
        conversation_id=UUID(conversation.id),
        message=response_message,
        citations=citations,
        metadata={
            "model": rag_result.metadata.get("model"),
            "tokens_used": rag_result.metadata.get("tokens_used", 0),
            "latency_ms": latency_ms,
            "confidence": rag_result.confidence,
        },
    )


@router.get(
    "/{conversation_id}",
    response_model=list[ChatMessage],
    status_code=status.HTTP_200_OK,
    summary="Get conversation history",
    description="Retrieve the message history for a conversation.",
)
async def get_conversation(
    conversation_id: UUID,
    session: DBSession,
) -> list[ChatMessage]:
    """Get conversation history.

    Args:
        conversation_id: UUID of the conversation.
        session: Database session.

    Returns:
        List of chat messages in the conversation.

    Raises:
        HTTPException: If conversation not found.
    """
    result = await session.execute(
        select(Conversation).where(Conversation.id == str(conversation_id))
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    return [
        ChatMessage(
            role=msg.role,
            content=msg.content,
            timestamp=msg.created_at,
        )
        for msg in conversation.messages
    ]


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete conversation",
    description="Delete a conversation and its history.",
)
async def delete_conversation(
    conversation_id: UUID,
    session: DBSession,
) -> None:
    """Delete a conversation.

    Args:
        conversation_id: UUID of the conversation to delete.
        session: Database session.

    Raises:
        HTTPException: If conversation not found.
    """
    result = await session.execute(
        select(Conversation).where(Conversation.id == str(conversation_id))
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    await session.delete(conversation)
    await session.commit()

    logger.info("conversation_deleted", conversation_id=str(conversation_id))
