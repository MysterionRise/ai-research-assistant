"""Document management endpoints.

Provides document ingestion, retrieval, and management capabilities.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import aiofiles
import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from aria.api.dependencies import DBSession
from aria.db.models import Chunk, Document
from aria.db.models.document import DocumentStatus
from aria.worker.tasks.ingestion import ingest_document

router = APIRouter(prefix="/documents")
logger = structlog.get_logger(__name__)

# Storage directory for uploaded files
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Request/Response Models
# =========================


class DocumentMetadata(BaseModel):
    """Document metadata."""

    title: str = Field(..., description="Document title")
    authors: list[str] = Field(default_factory=list, description="Authors")
    year: int | None = Field(default=None, description="Publication year")
    journal: str | None = Field(default=None, description="Journal or source")
    doi: str | None = Field(default=None, description="DOI")
    tags: list[str] = Field(default_factory=list, description="Tags")
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata fields",
    )


class DocumentResponse(BaseModel):
    """A document in the system."""

    id: UUID = Field(..., description="Document ID")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    file_type: str = Field(..., description="File type (pdf, docx, etc.)")
    file_size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(..., description="Number of chunks")
    status: str = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update")


class DocumentCreateRequest(BaseModel):
    """Request to create/update document metadata."""

    metadata: DocumentMetadata = Field(..., description="Document metadata")


class DocumentListResponse(BaseModel):
    """Response with list of documents."""

    total: int = Field(..., description="Total documents")
    documents: list[DocumentResponse] = Field(..., description="Document list")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


class IngestResponse(BaseModel):
    """Response from document ingestion."""

    document_id: UUID = Field(..., description="Created document ID")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class ChunkResponse(BaseModel):
    """Document chunk response."""

    id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in document")
    section: str | None = Field(default=None, description="Section name")
    page_number: int | None = Field(default=None, description="Page number")
    token_count: int = Field(..., description="Token count")


# =========================
# Endpoints
# =========================


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List documents",
    description="Get a paginated list of documents.",
)
async def list_documents(
    session: DBSession,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    status_filter: str | None = None,
    tags: list[str] | None = None,
) -> DocumentListResponse:
    """List documents with pagination.

    Args:
        session: Database session.
        page: Page number (1-indexed).
        page_size: Number of documents per page.
        search: Optional search query.
        status_filter: Optional status filter.
        tags: Optional tag filter.

    Returns:
        DocumentListResponse: Paginated document list.
    """
    # Build query
    query = select(Document)

    if search:
        query = query.where(Document.title.ilike(f"%{search}%"))

    if status_filter:
        query = query.where(Document.status == status_filter)

    if tags:
        query = query.where(Document.tags.overlap(tags))

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Document.created_at.desc())

    result = await session.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        total=total,
        documents=[_document_to_response(doc) for doc in documents],
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document",
    description="Get a document by ID.",
)
async def get_document(
    document_id: UUID,
    session: DBSession,
) -> DocumentResponse:
    """Get a document by ID.

    Args:
        document_id: Document UUID.
        session: Database session.

    Returns:
        DocumentResponse: Document details.

    Raises:
        HTTPException: If document not found.
    """
    result = await session.execute(
        select(Document).where(Document.id == str(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return _document_to_response(document)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest document",
    description="Upload and process a document for RAG.",
)
async def ingest_document_endpoint(
    session: DBSession,
    file: Annotated[UploadFile, File(description="Document file to ingest")],
    title: str | None = None,
) -> IngestResponse:
    """Ingest a document.

    Uploads a document and queues it for processing (chunking, embedding).

    Args:
        session: Database session.
        file: Uploaded file.
        title: Optional title override.

    Returns:
        IngestResponse: Ingestion status.

    Raises:
        HTTPException: If file type not supported.
    """
    # Validate file type
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
    }

    content_type = file.content_type or "application/octet-stream"
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {content_type} not supported. "
            f"Allowed: {', '.join(allowed_types)}",
        )

    # Generate document ID and file path
    document_id = str(uuid.uuid4())
    file_ext = Path(file.filename or "document").suffix
    file_path = UPLOAD_DIR / f"{document_id}{file_ext}"

    # Save file
    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
    except Exception as e:
        logger.error("file_save_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        ) from e

    # Create document record
    document = Document(
        id=document_id,
        title=title or file.filename or "Untitled Document",
        filename=file.filename or "document",
        file_type=content_type,
        file_size=len(content),
        file_path=str(file_path),
        status=DocumentStatus.PENDING.value,
    )
    session.add(document)
    await session.commit()

    # Queue for processing
    ingest_document.delay(document_id)

    logger.info(
        "document_ingestion_queued",
        document_id=document_id,
        filename=file.filename,
        file_size=len(content),
    )

    return IngestResponse(
        document_id=UUID(document_id),
        status="queued",
        message=f"Document '{file.filename}' queued for processing",
    )


@router.put(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Update document metadata",
    description="Update a document's metadata.",
)
async def update_document(
    document_id: UUID,
    request: DocumentCreateRequest,
    session: DBSession,
) -> DocumentResponse:
    """Update document metadata.

    Args:
        document_id: Document UUID.
        request: Updated metadata.
        session: Database session.

    Returns:
        DocumentResponse: Updated document.

    Raises:
        HTTPException: If document not found.
    """
    result = await session.execute(
        select(Document).where(Document.id == str(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Update fields
    document.title = request.metadata.title
    document.authors = request.metadata.authors
    document.year = request.metadata.year
    document.journal = request.metadata.journal
    document.doi = request.metadata.doi
    document.tags = request.metadata.tags
    document.metadata_ = request.metadata.custom_fields

    await session.commit()
    await session.refresh(document)

    logger.info("document_updated", document_id=str(document_id))

    return _document_to_response(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document and its embeddings.",
)
async def delete_document(
    document_id: UUID,
    session: DBSession,
) -> None:
    """Delete a document.

    Args:
        document_id: Document UUID.
        session: Database session.

    Raises:
        HTTPException: If document not found.
    """
    result = await session.execute(
        select(Document).where(Document.id == str(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Delete file if exists
    try:
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning("file_delete_failed", error=str(e))

    # Delete document (cascades to chunks)
    await session.delete(document)
    await session.commit()

    logger.info("document_deleted", document_id=str(document_id))


@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
    status_code=status.HTTP_200_OK,
    summary="Get document chunks",
    description="Get the text chunks for a document.",
)
async def get_document_chunks(
    document_id: UUID,
    session: DBSession,
) -> list[ChunkResponse]:
    """Get document chunks.

    Args:
        document_id: Document UUID.
        session: Database session.

    Returns:
        List of document chunks.

    Raises:
        HTTPException: If document not found.
    """
    # Verify document exists
    doc_result = await session.execute(
        select(Document).where(Document.id == str(document_id))
    )
    document = doc_result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Get chunks
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == str(document_id))
        .order_by(Chunk.chunk_index)
    )
    chunks = result.scalars().all()

    return [
        ChunkResponse(
            id=chunk.id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            section=chunk.section,
            page_number=chunk.page_number,
            token_count=chunk.token_count,
        )
        for chunk in chunks
    ]


def _document_to_response(document: Document) -> DocumentResponse:
    """Convert Document model to response schema."""
    return DocumentResponse(
        id=UUID(document.id),
        metadata=DocumentMetadata(
            title=document.title,
            authors=document.authors or [],
            year=document.year,
            journal=document.journal,
            doi=document.doi,
            tags=document.tags or [],
            custom_fields=document.metadata_ or {},
        ),
        file_type=document.file_type,
        file_size=document.file_size,
        chunk_count=document.chunk_count,
        status=document.status,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )
