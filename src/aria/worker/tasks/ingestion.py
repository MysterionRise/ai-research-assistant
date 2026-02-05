"""Document ingestion tasks."""

import asyncio
from pathlib import Path

import structlog
from celery import shared_task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aria.db.models import Chunk, Document
from aria.db.session import async_session_maker
from aria.document_processing.pipeline import DocumentProcessingPipeline
from aria.rag.chunking.semantic import SemanticChunker
from aria.rag.embedding.openai import OpenAIEmbedder

logger = structlog.get_logger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_document(self, document_id: str) -> dict:  # type: ignore[no-untyped-def]
    """Ingest a document: parse, chunk, and embed.

    Args:
        document_id: UUID of the document to process.

    Returns:
        Dict with processing status.
    """
    logger.info("starting_document_ingestion", document_id=document_id)

    try:
        # Run async processing
        result = asyncio.get_event_loop().run_until_complete(_process_document_async(document_id))
        return result
    except Exception as e:
        logger.error(
            "ingestion_failed",
            document_id=document_id,
            error=str(e),
        )
        # Mark document as failed
        asyncio.get_event_loop().run_until_complete(_mark_document_failed(document_id, str(e)))
        raise self.retry(exc=e) from e


async def _process_document_async(document_id: str) -> dict:
    """Async document processing logic.

    Args:
        document_id: Document UUID.

    Returns:
        Processing result dict.
    """
    async with async_session_maker() as session:
        # Get document
        result = await session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()

        if not document:
            raise ValueError(f"Document not found: {document_id}")

        # Mark as processing
        document.mark_processing()
        await session.commit()

        try:
            # Initialize pipeline
            pipeline = DocumentProcessingPipeline()
            chunker = SemanticChunker()
            embedder = OpenAIEmbedder()

            # Parse document
            file_path = Path(document.file_path)
            parsed, metadata, sections = await pipeline.process(
                file_path,
                document.file_type,
            )

            # Update document metadata
            if metadata.title:
                document.title = metadata.title
            if metadata.authors:
                document.authors = metadata.authors
            if metadata.year:
                document.year = metadata.year
            if metadata.doi:
                document.doi = metadata.doi
            if metadata.abstract:
                document.abstract = metadata.abstract

            # Chunk document
            chunks = chunker.chunk(
                parsed.full_text,
                metadata={"sections": sections},
            )

            # Generate embeddings
            chunk_texts = [c.content for c in chunks]
            embeddings = await embedder.embed_batch(chunk_texts)

            # Store chunks
            await _store_chunks(session, document_id, chunks, embeddings)

            # Mark as completed
            document.mark_completed(len(chunks))
            await session.commit()

            logger.info(
                "document_ingestion_completed",
                document_id=document_id,
                chunk_count=len(chunks),
            )

            return {
                "document_id": document_id,
                "status": "completed",
                "chunk_count": len(chunks),
            }

        except Exception as e:
            document.mark_failed(str(e))
            await session.commit()
            raise


async def _store_chunks(
    session: AsyncSession,
    document_id: str,
    chunks: list,
    embeddings: list[list[float]],
) -> None:
    """Store chunks with embeddings in database.

    Args:
        session: Database session.
        document_id: Document UUID.
        chunks: List of Chunk objects from chunker.
        embeddings: List of embedding vectors.
    """
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        db_chunk = Chunk(
            document_id=document_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            token_count=chunk.token_count,
            section=chunk.section,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            embedding=embedding,
        )
        session.add(db_chunk)

    await session.flush()


async def _mark_document_failed(document_id: str, error: str) -> None:
    """Mark document as failed in database.

    Args:
        document_id: Document UUID.
        error: Error message.
    """
    async with async_session_maker() as session:
        result = await session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if document:
            document.mark_failed(error)
            await session.commit()
