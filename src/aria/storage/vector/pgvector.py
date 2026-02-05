"""PostgreSQL pgvector implementation."""

from typing import Any

import structlog
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from aria.db.models import Chunk, Document
from aria.db.session import async_session_maker
from aria.storage.vector.base import BaseVectorStore, VectorSearchResult
from aria.types import Embedding

logger = structlog.get_logger(__name__)


class PgVectorStore(BaseVectorStore):
    """PostgreSQL pgvector-based vector store.

    Uses pgvector extension for efficient similarity search with
    IVFFlat indexing.
    """

    def __init__(self, session: AsyncSession | None = None) -> None:
        """Initialize pgvector store.

        Args:
            session: Optional database session. If not provided,
                    creates sessions as needed.
        """
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get or create a database session."""
        if self._session:
            return self._session
        return async_session_maker()

    async def search(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional filters (document_id, section, etc.).
            min_score: Minimum similarity score (0-1).

        Returns:
            List of search results sorted by similarity (descending).
        """
        session = await self._get_session()

        try:
            # Convert to pgvector format
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            # Build base query with cosine similarity
            # pgvector uses <=> for cosine distance (1 - similarity)
            # We compute 1 - distance to get similarity
            query = select(
                Chunk.id,
                Chunk.document_id,
                Chunk.content,
                Chunk.section,
                Chunk.page_number,
                Chunk.metadata_,
                (
                    1
                    - func.cast(Chunk.embedding, text("vector")).op("<=>")(
                        func.cast(embedding_str, text("vector"))
                    )
                ).label("similarity"),
            ).where(Chunk.embedding.isnot(None))

            # Apply filters
            if filters:
                if "document_id" in filters:
                    query = query.where(Chunk.document_id == filters["document_id"])
                if "section" in filters:
                    query = query.where(Chunk.section == filters["section"])
                if "document_ids" in filters:
                    query = query.where(Chunk.document_id.in_(filters["document_ids"]))

            # Order by similarity and limit
            query = query.order_by(text("similarity DESC")).limit(top_k)

            result = await session.execute(query)
            rows = result.fetchall()

            # Convert to results
            results = []
            for row in rows:
                score = float(row.similarity)
                if score >= min_score:
                    results.append(
                        VectorSearchResult(
                            chunk_id=row.id,
                            document_id=row.document_id,
                            content=row.content,
                            score=score,
                            section=row.section,
                            page_number=row.page_number,
                            metadata=row.metadata_,
                        )
                    )

            logger.info(
                "vector_search_completed",
                top_k=top_k,
                results_count=len(results),
            )

            return results

        finally:
            if not self._session:
                await session.close()

    async def search_with_document_info(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search with document metadata included.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional filters.

        Returns:
            List of search results with document title.
        """
        session = await self._get_session()

        try:
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            query = (
                select(
                    Chunk.id,
                    Chunk.document_id,
                    Chunk.content,
                    Chunk.section,
                    Chunk.page_number,
                    Chunk.metadata_,
                    Document.title.label("document_title"),
                    (
                        1
                        - func.cast(Chunk.embedding, text("vector")).op("<=>")(
                            func.cast(embedding_str, text("vector"))
                        )
                    ).label("similarity"),
                )
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.embedding.isnot(None))
                .order_by(text("similarity DESC"))
                .limit(top_k)
            )

            result = await session.execute(query)
            rows = result.fetchall()

            results = []
            for row in rows:
                metadata = row.metadata_ or {}
                metadata["document_title"] = row.document_title

                results.append(
                    VectorSearchResult(
                        chunk_id=row.id,
                        document_id=row.document_id,
                        content=row.content,
                        score=float(row.similarity),
                        section=row.section,
                        page_number=row.page_number,
                        metadata=metadata,
                    )
                )

            return results

        finally:
            if not self._session:
                await session.close()

    async def insert(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        embedding: Embedding,
        metadata: dict | None = None,
    ) -> None:
        """Insert a chunk with embedding.

        Args:
            chunk_id: Unique chunk identifier.
            document_id: Parent document identifier.
            content: Text content.
            embedding: Embedding vector.
            metadata: Optional metadata.
        """
        session = await self._get_session()

        try:
            # Check if chunk exists
            result = await session.execute(select(Chunk).where(Chunk.id == chunk_id))
            chunk = result.scalar_one_or_none()

            if chunk:
                # Update embedding
                chunk.embedding = embedding
                if metadata:
                    chunk.metadata_ = metadata
            else:
                # Create new chunk
                chunk = Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=content,
                    embedding=embedding,
                    chunk_index=0,  # Will be set properly during ingestion
                    token_count=len(content.split()),  # Approximate
                    metadata_=metadata,
                )
                session.add(chunk)

            await session.commit()

            logger.debug("vector_inserted", chunk_id=chunk_id)

        finally:
            if not self._session:
                await session.close()

    async def delete(self, chunk_id: str) -> None:
        """Delete a chunk.

        Args:
            chunk_id: Chunk identifier to delete.
        """
        session = await self._get_session()

        try:
            await session.execute(delete(Chunk).where(Chunk.id == chunk_id))
            await session.commit()

            logger.debug("vector_deleted", chunk_id=chunk_id)

        finally:
            if not self._session:
                await session.close()

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document identifier.

        Returns:
            Number of chunks deleted.
        """
        session = await self._get_session()

        try:
            result = await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            await session.commit()

            count = result.rowcount
            logger.info(
                "vectors_deleted_for_document",
                document_id=document_id,
                count=count,
            )

            return count

        finally:
            if not self._session:
                await session.close()
