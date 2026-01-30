"""FastAPI dependency injection for ARIA."""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from aria.connectors.aggregator import LiteratureAggregator
from aria.db.session import get_async_session
from aria.llm.chains.literature_qa import LiteratureQAChain
from aria.rag.embedding.openai import OpenAIEmbedder
from aria.rag.pipeline import RAGPipeline
from aria.storage.vector.pgvector import PgVectorStore


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async for session in get_async_session():
        yield session


# Type alias for dependency injection
DBSession = Annotated[AsyncSession, Depends(get_db_session)]


@lru_cache(maxsize=1)
def get_embedder() -> OpenAIEmbedder:
    """Get embedder singleton."""
    return OpenAIEmbedder()


@lru_cache(maxsize=1)
def get_vector_store() -> PgVectorStore:
    """Get vector store singleton."""
    return PgVectorStore()


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline singleton."""
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_literature_aggregator() -> LiteratureAggregator:
    """Get literature aggregator singleton."""
    return LiteratureAggregator()


@lru_cache(maxsize=1)
def get_literature_qa_chain() -> LiteratureQAChain:
    """Get literature QA chain singleton."""
    return LiteratureQAChain(
        rag_pipeline=get_rag_pipeline(),
        literature_aggregator=get_literature_aggregator(),
    )


# Dependency types
RAGPipelineDep = Annotated[RAGPipeline, Depends(get_rag_pipeline)]
EmbedderDep = Annotated[OpenAIEmbedder, Depends(get_embedder)]
VectorStoreDep = Annotated[PgVectorStore, Depends(get_vector_store)]
LiteratureAggregatorDep = Annotated[LiteratureAggregator, Depends(get_literature_aggregator)]
LiteratureQAChainDep = Annotated[LiteratureQAChain, Depends(get_literature_qa_chain)]
