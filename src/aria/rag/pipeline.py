"""Main RAG pipeline orchestration."""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from aria.config.settings import settings
from aria.rag.reranking.cross_encoder import CrossEncoderReranker
from aria.rag.retrieval.base import RetrievalResult
from aria.rag.retrieval.hybrid import HybridRetriever
from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer
from aria.types import Citation, RAGResponse

logger = structlog.get_logger(__name__)


@dataclass
class RAGPipelineResult:
    """Complete result from RAG pipeline."""

    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]
    reranked_chunks: list[RetrievalResult]
    confidence: float
    latency_ms: int
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """Main RAG pipeline for scientific literature QA.

    Orchestrates:
    1. Hybrid retrieval (semantic + keyword)
    2. Cross-encoder reranking
    3. Citation-aware synthesis
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        synthesizer: CitationAwareSynthesizer | None = None,
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> None:
        """Initialize RAG pipeline.

        Args:
            retriever: Document retriever (default: HybridRetriever).
            reranker: Reranker (default: CrossEncoderReranker).
            synthesizer: Answer synthesizer (default: CitationAwareSynthesizer).
            retrieval_top_k: Number of documents to retrieve.
            rerank_top_k: Number of documents after reranking.
        """
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.synthesizer = synthesizer or CitationAwareSynthesizer()

        self.retrieval_top_k = retrieval_top_k or settings.rag_retrieval_top_k
        self.rerank_top_k = rerank_top_k or settings.rag_rerank_top_k

        logger.info(
            "rag_pipeline_initialized",
            retrieval_top_k=self.retrieval_top_k,
            rerank_top_k=self.rerank_top_k,
        )

    async def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        max_tokens: int = 2048,
    ) -> RAGPipelineResult:
        """Execute the full RAG pipeline.

        Args:
            question: User's question.
            filters: Optional filters for retrieval.
            max_tokens: Maximum response tokens.

        Returns:
            RAGPipelineResult with answer, citations, and metadata.
        """
        start_time = time.time()

        logger.info("rag_pipeline_query", question=question[:100])

        # Step 1: Retrieve
        retrieved = await self.retriever.retrieve(
            query=question,
            top_k=self.retrieval_top_k,
            filters=filters,
        )

        logger.info("retrieval_completed", count=len(retrieved))

        # Step 2: Rerank
        if retrieved:
            reranked = await self.reranker.rerank(
                query=question,
                results=retrieved,
                top_k=self.rerank_top_k,
            )
        else:
            reranked = []

        logger.info("reranking_completed", count=len(reranked))

        # Step 3: Synthesize
        synthesis_result = await self.synthesizer.synthesize(
            query=question,
            context=reranked,
            max_tokens=max_tokens,
        )

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        result = RAGPipelineResult(
            answer=synthesis_result.answer,
            citations=synthesis_result.citations,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            confidence=synthesis_result.confidence,
            latency_ms=latency_ms,
            metadata={
                "model": synthesis_result.metadata.get("model"),
                "tokens_used": synthesis_result.tokens_used,
                "retrieval_count": len(retrieved),
                "rerank_count": len(reranked),
            },
        )

        logger.info(
            "rag_pipeline_completed",
            latency_ms=latency_ms,
            confidence=result.confidence,
            citations_count=len(result.citations),
        )

        return result

    def to_rag_response(self, result: RAGPipelineResult, query: str) -> RAGResponse:
        """Convert pipeline result to RAGResponse type.

        Args:
            result: Pipeline result.
            query: Original query.

        Returns:
            RAGResponse for API response.
        """
        return RAGResponse(
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            query=query,
            sources_used=len(result.citations),
            metadata={
                "latency_ms": result.latency_ms,
                **result.metadata,
            },
        )
