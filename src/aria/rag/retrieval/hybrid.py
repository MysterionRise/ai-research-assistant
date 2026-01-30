"""Hybrid retrieval combining semantic and keyword search."""

from typing import Any

import structlog

from aria.rag.retrieval.base import BaseRetriever, RetrievalResult
from aria.rag.retrieval.keyword import KeywordRetriever
from aria.rag.retrieval.semantic import SemanticRetriever

logger = structlog.get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    both retrieval methods.
    """

    # RRF constant (controls impact of ranking position)
    RRF_K = 60

    def __init__(
        self,
        semantic_retriever: SemanticRetriever | None = None,
        keyword_retriever: KeywordRetriever | None = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            semantic_retriever: Semantic retriever instance.
            keyword_retriever: Keyword retriever instance.
            semantic_weight: Weight for semantic results (0-1).
            keyword_weight: Weight for keyword results (0-1).
        """
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        self.keyword_retriever = keyword_retriever or KeywordRetriever()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        logger.info(
            "hybrid_retriever_initialized",
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using hybrid search.

        Combines semantic and keyword retrieval using RRF.

        Args:
            query: Search query.
            top_k: Number of results to return.
            filters: Optional filters.

        Returns:
            List of retrieval results sorted by combined score.
        """
        logger.info("hybrid_retrieval", query=query[:100], top_k=top_k)

        # Retrieve more candidates for fusion
        candidate_k = top_k * 3

        # Run both retrievers
        import asyncio
        semantic_task = self.semantic_retriever.retrieve(query, candidate_k, filters)
        keyword_task = self.keyword_retriever.retrieve(query, candidate_k, filters)

        semantic_results, keyword_results = await asyncio.gather(
            semantic_task,
            keyword_task,
        )

        # Apply RRF fusion
        fused_results = self._rrf_fusion(
            semantic_results,
            keyword_results,
        )

        # Return top_k
        results = fused_results[:top_k]

        logger.info(
            "hybrid_retrieval_completed",
            semantic_count=len(semantic_results),
            keyword_count=len(keyword_results),
            fused_count=len(results),
        )

        return results

    def _rrf_fusion(
        self,
        semantic_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list

        Args:
            semantic_results: Results from semantic retrieval.
            keyword_results: Results from keyword retrieval.

        Returns:
            Fused and sorted results.
        """
        # Calculate RRF scores
        scores: dict[str, float] = {}
        results_map: dict[str, RetrievalResult] = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = result.chunk_id
            rrf_score = self.semantic_weight / (self.RRF_K + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            results_map[chunk_id] = result

        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            chunk_id = result.chunk_id
            rrf_score = self.keyword_weight / (self.RRF_K + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in results_map:
                results_map[chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build result list with normalized scores
        max_score = max(scores.values()) if scores else 1.0
        fused_results = []
        for chunk_id in sorted_ids:
            result = results_map[chunk_id]
            # Normalize score to 0-1
            result.score = scores[chunk_id] / max_score
            fused_results.append(result)

        return fused_results
