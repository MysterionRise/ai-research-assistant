"""Cross-encoder reranking for improved relevance."""

import structlog

from aria.config.settings import settings
from aria.rag.retrieval.base import RetrievalResult

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for improved relevance scoring.

    Uses a cross-encoder model to rerank retrieval results by
    computing query-document relevance scores.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int | None = None,
    ) -> None:
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder.
            top_k: Default number of results to return after reranking.
        """
        self.model_name = model_name
        self.top_k = top_k or settings.rag_rerank_top_k
        self._model = None

        logger.info(
            "cross_encoder_reranker_initialized",
            model=model_name,
            top_k=self.top_k,
        )

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info("cross_encoder_model_loaded", model=self.model_name)
            except ImportError:
                logger.warning(
                    "sentence_transformers_not_available",
                    message="Falling back to score-based reranking",
                )
                self._model = "fallback"

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using cross-encoder.

        Args:
            query: Original search query.
            results: Retrieval results to rerank.
            top_k: Number of results to return (default: self.top_k).

        Returns:
            Reranked results sorted by relevance.
        """
        if not results:
            return []

        top_k = top_k or self.top_k

        logger.info(
            "reranking_results",
            query=query[:100],
            input_count=len(results),
            top_k=top_k,
        )

        self._load_model()

        if self._model == "fallback":
            # Fallback: just use existing scores
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            return sorted_results[:top_k]

        # Prepare query-document pairs
        pairs = [(query, r.content) for r in results]

        # Get cross-encoder scores
        scores = self._model.predict(pairs)

        # Combine with results
        scored_results = list(zip(scores, results, strict=True))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Update scores and return top_k
        reranked = []
        max_score = max(scores) if scores.any() else 1.0  # type: ignore[union-attr]
        min_score = min(scores) if scores.any() else 0.0  # type: ignore[union-attr]
        score_range = max_score - min_score if max_score != min_score else 1.0

        for score, result in scored_results[:top_k]:
            # Normalize score to 0-1
            result.score = (score - min_score) / score_range
            result.metadata["rerank_score"] = float(score)
            reranked.append(result)

        logger.info(
            "reranking_completed",
            output_count=len(reranked),
        )

        return reranked
