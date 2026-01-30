"""RAG evaluation metrics."""

from dataclasses import dataclass, field


@dataclass
class RAGMetrics:
    """Aggregated RAG evaluation metrics."""

    # Core Ragas metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Performance metrics
    latency_p50_ms: int = 0
    latency_p95_ms: int = 0
    latency_p99_ms: int = 0

    # Quality metrics
    citation_accuracy: float = 0.0
    answer_completeness: float = 0.0

    # Coverage metrics
    queries_evaluated: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    # Additional details
    details: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate query success rate."""
        if self.queries_evaluated == 0:
            return 0.0
        return self.successful_queries / self.queries_evaluated

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score (average of core metrics)."""
        metrics = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        non_zero = [m for m in metrics if m > 0]
        return sum(non_zero) / len(non_zero) if non_zero else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "citation_accuracy": self.citation_accuracy,
            "answer_completeness": self.answer_completeness,
            "queries_evaluated": self.queries_evaluated,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.success_rate,
            "overall_score": self.overall_score,
        }

    def passes_threshold(
        self,
        faithfulness: float = 0.75,
        answer_relevancy: float = 0.75,
        latency_p95_ms: int = 10000,
    ) -> bool:
        """Check if metrics pass the acceptance threshold.

        Args:
            faithfulness: Minimum faithfulness score.
            answer_relevancy: Minimum answer relevancy score.
            latency_p95_ms: Maximum P95 latency in milliseconds.

        Returns:
            True if all thresholds are met.
        """
        return (
            self.faithfulness >= faithfulness
            and self.answer_relevancy >= answer_relevancy
            and self.latency_p95_ms <= latency_p95_ms
        )
