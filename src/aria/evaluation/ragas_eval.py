"""Ragas-based RAG evaluation."""

import json
import statistics
import time
from pathlib import Path

import structlog

from aria.evaluation.metrics import RAGMetrics
from aria.rag.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)


class RAGASEvaluator:
    """RAG evaluation using Ragas framework.

    Evaluates:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Is the retrieved context precise?
    - Context Recall: Is the retrieved context complete?
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            rag_pipeline: RAG pipeline to evaluate.
        """
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self._ragas_available = self._check_ragas()

        logger.info(
            "ragas_evaluator_initialized",
            ragas_available=self._ragas_available,
        )

    def _check_ragas(self) -> bool:
        """Check if Ragas is available."""
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "ragas_not_available",
                message="Install ragas for full evaluation: pip install ragas",
            )
            return False

    async def evaluate_golden_set(
        self,
        golden_set_path: Path | str,
    ) -> RAGMetrics:
        """Evaluate RAG pipeline against a golden set.

        Args:
            golden_set_path: Path to golden set JSON file.

        Returns:
            RAGMetrics with evaluation results.
        """
        logger.info("evaluating_golden_set", path=str(golden_set_path))

        # Load golden set
        with open(golden_set_path) as f:
            golden_set = json.load(f)

        test_cases = golden_set.get("test_cases", [])
        if not test_cases:
            logger.warning("empty_golden_set")
            return RAGMetrics()

        # Run evaluation
        results = await self._run_evaluation(test_cases)

        logger.info(
            "evaluation_completed",
            queries=results.queries_evaluated,
            faithfulness=results.faithfulness,
            relevancy=results.answer_relevancy,
        )

        return results

    async def _run_evaluation(
        self,
        test_cases: list[dict],
    ) -> RAGMetrics:
        """Run evaluation on test cases.

        Args:
            test_cases: List of test case dicts with query, expected_answer.

        Returns:
            Aggregated metrics.
        """
        latencies = []
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []

        successful = 0
        failed = 0

        for case in test_cases:
            query = case.get("query", "")
            expected_answer = case.get("expected_answer")
            _expected_sources = case.get("expected_sources", [])  # Reserved for future use

            try:
                # Run RAG pipeline
                start_time = time.time()
                result = await self.rag_pipeline.query(query)
                latency_ms = int((time.time() - start_time) * 1000)
                latencies.append(latency_ms)

                # Calculate metrics
                if self._ragas_available:
                    scores = await self._calculate_ragas_metrics(
                        query=query,
                        answer=result.answer,
                        contexts=[c.content for c in result.reranked_chunks],
                        ground_truth=expected_answer,
                    )
                else:
                    # Fallback to simple heuristics
                    scores = self._calculate_simple_metrics(
                        query=query,
                        answer=result.answer,
                        contexts=[c.content for c in result.reranked_chunks],
                        ground_truth=expected_answer,
                    )

                faithfulness_scores.append(scores.get("faithfulness", 0))
                relevancy_scores.append(scores.get("answer_relevancy", 0))
                precision_scores.append(scores.get("context_precision", 0))
                recall_scores.append(scores.get("context_recall", 0))

                successful += 1

            except Exception as e:
                logger.error("evaluation_case_failed", query=query, error=str(e))
                failed += 1

        # Aggregate metrics
        metrics = RAGMetrics(
            faithfulness=statistics.mean(faithfulness_scores) if faithfulness_scores else 0,
            answer_relevancy=statistics.mean(relevancy_scores) if relevancy_scores else 0,
            context_precision=statistics.mean(precision_scores) if precision_scores else 0,
            context_recall=statistics.mean(recall_scores) if recall_scores else 0,
            latency_p50_ms=int(statistics.median(latencies)) if latencies else 0,
            latency_p95_ms=int(statistics.quantiles(latencies, n=20)[18]) if len(latencies) >= 20 else max(latencies) if latencies else 0,
            latency_p99_ms=int(statistics.quantiles(latencies, n=100)[98]) if len(latencies) >= 100 else max(latencies) if latencies else 0,
            queries_evaluated=len(test_cases),
            successful_queries=successful,
            failed_queries=failed,
        )

        return metrics

    async def _calculate_ragas_metrics(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> dict:
        """Calculate metrics using Ragas.

        Args:
            query: User question.
            answer: Generated answer.
            contexts: Retrieved contexts.
            ground_truth: Expected answer (if available).

        Returns:
            Dict of metric scores.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }

            if ground_truth:
                data["ground_truth"] = [ground_truth]
                metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            else:
                metrics = [faithfulness, answer_relevancy, context_precision]

            dataset = Dataset.from_dict(data)
            result = evaluate(dataset, metrics=metrics)

            return {
                "faithfulness": result.get("faithfulness", 0),
                "answer_relevancy": result.get("answer_relevancy", 0),
                "context_precision": result.get("context_precision", 0),
                "context_recall": result.get("context_recall", 0) if ground_truth else 0,
            }

        except Exception as e:
            logger.warning("ragas_calculation_failed", error=str(e))
            return self._calculate_simple_metrics(query, answer, contexts, ground_truth)

    def _calculate_simple_metrics(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> dict:
        """Calculate simple heuristic metrics.

        Fallback when Ragas is not available.

        Args:
            query: User question.
            answer: Generated answer.
            contexts: Retrieved contexts.
            ground_truth: Expected answer.

        Returns:
            Dict of estimated metric scores.
        """
        # Simple faithfulness: check if answer terms appear in context
        answer_terms = set(answer.lower().split())
        context_text = " ".join(contexts).lower()
        context_terms = set(context_text.split())

        overlap = len(answer_terms & context_terms)
        faithfulness = overlap / len(answer_terms) if answer_terms else 0

        # Simple relevancy: check if query terms appear in answer
        query_terms = set(query.lower().split())
        query_overlap = len(query_terms & answer_terms)
        relevancy = query_overlap / len(query_terms) if query_terms else 0

        # Simple precision: answer length vs context length ratio
        precision = min(1.0, len(answer) / (len(context_text) + 1) * 10)

        # Simple recall: if ground truth available
        recall = 0.0
        if ground_truth:
            gt_terms = set(ground_truth.lower().split())
            gt_overlap = len(gt_terms & answer_terms)
            recall = gt_overlap / len(gt_terms) if gt_terms else 0

        return {
            "faithfulness": min(1.0, faithfulness),
            "answer_relevancy": min(1.0, relevancy),
            "context_precision": min(1.0, precision),
            "context_recall": min(1.0, recall),
        }


async def main() -> None:
    """Run evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument(
        "--golden-set",
        type=str,
        required=True,
        help="Path to golden set JSON file",
    )
    args = parser.parse_args()

    evaluator = RAGASEvaluator()
    metrics = await evaluator.evaluate_golden_set(args.golden_set)

    print("\n=== RAG Evaluation Results ===")
    print(f"Faithfulness:      {metrics.faithfulness:.3f}")
    print(f"Answer Relevancy:  {metrics.answer_relevancy:.3f}")
    print(f"Context Precision: {metrics.context_precision:.3f}")
    print(f"Context Recall:    {metrics.context_recall:.3f}")
    print(f"Latency P95:       {metrics.latency_p95_ms}ms")
    print(f"Overall Score:     {metrics.overall_score:.3f}")
    print(f"Queries:           {metrics.queries_evaluated}")
    print(f"Success Rate:      {metrics.success_rate:.1%}")

    # Check thresholds
    if metrics.passes_threshold():
        print("\n✓ All thresholds passed!")
    else:
        print("\n✗ Some thresholds not met")
        if metrics.faithfulness < 0.75:
            print(f"  - Faithfulness {metrics.faithfulness:.3f} < 0.75")
        if metrics.answer_relevancy < 0.75:
            print(f"  - Answer Relevancy {metrics.answer_relevancy:.3f} < 0.75")
        if metrics.latency_p95_ms > 10000:
            print(f"  - Latency P95 {metrics.latency_p95_ms}ms > 10000ms")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
