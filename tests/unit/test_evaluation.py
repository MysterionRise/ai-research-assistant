"""Tests for evaluation framework."""

import json
import tempfile
from pathlib import Path

from aria.evaluation.golden_set import EvalCase, GoldenSet, create_sample_golden_set
from aria.evaluation.metrics import RAGMetrics


class TestRAGMetrics:
    """Tests for RAGMetrics."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = RAGMetrics(
            queries_evaluated=10,
            successful_queries=8,
            failed_queries=2,
        )

        assert metrics.success_rate == 0.8

    def test_success_rate_zero_queries(self):
        """Test success rate with zero queries."""
        metrics = RAGMetrics(queries_evaluated=0)
        assert metrics.success_rate == 0.0

    def test_overall_score_calculation(self):
        """Test overall score calculation."""
        metrics = RAGMetrics(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.6,
        )

        score = metrics.overall_score
        expected = (0.8 + 0.9 + 0.7 + 0.6) / 4

        assert abs(score - expected) < 0.001

    def test_passes_threshold_all_pass(self):
        """Test threshold checking when all pass."""
        metrics = RAGMetrics(
            faithfulness=0.85,
            answer_relevancy=0.9,
            latency_p95_ms=5000,
        )

        assert metrics.passes_threshold() is True

    def test_passes_threshold_faithfulness_fail(self):
        """Test threshold checking when faithfulness fails."""
        metrics = RAGMetrics(
            faithfulness=0.5,  # Below 0.75 threshold
            answer_relevancy=0.9,
            latency_p95_ms=5000,
        )

        assert metrics.passes_threshold() is False

    def test_passes_threshold_latency_fail(self):
        """Test threshold checking when latency fails."""
        metrics = RAGMetrics(
            faithfulness=0.85,
            answer_relevancy=0.9,
            latency_p95_ms=15000,  # Above 10000ms threshold
        )

        assert metrics.passes_threshold() is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RAGMetrics(
            faithfulness=0.8,
            answer_relevancy=0.9,
            latency_p95_ms=5000,
            queries_evaluated=10,
        )

        d = metrics.to_dict()

        assert d["faithfulness"] == 0.8
        assert d["answer_relevancy"] == 0.9
        assert d["latency_p95_ms"] == 5000
        assert d["queries_evaluated"] == 10
        assert "overall_score" in d
        assert "success_rate" in d


class TestGoldenSet:
    """Tests for GoldenSet."""

    def test_create_sample_golden_set(self):
        """Test sample golden set creation."""
        gs = create_sample_golden_set()

        assert gs.name
        assert len(gs.test_cases) > 0
        assert all(tc.query for tc in gs.test_cases)

    def test_filter_by_category(self):
        """Test filtering by category."""
        gs = create_sample_golden_set()
        materials = gs.filter_by_category("materials_science")

        assert all(tc.category == "materials_science" for tc in materials.test_cases)
        assert len(materials.test_cases) > 0

    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        gs = create_sample_golden_set()
        easy = gs.filter_by_difficulty("easy")

        assert all(tc.difficulty == "easy" for tc in easy.test_cases)

    def test_eval_case_fields(self):
        """Test EvalCase has required fields."""
        tc = EvalCase(
            id="test_1",
            query="What is X?",
            expected_answer="X is...",
            category="test",
        )

        assert tc.id == "test_1"
        assert tc.query == "What is X?"
        assert tc.expected_answer == "X is..."
        assert tc.category == "test"
        assert tc.difficulty == "medium"  # Default

    def test_to_json_and_from_json(self):
        """Test saving and loading golden set to/from JSON."""
        # Create a golden set
        gs = GoldenSet(
            name="Test Golden Set",
            description="Test description",
            test_cases=[
                EvalCase(
                    id="tc1",
                    query="Test query?",
                    expected_answer="Test answer",
                    category="test",
                    difficulty="easy",
                ),
            ],
            version="1.0",
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            gs.to_json(temp_path)

            # Load back
            loaded = GoldenSet.from_json(temp_path)

            assert loaded.name == "Test Golden Set"
            assert loaded.description == "Test description"
            assert len(loaded.test_cases) == 1
            assert loaded.test_cases[0].id == "tc1"
            assert loaded.test_cases[0].query == "Test query?"
        finally:
            temp_path.unlink()

    def test_from_json_minimal(self):
        """Test loading golden set with minimal data."""
        data = {
            "test_cases": [
                {"query": "Simple query?"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            loaded = GoldenSet.from_json(temp_path)

            assert loaded.name == "Golden Set"  # Default
            assert len(loaded.test_cases) == 1
            assert loaded.test_cases[0].query == "Simple query?"
            assert loaded.test_cases[0].category == "general"  # Default
        finally:
            temp_path.unlink()

    def test_eval_case_with_all_fields(self):
        """Test EvalCase with all fields."""
        tc = EvalCase(
            id="full_1",
            query="What is Y?",
            expected_answer="Y is...",
            expected_sources=["source1", "source2"],
            category="science",
            difficulty="hard",
            metadata={"key": "value"},
        )

        assert tc.expected_sources == ["source1", "source2"]
        assert tc.metadata == {"key": "value"}
