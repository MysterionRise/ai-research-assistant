"""Tests for literature aggregator."""

from aria.connectors.aggregator import LiteratureAggregator
from aria.connectors.base import LiteratureResult


class TestLiteratureAggregator:
    """Tests for LiteratureAggregator."""

    def test_deduplicate_by_doi(self):
        """Test deduplication by DOI."""
        aggregator = LiteratureAggregator(sources=[])

        results = [
            LiteratureResult(
                id="1",
                title="Paper One",
                doi="10.1234/test.123",
                source="pubmed",
                score=0.8,
            ),
            LiteratureResult(
                id="2",
                title="Paper One (Duplicate)",
                doi="10.1234/test.123",  # Same DOI
                source="semantic_scholar",
                score=0.7,
            ),
            LiteratureResult(
                id="3",
                title="Different Paper",
                doi="10.1234/test.456",
                source="pubmed",
                score=0.6,
            ),
        ]

        deduped = aggregator._deduplicate_results(results)

        # Should have 2 unique papers
        assert len(deduped) == 2

        # DOIs should be unique
        dois = [r.doi for r in deduped if r.doi]
        assert len(dois) == len(set(dois))

    def test_merge_results_takes_best_fields(self):
        """Test merging takes best fields from duplicates."""
        aggregator = LiteratureAggregator(sources=[])

        results = [
            LiteratureResult(
                id="1",
                title="Paper",
                abstract="Short abstract",
                authors=["Author A"],
                doi="10.1234/test",
                source="pubmed",
                score=0.7,
            ),
            LiteratureResult(
                id="2",
                title="Paper",
                abstract="This is a much longer and more detailed abstract with more information",
                authors=["Author A", "Author B", "Author C"],
                doi="10.1234/test",
                source="semantic_scholar",
                score=0.9,
            ),
        ]

        merged = aggregator._merge_results(results)

        # Should take longer abstract
        assert "longer and more detailed" in (merged.abstract or "")
        # Should take more complete author list
        assert len(merged.authors) == 3
        # Should take higher score
        assert merged.score == 0.9

    def test_deduplicate_by_title_when_no_doi(self):
        """Test deduplication by title when DOI not available."""
        aggregator = LiteratureAggregator(sources=[])

        results = [
            LiteratureResult(
                id="1",
                title="Unique Paper Title",
                doi=None,
                source="arxiv",
                score=0.8,
            ),
            LiteratureResult(
                id="2",
                title="Unique Paper Title",  # Same title
                doi=None,
                source="pubmed",
                score=0.7,
            ),
            LiteratureResult(
                id="3",
                title="Different Title",
                doi=None,
                source="arxiv",
                score=0.6,
            ),
        ]

        deduped = aggregator._deduplicate_results(results)

        # Should have 2 unique by title
        assert len(deduped) == 2
