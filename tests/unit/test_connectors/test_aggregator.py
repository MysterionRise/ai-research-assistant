"""Tests for literature aggregator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


class TestLiteratureAggregatorSearch:
    """Tests for LiteratureAggregator search functionality."""

    @pytest.mark.asyncio
    async def test_search_combines_results_from_sources(self) -> None:
        """Test that search combines results from multiple sources."""
        mock_pubmed = MagicMock()
        mock_pubmed.search = AsyncMock(
            return_value=[
                LiteratureResult(
                    id="pm1",
                    title="PubMed Paper",
                    source="pubmed",
                    score=0.9,
                    doi="10.1/pm1",
                )
            ]
        )

        mock_arxiv = MagicMock()
        mock_arxiv.search = AsyncMock(
            return_value=[
                LiteratureResult(
                    id="ax1",
                    title="ArXiv Paper",
                    source="arxiv",
                    score=0.8,
                    doi="10.1/ax1",
                )
            ]
        )

        with (
            patch("aria.connectors.aggregator.PubMedConnector", return_value=mock_pubmed),
            patch("aria.connectors.aggregator.ArxivConnector", return_value=mock_arxiv),
            patch("aria.connectors.aggregator.SemanticScholarConnector"),
        ):
            aggregator = LiteratureAggregator(sources=["pubmed", "arxiv"])
            results = await aggregator.search("test query")

            assert len(results) == 2
            mock_pubmed.search.assert_called_once()
            mock_arxiv.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_handles_source_failure(self) -> None:
        """Test that search continues when one source fails."""
        mock_pubmed = MagicMock()
        mock_pubmed.search = AsyncMock(side_effect=Exception("PubMed API error"))
        mock_pubmed.source_name = "pubmed"

        mock_arxiv = MagicMock()
        mock_arxiv.search = AsyncMock(
            return_value=[
                LiteratureResult(
                    id="ax1",
                    title="ArXiv Paper",
                    source="arxiv",
                    score=0.8,
                )
            ]
        )

        with (
            patch("aria.connectors.aggregator.PubMedConnector", return_value=mock_pubmed),
            patch("aria.connectors.aggregator.ArxivConnector", return_value=mock_arxiv),
            patch("aria.connectors.aggregator.SemanticScholarConnector"),
        ):
            aggregator = LiteratureAggregator(sources=["pubmed", "arxiv"])
            results = await aggregator.search("test query")

            # Should still get arxiv results
            assert len(results) == 1
            assert results[0].source == "arxiv"

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score(self) -> None:
        """Test that results are sorted by score in descending order."""
        mock_pubmed = MagicMock()
        mock_pubmed.search = AsyncMock(
            return_value=[
                LiteratureResult(
                    id="pm1",
                    title="Low Score Paper",
                    source="pubmed",
                    score=0.5,
                    doi="10.1/low",
                ),
                LiteratureResult(
                    id="pm2",
                    title="High Score Paper",
                    source="pubmed",
                    score=0.95,
                    doi="10.1/high",
                ),
                LiteratureResult(
                    id="pm3",
                    title="Medium Score Paper",
                    source="pubmed",
                    score=0.75,
                    doi="10.1/med",
                ),
            ]
        )

        with (
            patch("aria.connectors.aggregator.PubMedConnector", return_value=mock_pubmed),
            patch("aria.connectors.aggregator.ArxivConnector"),
            patch("aria.connectors.aggregator.SemanticScholarConnector"),
        ):
            aggregator = LiteratureAggregator(sources=["pubmed"])
            results = await aggregator.search("test")

            # Results should be sorted by score descending
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_with_specific_sources(self) -> None:
        """Test searching with specific sources parameter."""
        mock_pubmed = MagicMock()
        mock_pubmed.search = AsyncMock(return_value=[])

        mock_arxiv = MagicMock()
        mock_arxiv.search = AsyncMock(return_value=[])

        with (
            patch("aria.connectors.aggregator.PubMedConnector", return_value=mock_pubmed),
            patch("aria.connectors.aggregator.ArxivConnector", return_value=mock_arxiv),
            patch("aria.connectors.aggregator.SemanticScholarConnector"),
        ):
            aggregator = LiteratureAggregator(sources=["pubmed", "arxiv"])
            # Only query pubmed
            await aggregator.search("test", sources=["pubmed"])

            mock_pubmed.search.assert_called_once()
            mock_arxiv.search.assert_not_called()


class TestLiteratureAggregatorClose:
    """Tests for closing the aggregator."""

    @pytest.mark.asyncio
    async def test_close_calls_connector_close(self) -> None:
        """Test that close calls close on connectors that have it."""
        mock_pubmed = MagicMock()
        mock_pubmed.close = AsyncMock()

        with (
            patch("aria.connectors.aggregator.PubMedConnector", return_value=mock_pubmed),
            patch("aria.connectors.aggregator.ArxivConnector"),
            patch("aria.connectors.aggregator.SemanticScholarConnector"),
        ):
            aggregator = LiteratureAggregator(sources=["pubmed"])
            await aggregator.close()

            mock_pubmed.close.assert_called_once()


class TestLiteratureAggregatorMerge:
    """Additional tests for result merging."""

    def test_merge_single_result_returns_same(self) -> None:
        """Test that merging single result returns it unchanged."""
        aggregator = LiteratureAggregator(sources=[])

        result = LiteratureResult(
            id="1",
            title="Single Paper",
            source="pubmed",
            score=0.8,
        )

        merged = aggregator._merge_results([result])

        assert merged.id == "1"
        assert merged.title == "Single Paper"

    def test_merge_combines_metadata(self) -> None:
        """Test that merge combines metadata from both results."""
        aggregator = LiteratureAggregator(sources=[])

        results = [
            LiteratureResult(
                id="1",
                title="Paper",
                source="pubmed",
                score=0.5,
                metadata={"citations": 10},
            ),
            LiteratureResult(
                id="2",
                title="Paper",
                source="arxiv",
                score=0.5,
                metadata={"downloads": 100},
            ),
        ]

        merged = aggregator._merge_results(results)

        assert "citations" in merged.metadata or "downloads" in merged.metadata
        assert "sources" in merged.metadata
