"""Literature aggregator for combining multiple sources."""

import asyncio
from collections import defaultdict
from typing import Any

import structlog

from aria.connectors.arxiv import ArxivConnector
from aria.connectors.base import BaseConnector, LiteratureResult
from aria.connectors.pubmed import PubMedConnector
from aria.connectors.semantic_scholar import SemanticScholarConnector

logger = structlog.get_logger(__name__)


class LiteratureAggregator:
    """Aggregates search results from multiple literature sources.

    Features:
    - Parallel querying of multiple sources
    - Deduplication by DOI
    - Result merging and reranking
    """

    def __init__(
        self,
        sources: list[str] | None = None,
    ) -> None:
        """Initialize aggregator with specified sources.

        Args:
            sources: List of source names to use. Defaults to all.
        """
        self.available_sources: dict[str, BaseConnector] = {}

        # Initialize requested connectors
        all_sources = sources or ["pubmed", "arxiv", "semantic_scholar"]

        if "pubmed" in all_sources:
            self.available_sources["pubmed"] = PubMedConnector()

        if "arxiv" in all_sources:
            self.available_sources["arxiv"] = ArxivConnector()

        if "semantic_scholar" in all_sources:
            self.available_sources["semantic_scholar"] = SemanticScholarConnector()

        logger.info(
            "literature_aggregator_initialized",
            sources=list(self.available_sources.keys()),
        )

    async def search(
        self,
        query: str,
        limit: int = 20,
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> list[LiteratureResult]:
        """Search across multiple sources and aggregate results.

        Args:
            query: Search query.
            limit: Maximum results per source.
            sources: Specific sources to query (defaults to all).
            **kwargs: Additional filters (year_from, year_to, etc.).

        Returns:
            Aggregated and deduplicated results.
        """
        logger.info(
            "aggregated_search",
            query=query,
            limit=limit,
            sources=sources or list(self.available_sources.keys()),
        )

        # Determine which sources to query
        active_sources = sources or list(self.available_sources.keys())

        # Query sources in parallel
        tasks = []
        for source_name in active_sources:
            if source_name in self.available_sources:
                connector = self.available_sources[source_name]
                tasks.append(self._search_source(connector, query, limit, **kwargs))

        # Gather results with error handling
        source_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results: list[LiteratureResult] = []
        for result in source_results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.warning("source_search_failed", error=str(result))

        # Deduplicate and merge
        deduplicated = self._deduplicate_results(all_results)

        # Sort by score (descending)
        deduplicated.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "aggregated_search_completed",
            query=query,
            total_raw=len(all_results),
            total_deduplicated=len(deduplicated),
        )

        return deduplicated

    async def _search_source(
        self,
        connector: BaseConnector,
        query: str,
        limit: int,
        **kwargs: Any,
    ) -> list[LiteratureResult]:
        """Search a single source.

        Args:
            connector: Source connector.
            query: Search query.
            limit: Max results.
            **kwargs: Additional filters.

        Returns:
            List of results from source.
        """
        try:
            return await connector.search(query, limit, **kwargs)
        except Exception as e:
            logger.warning(
                "source_search_error",
                source=connector.source_name,
                error=str(e),
            )
            return []

    def _deduplicate_results(
        self,
        results: list[LiteratureResult],
    ) -> list[LiteratureResult]:
        """Deduplicate results by DOI or title similarity.

        When duplicates are found, merge metadata from multiple sources.

        Args:
            results: List of results to deduplicate.

        Returns:
            Deduplicated results.
        """
        # Group by DOI
        by_doi: dict[str, list[LiteratureResult]] = defaultdict(list)
        no_doi: list[LiteratureResult] = []

        for result in results:
            if result.doi:
                by_doi[result.doi.lower()].append(result)
            else:
                no_doi.append(result)

        deduplicated: list[LiteratureResult] = []

        # Merge duplicates by DOI
        for _doi, group in by_doi.items():
            merged = self._merge_results(group)
            deduplicated.append(merged)

        # For results without DOI, try to match by title
        title_seen: set[str] = set()
        for result in no_doi:
            title_key = result.title.lower().strip()[:100]
            if title_key not in title_seen:
                title_seen.add(title_key)
                deduplicated.append(result)

        return deduplicated

    def _merge_results(
        self,
        results: list[LiteratureResult],
    ) -> LiteratureResult:
        """Merge multiple results for the same paper.

        Takes the best information from each source.

        Args:
            results: List of results for the same paper.

        Returns:
            Merged result.
        """
        if len(results) == 1:
            return results[0]

        # Start with the first result
        merged = results[0]

        # Track sources
        sources = [r.source for r in results]

        for result in results[1:]:
            # Prefer longer abstract
            if result.abstract and (
                not merged.abstract or len(result.abstract) > len(merged.abstract)
            ):
                merged.abstract = result.abstract

            # Prefer more complete author list
            if len(result.authors) > len(merged.authors):
                merged.authors = result.authors

            # Take the highest score
            merged.score = max(merged.score, result.score)

            # Merge metadata
            if result.metadata:
                merged.metadata.update(result.metadata)

        # Record that this came from multiple sources
        merged.metadata["sources"] = sources

        return merged

    async def close(self) -> None:
        """Close all connectors."""
        for connector in self.available_sources.values():
            if hasattr(connector, "close"):
                await connector.close()
