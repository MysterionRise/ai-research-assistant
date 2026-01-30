"""Semantic Scholar connector."""

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from aria.config.settings import settings
from aria.connectors.base import BaseConnector, LiteratureResult
from aria.exceptions import ConnectorError, RateLimitError

logger = structlog.get_logger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarConnector(BaseConnector):
    """Semantic Scholar API connector.

    Uses the Semantic Scholar Academic Graph API for comprehensive
    literature search with citation information.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Semantic Scholar connector.

        Args:
            api_key: Semantic Scholar API key (optional but recommended).
        """
        self.api_key = api_key or settings.semantic_scholar_api_key
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
        )

        logger.info(
            "semantic_scholar_connector_initialized",
            has_api_key=bool(self.api_key),
        )

    @property
    def source_name(self) -> str:
        """Return source name."""
        return "semantic_scholar"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[LiteratureResult]:
        """Search Semantic Scholar.

        Args:
            query: Search query.
            limit: Maximum results to return.
            **kwargs: Additional parameters (year, fields_of_study, etc.).

        Returns:
            List of search results.

        Raises:
            ConnectorError: If search fails.
            RateLimitError: If rate limited.
        """
        logger.info("semantic_scholar_search", query=query, limit=limit)

        try:
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,abstract,authors,year,venue,externalIds,url,citationCount,influentialCitationCount",
            }

            # Add year filter if provided
            if kwargs.get("year"):
                params["year"] = str(kwargs["year"])
            elif kwargs.get("year_from") and kwargs.get("year_to"):
                params["year"] = f"{kwargs['year_from']}-{kwargs['year_to']}"

            # Add fields of study filter
            if kwargs.get("fields_of_study"):
                params["fieldsOfStudy"] = kwargs["fields_of_study"]

            response = await self.client.get(
                f"{S2_API_URL}/paper/search",
                params=params,
            )

            if response.status_code == 429:
                raise RateLimitError("semantic_scholar")

            response.raise_for_status()

            data = response.json()
            results = self._parse_results(data.get("data", []))

            logger.info(
                "semantic_scholar_search_completed",
                query=query,
                results_count=len(results),
            )

            return results

        except RateLimitError:
            raise
        except Exception as e:
            raise ConnectorError("semantic_scholar", str(e)) from e

    def _parse_results(self, papers: list[dict]) -> list[LiteratureResult]:
        """Parse Semantic Scholar API results.

        Args:
            papers: List of paper dicts from API.

        Returns:
            List of LiteratureResult objects.
        """
        results = []

        for paper in papers:
            try:
                # Extract authors
                authors = []
                for author in paper.get("authors", []):
                    if author.get("name"):
                        authors.append(author["name"])

                # Extract DOI
                external_ids = paper.get("externalIds", {}) or {}
                doi = external_ids.get("DOI")

                # Calculate a relevance score based on citations
                citation_count = paper.get("citationCount", 0) or 0
                influential_count = paper.get("influentialCitationCount", 0) or 0
                # Simple scoring: log scale of citations
                import math
                score = min(1.0, math.log10(citation_count + 1) / 5) if citation_count > 0 else 0.5
                if influential_count > 0:
                    score = min(1.0, score + 0.2)

                results.append(
                    LiteratureResult(
                        id=paper.get("paperId", ""),
                        title=paper.get("title", "No title"),
                        abstract=paper.get("abstract"),
                        authors=authors,
                        year=paper.get("year"),
                        journal=paper.get("venue"),
                        doi=doi,
                        url=paper.get("url"),
                        source=self.source_name,
                        score=score,
                        metadata={
                            "citation_count": citation_count,
                            "influential_citation_count": influential_count,
                            "arxiv_id": external_ids.get("ArXiv"),
                            "pubmed_id": external_ids.get("PubMed"),
                        },
                    )
                )

            except Exception as e:
                logger.warning("semantic_scholar_parse_error", error=str(e))
                continue

        return results

    async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
        """Get a paper by Semantic Scholar ID or DOI.

        Args:
            paper_id: Semantic Scholar paper ID, DOI, or other identifier.

        Returns:
            LiteratureResult or None if not found.
        """
        try:
            params = {
                "fields": "paperId,title,abstract,authors,year,venue,externalIds,url,citationCount,influentialCitationCount",
            }

            response = await self.client.get(
                f"{S2_API_URL}/paper/{paper_id}",
                params=params,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            data = response.json()
            results = self._parse_results([data])
            return results[0] if results else None

        except Exception as e:
            logger.error(
                "semantic_scholar_fetch_error",
                paper_id=paper_id,
                error=str(e),
            )
            return None

    async def get_references(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> list[LiteratureResult]:
        """Get references for a paper.

        Args:
            paper_id: Semantic Scholar paper ID.
            limit: Maximum references to return.

        Returns:
            List of referenced papers.
        """
        try:
            params = {
                "fields": "paperId,title,abstract,authors,year,venue,externalIds,url",
                "limit": limit,
            }

            response = await self.client.get(
                f"{S2_API_URL}/paper/{paper_id}/references",
                params=params,
            )
            response.raise_for_status()

            data = response.json()
            papers = [
                ref.get("citedPaper", {})
                for ref in data.get("data", [])
                if ref.get("citedPaper")
            ]

            return self._parse_results(papers)

        except Exception as e:
            logger.error(
                "semantic_scholar_references_error",
                paper_id=paper_id,
                error=str(e),
            )
            return []

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
