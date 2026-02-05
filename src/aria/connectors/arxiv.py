"""arXiv connector using the arXiv API."""

import re
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from aria.connectors.base import BaseConnector, LiteratureResult
from aria.exceptions import ConnectorError

logger = structlog.get_logger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"


class ArxivConnector(BaseConnector):
    """arXiv literature connector.

    Uses the arXiv API for searching preprints in physics,
    mathematics, computer science, and related fields.
    """

    # Namespace for arXiv Atom feed
    ATOM_NS: ClassVar[dict[str, str]] = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(self) -> None:
        """Initialize arXiv connector."""
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("arxiv_connector_initialized")

    @property
    def source_name(self) -> str:
        """Return source name."""
        return "arxiv"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[LiteratureResult]:
        """Search arXiv for papers.

        Args:
            query: Search query.
            limit: Maximum results to return.
            **kwargs: Additional parameters (category, etc.).

        Returns:
            List of arXiv results.

        Raises:
            ConnectorError: If search fails.
        """
        logger.info("arxiv_search", query=query, limit=limit)

        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            # Add category filter if provided
            if kwargs.get("category"):
                params["search_query"] += f" AND cat:{kwargs['category']}"

            response = await self.client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()

            results = self._parse_arxiv_response(response.text)

            logger.info(
                "arxiv_search_completed",
                query=query,
                results_count=len(results),
            )

            return results

        except Exception as e:
            raise ConnectorError("arxiv", str(e)) from e

    def _parse_arxiv_response(self, xml_text: str) -> list[LiteratureResult]:
        """Parse arXiv Atom feed response.

        Args:
            xml_text: XML response from arXiv.

        Returns:
            List of LiteratureResult objects.
        """
        results = []

        # Handle namespace
        # arXiv returns Atom feed format
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error("arxiv_xml_parse_error", error=str(e))
            return []

        # Find all entries
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            try:
                # Extract arXiv ID from the id URL
                id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                arxiv_url = id_elem.text if id_elem is not None else ""
                arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""

                # Title
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                title = title_elem.text if title_elem is not None else "No title"
                title = " ".join(title.split())  # Clean whitespace

                # Abstract (summary)
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                abstract = summary_elem.text if summary_elem is not None else None
                if abstract:
                    abstract = " ".join(abstract.split())

                # Authors
                authors = []
                for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                    name_elem = author.find("{http://www.w3.org/2005/Atom}name")
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)

                # Published date -> year
                published_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                year = None
                if published_elem is not None and published_elem.text:
                    try:
                        year = int(published_elem.text[:4])
                    except (ValueError, IndexError):
                        pass

                # Category
                category_elem = entry.find("{http://arxiv.org/schemas/atom}primary_category")
                category = None
                if category_elem is not None:
                    category = category_elem.get("term")

                # PDF link
                pdf_url = None
                for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href")
                        break

                # DOI (if available in journal_ref or comment)
                doi = None
                journal_ref = entry.find("{http://arxiv.org/schemas/atom}journal_ref")
                if journal_ref is not None and journal_ref.text:
                    doi_match = re.search(r"10\.\d{4,}/[^\s]+", journal_ref.text)
                    if doi_match:
                        doi = doi_match.group(0)

                results.append(
                    LiteratureResult(
                        id=arxiv_id,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        year=year,
                        journal="arXiv",
                        doi=doi,
                        url=arxiv_url,
                        source=self.source_name,
                        score=1.0,
                        metadata={"category": category, "pdf_url": pdf_url},
                    )
                )

            except Exception as e:
                logger.warning("arxiv_parse_error", error=str(e))
                continue

        return results

    async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
        """Get a paper by arXiv ID.

        Args:
            paper_id: arXiv ID (e.g., "2301.12345").

        Returns:
            LiteratureResult or None if not found.
        """
        try:
            params = {"id_list": paper_id}
            response = await self.client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()

            results = self._parse_arxiv_response(response.text)
            return results[0] if results else None

        except Exception as e:
            logger.error("arxiv_fetch_error", paper_id=paper_id, error=str(e))
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
