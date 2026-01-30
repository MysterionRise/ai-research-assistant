"""PubMed connector using E-utilities API."""

import xml.etree.ElementTree as ET
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from aria.config.settings import settings
from aria.connectors.base import BaseConnector, LiteratureResult
from aria.exceptions import ConnectorError, RateLimitError

logger = structlog.get_logger(__name__)

# PubMed E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedConnector(BaseConnector):
    """PubMed literature connector using NCBI E-utilities.

    Implements search and fetch using the E-utilities API with
    rate limiting and retry logic.
    """

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize PubMed connector.

        Args:
            email: Email for NCBI (required for higher rate limits).
            api_key: NCBI API key (optional, for higher rate limits).
        """
        self.email = email or settings.pubmed_email
        self.api_key = api_key or settings.pubmed_api_key
        self.client = httpx.AsyncClient(timeout=30.0)

        logger.info("pubmed_connector_initialized", email=self.email)

    @property
    def source_name(self) -> str:
        """Return source name."""
        return "pubmed"

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
        """Search PubMed for literature.

        Args:
            query: Search query (supports PubMed query syntax).
            limit: Maximum results to return.
            **kwargs: Additional parameters (year_from, year_to, etc.).

        Returns:
            List of PubMed results.

        Raises:
            ConnectorError: If search fails.
            RateLimitError: If rate limited.
        """
        logger.info("pubmed_search", query=query, limit=limit)

        try:
            # Step 1: Search for PMIDs
            pmids = await self._search_pmids(query, limit, **kwargs)

            if not pmids:
                return []

            # Step 2: Fetch details for PMIDs
            results = await self._fetch_details(pmids)

            logger.info(
                "pubmed_search_completed",
                query=query,
                results_count=len(results),
            )

            return results

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("pubmed") from e
            raise ConnectorError("pubmed", str(e)) from e
        except Exception as e:
            raise ConnectorError("pubmed", str(e)) from e

    async def _search_pmids(
        self,
        query: str,
        limit: int,
        **kwargs,
    ) -> list[str]:
        """Search for PMIDs matching query.

        Args:
            query: Search query.
            limit: Maximum results.
            **kwargs: Additional filters.

        Returns:
            List of PMIDs.
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
            "tool": "aria",
            "email": self.email,
        }

        # Add date filters if provided
        if kwargs.get("year_from"):
            params["mindate"] = f"{kwargs['year_from']}/01/01"
        if kwargs.get("year_to"):
            params["maxdate"] = f"{kwargs['year_to']}/12/31"
        if kwargs.get("year_from") or kwargs.get("year_to"):
            params["datetype"] = "pdat"

        if self.api_key:
            params["api_key"] = self.api_key

        response = await self.client.get(ESEARCH_URL, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    async def _fetch_details(self, pmids: list[str]) -> list[LiteratureResult]:
        """Fetch details for a list of PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of LiteratureResult objects.
        """
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "aria",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        response = await self.client.get(EFETCH_URL, params=params)
        response.raise_for_status()

        return self._parse_pubmed_xml(response.text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[LiteratureResult]:
        """Parse PubMed XML response.

        Args:
            xml_text: XML response from efetch.

        Returns:
            List of LiteratureResult objects.
        """
        results = []
        root = ET.fromstring(xml_text)

        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find("MedlineCitation")
                if medline is None:
                    continue

                pmid_elem = medline.find("PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""

                article_elem = medline.find("Article")
                if article_elem is None:
                    continue

                # Title
                title_elem = article_elem.find("ArticleTitle")
                title = title_elem.text if title_elem is not None else "No title"

                # Abstract
                abstract_elem = article_elem.find(".//AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else None

                # Authors
                authors = []
                author_list = article_elem.find("AuthorList")
                if author_list is not None:
                    for author in author_list.findall("Author"):
                        last = author.find("LastName")
                        first = author.find("ForeName")
                        if last is not None:
                            name = last.text
                            if first is not None:
                                name = f"{first.text} {name}"
                            authors.append(name)

                # Journal
                journal_elem = article_elem.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else None

                # Year
                year = None
                year_elem = article_elem.find(".//PubDate/Year")
                if year_elem is not None and year_elem.text:
                    try:
                        year = int(year_elem.text)
                    except ValueError:
                        pass

                # DOI
                doi = None
                for id_elem in article.findall(".//ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        doi = id_elem.text
                        break

                results.append(
                    LiteratureResult(
                        id=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        year=year,
                        journal=journal,
                        doi=doi,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        source=self.source_name,
                        score=1.0,  # PubMed doesn't provide relevance scores
                    )
                )

            except Exception as e:
                logger.warning("pubmed_parse_error", error=str(e))
                continue

        return results

    async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
        """Get a paper by PMID.

        Args:
            paper_id: PubMed ID (PMID).

        Returns:
            LiteratureResult or None if not found.
        """
        results = await self._fetch_details([paper_id])
        return results[0] if results else None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
