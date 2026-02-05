"""Unit tests for PubMed connector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.connectors.pubmed import EFETCH_URL, ESEARCH_URL, PubMedConnector


class TestPubMedConnectorInit:
    """Tests for PubMedConnector initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default settings."""
        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = "test-api-key"

            connector = PubMedConnector()

            assert connector.email == "test@example.com"
            assert connector.api_key == "test-api-key"
            assert connector.client is not None

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        with patch("aria.connectors.pubmed.settings"):
            connector = PubMedConnector(
                email="custom@example.com",
                api_key="custom-key",
            )

            assert connector.email == "custom@example.com"
            assert connector.api_key == "custom-key"

    def test_source_name(self) -> None:
        """Test source_name property."""
        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = None
            mock_settings.pubmed_api_key = None

            connector = PubMedConnector()

            assert connector.source_name == "pubmed"


class TestPubMedConnectorConstants:
    """Tests for PubMed connector constants."""

    def test_esearch_url(self) -> None:
        """Test ESEARCH_URL constant."""
        assert ESEARCH_URL == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    def test_efetch_url(self) -> None:
        """Test EFETCH_URL constant."""
        assert EFETCH_URL == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class TestPubMedConnectorSearch:
    """Tests for PubMedConnector.search method."""

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search returning empty results."""
        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = None

            connector = PubMedConnector()

            # Mock _search_pmids to return empty list
            connector._search_pmids = AsyncMock(return_value=[])

            results = await connector.search("nonexistent query")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_results(self) -> None:
        """Test search returning results."""
        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = None

            from aria.connectors.base import LiteratureResult

            connector = PubMedConnector()

            # Mock the internal methods
            connector._search_pmids = AsyncMock(return_value=["12345678", "23456789"])
            connector._fetch_details = AsyncMock(
                return_value=[
                    LiteratureResult(
                        id="pmid:12345678",
                        title="Paper 1",
                        source="pubmed",
                    ),
                    LiteratureResult(
                        id="pmid:23456789",
                        title="Paper 2",
                        source="pubmed",
                    ),
                ]
            )

            results = await connector.search("CRISPR gene editing", limit=10)

            assert len(results) == 2
            assert results[0].id == "pmid:12345678"
            assert results[1].id == "pmid:23456789"
            connector._search_pmids.assert_called_once()
            connector._fetch_details.assert_called_once_with(["12345678", "23456789"])

    @pytest.mark.asyncio
    async def test_search_rate_limit_error(self) -> None:
        """Test search handling rate limit error (retries exhausted)."""
        import httpx
        import tenacity

        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = None

            connector = PubMedConnector()

            # Create a mock response for 429 status
            mock_response = MagicMock()
            mock_response.status_code = 429

            # Make _search_pmids raise HTTPStatusError
            connector._search_pmids = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Rate limited",
                    request=MagicMock(),
                    response=mock_response,
                )
            )

            # The retry decorator will exhaust retries and raise RetryError
            with pytest.raises(tenacity.RetryError):
                await connector.search("test query")

    @pytest.mark.asyncio
    async def test_search_connector_error(self) -> None:
        """Test search handling general errors (retries exhausted)."""
        import tenacity

        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = None

            connector = PubMedConnector()

            # Make _search_pmids raise a general exception
            connector._search_pmids = AsyncMock(side_effect=Exception("Network error"))

            # The retry decorator will exhaust retries and raise RetryError
            with pytest.raises(tenacity.RetryError):
                await connector.search("test query")


class TestPubMedConnectorClose:
    """Tests for PubMedConnector.close method."""

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing the connector."""
        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = None

            connector = PubMedConnector()

            # Mock the httpx client
            connector.client = MagicMock()
            connector.client.aclose = AsyncMock()

            await connector.close()

            connector.client.aclose.assert_called_once()


class TestPubMedConnectorGetById:
    """Tests for PubMedConnector.get_by_id method."""

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self) -> None:
        """Test get_by_id returns None when not found."""
        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = None

            connector = PubMedConnector()

            # Mock _fetch_details to return empty list
            connector._fetch_details = AsyncMock(return_value=[])

            result = await connector.get_by_id("nonexistent-pmid")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_found(self) -> None:
        """Test get_by_id returns result when found."""
        with patch("aria.connectors.pubmed.settings") as mock_settings:
            mock_settings.pubmed_email = "test@example.com"
            mock_settings.pubmed_api_key = None

            from aria.connectors.base import LiteratureResult

            connector = PubMedConnector()

            expected_result = LiteratureResult(
                id="pmid:12345678",
                title="Found Paper",
                source="pubmed",
            )
            connector._fetch_details = AsyncMock(return_value=[expected_result])

            result = await connector.get_by_id("12345678")

            assert result is not None
            assert result.id == "pmid:12345678"


class TestPubMedConnectorUrlBuilding:
    """Tests for URL and parameter building."""

    def test_esearch_url_constant(self) -> None:
        """Test ESEARCH_URL is correctly set."""
        assert "esearch.fcgi" in ESEARCH_URL

    def test_efetch_url_constant(self) -> None:
        """Test EFETCH_URL is correctly set."""
        assert "efetch.fcgi" in EFETCH_URL
