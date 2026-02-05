"""Unit tests for Semantic Scholar connector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.connectors.semantic_scholar import S2_API_URL, SemanticScholarConnector


class TestSemanticScholarConnectorInit:
    """Tests for SemanticScholarConnector initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector(api_key="test-api-key")

            assert connector.api_key == "test-api-key"
            assert connector.client is not None

    def test_init_with_settings_api_key(self) -> None:
        """Test initialization with API key from settings."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = "settings-api-key"

            connector = SemanticScholarConnector()

            assert connector.api_key == "settings-api-key"

    def test_init_without_api_key(self) -> None:
        """Test initialization without API key."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            assert connector.api_key is None
            assert connector.client is not None


class TestSemanticScholarConnectorConstants:
    """Tests for Semantic Scholar connector constants."""

    def test_api_url(self) -> None:
        """Test S2_API_URL constant."""
        assert S2_API_URL == "https://api.semanticscholar.org/graph/v1"


class TestSemanticScholarConnectorSourceName:
    """Tests for source_name property."""

    def test_source_name(self) -> None:
        """Test that source_name returns 'semantic_scholar'."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            assert connector.source_name == "semantic_scholar"


class TestSemanticScholarConnectorSearch:
    """Tests for SemanticScholarConnector.search method."""

    @pytest.mark.asyncio
    async def test_search_rate_limit_error(self) -> None:
        """Test search handling rate limit error (retries exhausted)."""
        import httpx
        import tenacity

        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            # Create a mock response for 429 status
            mock_response = MagicMock()
            mock_response.status_code = 429

            # Make client.get raise HTTPStatusError
            mock_client = MagicMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Rate limited",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            connector.client = mock_client

            # The retry decorator will exhaust retries and raise RetryError
            with pytest.raises(tenacity.RetryError):
                await connector.search("test query")

    @pytest.mark.asyncio
    async def test_search_connector_error(self) -> None:
        """Test search handling general errors (retries exhausted)."""
        import tenacity

        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            # Make client.get raise a general exception
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Network error"))
            connector.client = mock_client

            # The retry decorator will exhaust retries and raise RetryError
            with pytest.raises(tenacity.RetryError):
                await connector.search("test query")

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search with empty results."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            # Mock response with empty results
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": []}
            mock_response.raise_for_status = MagicMock()

            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            connector.client = mock_client

            results = await connector.search("nonexistent topic xyz123")

            assert results == []


class TestSemanticScholarConnectorGetById:
    """Tests for SemanticScholarConnector.get_by_id method."""

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self) -> None:
        """Test get_by_id when paper is not found."""
        import httpx

        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            # Create a mock response for 404 status
            mock_response = MagicMock()
            mock_response.status_code = 404

            # Make client.get raise HTTPStatusError for 404
            mock_client = MagicMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not found",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            connector.client = mock_client

            result = await connector.get_by_id("nonexistent-paper-id")

            assert result is None


class TestSemanticScholarConnectorClose:
    """Tests for SemanticScholarConnector.close method."""

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing the connector."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            # Mock the httpx client
            connector.client = MagicMock()
            connector.client.aclose = AsyncMock()

            await connector.close()

            connector.client.aclose.assert_called_once()


class TestSemanticScholarConnectorParseResults:
    """Tests for _parse_results method."""

    def test_parse_results_with_full_data(self) -> None:
        """Test parsing results with complete paper data."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            papers = [
                {
                    "paperId": "abc123",
                    "title": "Test Paper",
                    "abstract": "This is the abstract.",
                    "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
                    "year": 2024,
                    "venue": "Nature",
                    "externalIds": {"DOI": "10.1234/test"},
                    "url": "https://example.com/paper",
                    "citationCount": 100,
                    "influentialCitationCount": 10,
                }
            ]

            results = connector._parse_results(papers)

            assert len(results) == 1
            assert results[0].id == "abc123"
            assert results[0].title == "Test Paper"
            assert len(results[0].authors) == 2
            assert results[0].doi == "10.1234/test"
            assert results[0].year == 2024

    def test_parse_results_with_missing_fields(self) -> None:
        """Test parsing results with missing optional fields."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            papers = [
                {
                    "paperId": "minimal",
                    "title": "Minimal Paper",
                }
            ]

            results = connector._parse_results(papers)

            assert len(results) == 1
            assert results[0].id == "minimal"
            assert results[0].authors == []
            assert results[0].abstract is None

    def test_parse_results_empty_list(self) -> None:
        """Test parsing empty results list."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            results = connector._parse_results([])

            assert len(results) == 0

    def test_parse_results_calculates_score(self) -> None:
        """Test that score is calculated from citations."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            # Paper with many citations should have higher score
            high_citation_paper = {
                "paperId": "high",
                "title": "High Citation Paper",
                "citationCount": 1000,
                "influentialCitationCount": 50,
            }

            # Paper with no citations
            no_citation_paper = {
                "paperId": "low",
                "title": "Low Citation Paper",
                "citationCount": 0,
            }

            results = connector._parse_results([high_citation_paper, no_citation_paper])

            assert len(results) == 2
            # High citation paper should have higher score
            high_score = next(r for r in results if r.id == "high").score
            low_score = next(r for r in results if r.id == "low").score
            assert high_score > low_score


class TestSemanticScholarConnectorGetReferences:
    """Tests for get_references method."""

    @pytest.mark.asyncio
    async def test_get_references_error_returns_empty(self) -> None:
        """Test that get_references returns empty list on error."""
        with patch("aria.connectors.semantic_scholar.settings") as mock_settings:
            mock_settings.semantic_scholar_api_key = None

            connector = SemanticScholarConnector()

            # Mock client to raise exception
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Network error"))
            connector.client = mock_client

            results = await connector.get_references("paper-123")

            assert results == []
