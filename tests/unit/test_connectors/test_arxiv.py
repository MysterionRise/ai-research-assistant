"""Unit tests for arXiv connector."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.connectors.arxiv import ARXIV_API_URL, ArxivConnector


class TestArxivConnectorInit:
    """Tests for ArxivConnector initialization."""

    def test_init(self) -> None:
        """Test initialization."""
        connector = ArxivConnector()

        assert connector.client is not None
        assert connector.source_name == "arxiv"

    def test_atom_namespace(self) -> None:
        """Test ATOM namespace constant."""
        assert "atom" in ArxivConnector.ATOM_NS
        assert ArxivConnector.ATOM_NS["atom"] == "http://www.w3.org/2005/Atom"


class TestArxivConnectorConstants:
    """Tests for arXiv connector constants."""

    def test_api_url(self) -> None:
        """Test ARXIV_API_URL constant."""
        assert ARXIV_API_URL == "http://export.arxiv.org/api/query"


class TestArxivConnectorSourceName:
    """Tests for source_name property."""

    def test_source_name(self) -> None:
        """Test that source_name returns 'arxiv'."""
        connector = ArxivConnector()
        assert connector.source_name == "arxiv"


class TestArxivConnectorSearch:
    """Tests for ArxivConnector.search method."""

    @pytest.mark.asyncio
    async def test_search_connector_error(self) -> None:
        """Test search handling errors (retries exhausted)."""
        import tenacity

        connector = ArxivConnector()

        # Mock the httpx client to raise an exception
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Network error"))
        connector.client = mock_client

        # The retry decorator will exhaust retries and raise RetryError
        with pytest.raises(tenacity.RetryError):
            await connector.search("quantum computing")

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search with empty results."""
        connector = ArxivConnector()

        # Mock response with empty results XML
        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        mock_response = MagicMock()
        mock_response.text = empty_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        connector.client = mock_client

        results = await connector.search("nonexistent topic xyz123")

        assert results == []


class TestArxivConnectorClose:
    """Tests for ArxivConnector.close method."""

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing the connector."""
        connector = ArxivConnector()

        # Mock the httpx client
        connector.client = MagicMock()
        connector.client.aclose = AsyncMock()

        await connector.close()

        connector.client.aclose.assert_called_once()


class TestArxivConnectorParsing:
    """Tests for Arxiv result parsing."""

    def test_parse_arxiv_response_valid_xml(self) -> None:
        """Test parsing valid arXiv XML response."""
        connector = ArxivConnector()

        valid_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2401.12345v1</id>
                <title>Test Paper Title</title>
                <summary>This is the abstract of the paper.</summary>
                <author><name>John Doe</name></author>
                <author><name>Jane Smith</name></author>
                <published>2024-01-15T00:00:00Z</published>
                <link href="http://arxiv.org/abs/2401.12345v1"/>
            </entry>
        </feed>"""

        results = connector._parse_arxiv_response(valid_xml)

        assert len(results) == 1
        assert "Test Paper Title" in results[0].title
        assert results[0].source == "arxiv"

    def test_parse_arxiv_response_empty_feed(self) -> None:
        """Test parsing empty arXiv feed."""
        connector = ArxivConnector()

        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        results = connector._parse_arxiv_response(empty_xml)

        assert len(results) == 0


class TestArxivConnectorSearchParams:
    """Tests for search parameter handling."""

    @pytest.mark.asyncio
    async def test_search_with_category(self) -> None:
        """Test search with category parameter."""
        connector = ArxivConnector()

        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        mock_response = MagicMock()
        mock_response.text = empty_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        connector.client = mock_client

        await connector.search("test query", category="cs.AI")

        # Check that category was included in params
        call_kwargs = mock_client.get.call_args[1]
        assert "cs.AI" in call_kwargs["params"]["search_query"]
