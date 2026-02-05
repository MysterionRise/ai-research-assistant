"""Integration tests for API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    async def test_health_check_returns_200(self, async_client: AsyncClient) -> None:
        """Test basic health check returns 200."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    async def test_api_health_check_returns_200(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test API health check returns 200 with metadata."""
        response = await async_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "environment" in data

    async def test_readiness_check_returns_status(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test readiness check returns component statuses."""
        response = await async_client.get("/api/v1/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data

    async def test_liveness_check_returns_alive(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test liveness check returns alive status."""
        response = await async_client.get("/api/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


@pytest.mark.integration
class TestRootEndpoint:
    """Tests for root endpoint."""

    async def test_root_returns_api_info(self, async_client: AsyncClient) -> None:
        """Test root endpoint returns API information."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "ARIA - AI Research Intelligence Assistant"


@pytest.mark.integration
class TestChatEndpoints:
    """Tests for chat endpoints."""

    async def test_send_message_returns_response(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test sending a chat message returns a response."""
        response = await async_client.post(
            "/api/v1/chat",
            json={"message": "What are the latest advances in glass fiber optics?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "message" in data
        assert data["message"]["role"] == "assistant"

    async def test_send_empty_message_returns_422(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test sending empty message returns validation error."""
        response = await async_client.post(
            "/api/v1/chat",
            json={"message": ""},
        )

        assert response.status_code == 422


@pytest.mark.integration
class TestSearchEndpoints:
    """Tests for search endpoints."""

    async def test_literature_search_returns_results(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test literature search returns results structure."""
        response = await async_client.post(
            "/api/v1/search/literature",
            json={"query": "glass fiber optics"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "total_results" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    async def test_quick_search_via_get(self, async_client: AsyncClient) -> None:
        """Test quick search via GET endpoint."""
        response = await async_client.get(
            "/api/v1/search/literature/quick",
            params={"q": "glass fiber", "limit": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data

    async def test_molecular_search_returns_results(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test molecular search returns results structure."""
        response = await async_client.post(
            "/api/v1/search/molecules",
            json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "query_smiles" in data
        assert "results" in data


@pytest.mark.integration
class TestDocumentEndpoints:
    """Tests for document endpoints."""

    async def test_list_documents_returns_paginated_response(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test listing documents returns paginated response."""
        response = await async_client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "documents" in data
        assert "page" in data
        assert "page_size" in data

    async def test_get_nonexistent_document_returns_404(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test getting nonexistent document returns 404."""
        response = await async_client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


@pytest.mark.integration
class TestProtocolEndpoints:
    """Tests for protocol endpoints."""

    async def test_list_protocols_returns_paginated_response(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test listing protocols returns paginated response."""
        response = await async_client.get("/api/v1/protocols")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "protocols" in data

    async def test_create_protocol_returns_created_protocol(
        self,
        async_client: AsyncClient,
    ) -> None:
        """Test creating a protocol returns the created protocol."""
        response = await async_client.post(
            "/api/v1/protocols",
            json={
                "name": "Test Protocol",
                "description": "A test protocol for glass synthesis",
                "steps": [
                    {
                        "step_number": 1,
                        "title": "Prepare materials",
                        "description": "Gather all necessary materials",
                    },
                ],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Protocol"
        assert data["status"] == "draft"
        assert len(data["steps"]) == 1
