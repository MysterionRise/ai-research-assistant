"""Unit tests for health check endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_healthy_status(self) -> None:
        """Test that /health returns healthy status."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_includes_timestamp(self) -> None:
        """Test that /health includes timestamp."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/health")

            data = response.json()
            assert "timestamp" in data
            # Timestamp should be ISO format
            assert "T" in data["timestamp"]

    @pytest.mark.asyncio
    async def test_health_includes_version(self) -> None:
        """Test that /health includes version."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/health")

            data = response.json()
            assert "version" in data
            assert data["version"] == "0.1.0"


class TestApiHealthEndpoint:
    """Tests for /api/v1/health endpoint."""

    @pytest.mark.asyncio
    async def test_api_health_returns_healthy_status(self) -> None:
        """Test that /api/v1/health returns healthy status."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_api_health_includes_environment(self) -> None:
        """Test that /api/v1/health includes environment."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health")

            data = response.json()
            assert "environment" in data


class TestReadinessEndpoint:
    """Tests for /api/v1/health/ready endpoint."""

    @pytest.mark.asyncio
    async def test_readiness_returns_ready_status(self) -> None:
        """Test that /api/v1/health/ready returns ready status."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_readiness_includes_checks(self) -> None:
        """Test that /api/v1/health/ready includes component checks."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health/ready")

            data = response.json()
            assert "checks" in data
            checks = data["checks"]
            assert "database" in checks
            assert "redis" in checks
            assert "vector_store" in checks

    @pytest.mark.asyncio
    async def test_readiness_check_structure(self) -> None:
        """Test that readiness checks have correct structure."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health/ready")

            data = response.json()
            for check_name, check_data in data["checks"].items():
                assert "status" in check_data
                assert "latency_ms" in check_data


class TestLivenessEndpoint:
    """Tests for /api/v1/health/live endpoint."""

    @pytest.mark.asyncio
    async def test_liveness_returns_alive_status(self) -> None:
        """Test that /api/v1/health/live returns alive status."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health/live")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "alive"

    @pytest.mark.asyncio
    async def test_liveness_minimal_response(self) -> None:
        """Test that liveness check returns minimal response."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health/live")

            data = response.json()
            # Liveness should be minimal - just status
            assert "status" in data
            # Should not have complex checks like readiness
            assert "checks" not in data


class TestHealthEndpointStatusCodes:
    """Tests for health endpoint HTTP status codes."""

    @pytest.mark.asyncio
    async def test_all_health_endpoints_return_200(self) -> None:
        """Test that all health endpoints return 200 status code."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            endpoints = [
                "/health",
                "/api/v1/health",
                "/api/v1/health/ready",
                "/api/v1/health/live",
            ]

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                for endpoint in endpoints:
                    response = await client.get(endpoint)
                    assert response.status_code == 200, f"Failed for {endpoint}"
