"""Unit tests for FastAPI application factory."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """Test that create_app returns a FastAPI instance."""
        from aria.api.app import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_correct_title(self) -> None:
        """Test that app has correct title."""
        from aria.api.app import create_app

        app = create_app()
        assert "ARIA" in app.title

    def test_app_has_correct_version(self) -> None:
        """Test that app has correct version."""
        from aria.api.app import create_app

        app = create_app()
        assert app.version == "0.1.0"

    def test_app_has_cors_middleware(self) -> None:
        """Test that CORS middleware is added."""
        from aria.api.app import create_app

        app = create_app()
        middleware_classes = [m.cls for m in app.user_middleware]
        assert CORSMiddleware in middleware_classes

    def test_app_has_gzip_middleware(self) -> None:
        """Test that GZip middleware is added."""
        from aria.api.app import create_app

        app = create_app()
        middleware_classes = [m.cls for m in app.user_middleware]
        assert GZipMiddleware in middleware_classes


class TestRouteRegistration:
    """Tests for route registration."""

    def test_health_routes_registered(self) -> None:
        """Test that health routes are registered."""
        from aria.api.app import create_app

        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/api/v1/health" in routes

    def test_api_routes_registered(self) -> None:
        """Test that API routes are registered."""
        from aria.api.app import create_app

        app = create_app()
        routes = [r.path for r in app.routes]

        # Check that major route prefixes exist
        api_routes = [r for r in routes if r.startswith("/api/v1")]
        assert len(api_routes) > 0

    def test_root_endpoint_registered(self) -> None:
        """Test that root endpoint is registered on default app."""
        # The root endpoint is added to the default `app` instance, not create_app()
        from aria.api.app import app

        routes = [r.path for r in app.routes]
        assert "/" in routes


class TestDefaultApp:
    """Tests for default app instance."""

    def test_app_instance_exists(self) -> None:
        """Test that default app instance is created."""
        from aria.api.app import app

        assert app is not None
        assert isinstance(app, FastAPI)


class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_calls_init_db(self) -> None:
        """Test that lifespan calls init_db on startup."""
        from aria.api.app import create_app, lifespan

        app = create_app()

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock) as mock_init,
            patch("aria.api.app.close_db", new_callable=AsyncMock) as mock_close,
        ):
            async with lifespan(app):
                mock_init.assert_called_once()

            # close_db should be called after exiting context
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_calls_close_db_on_shutdown(self) -> None:
        """Test that lifespan calls close_db on shutdown."""
        from aria.api.app import create_app, lifespan

        app = create_app()

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock) as mock_close,
        ):
            async with lifespan(app):
                pass  # startup
            # After exiting context, shutdown should have occurred
            mock_close.assert_called_once()


class TestExceptionHandlers:
    """Tests for exception handlers."""

    def test_exception_handlers_are_registered(self) -> None:
        """Test that exception handlers are registered."""
        from aria.api.app import create_app

        app = create_app()

        # Check that the app has exception handlers configured
        # The global exception handler should be registered
        assert app.exception_handlers is not None
        assert Exception in app.exception_handlers


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_api_info(self) -> None:
        """Test that root endpoint returns API info."""
        from httpx import ASGITransport, AsyncClient

        # The root endpoint is registered on the default app instance
        from aria.api.app import app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/")

            assert response.status_code == 200
            data = response.json()
            assert "name" in data
            assert "ARIA" in data["name"]
            assert "version" in data
            assert "health" in data


class TestDocsConfiguration:
    """Tests for documentation endpoint configuration."""

    def test_docs_enabled_in_debug_mode(self) -> None:
        """Test that docs are enabled in debug mode."""
        from aria.api.app import create_app

        with patch("aria.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["*"]
            mock_settings.environment = "test"
            app = create_app()

            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"
            assert app.openapi_url == "/openapi.json"

    def test_docs_disabled_in_production(self) -> None:
        """Test that docs are disabled in production."""
        from aria.api.app import create_app

        with patch("aria.api.app.settings") as mock_settings:
            mock_settings.debug = False
            mock_settings.cors_origins = ["*"]
            mock_settings.environment = "production"
            app = create_app()

            assert app.docs_url is None
            assert app.redoc_url is None
            assert app.openapi_url is None
