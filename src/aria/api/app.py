"""FastAPI application factory.

Creates and configures the ARIA API application.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from aria.api.routes import chat, documents, health, protocols, search
from aria.config.settings import settings
from aria.db.session import close_db, init_db

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events.

    Args:
        _app: FastAPI application instance (unused but required by FastAPI).

    Yields:
        None
    """
    # Startup
    logger.info(
        "starting_application",
        environment=settings.environment,
        debug=settings.debug,
    )

    # Initialize database connections
    await init_db()

    # Initialize Redis connections
    # TODO: Add Redis connection pool

    logger.info("application_started")

    yield

    # Shutdown
    logger.info("shutting_down_application")

    # Close database connections
    await close_db()

    logger.info("application_shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title="ARIA - AI Research Intelligence Assistant",
        description=(
            "Enterprise-grade AI Research Assistant for Life Sciences "
            "and Materials Science R&D"
        ),
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Register routes
    _register_routes(app)

    # Register exception handlers
    _register_exception_handlers(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register API routes.

    Args:
        app: FastAPI application instance.
    """
    api_prefix = "/api/v1"

    # Health checks (no prefix for k8s probes)
    app.include_router(health.router, tags=["Health"])

    # API routes
    app.include_router(chat.router, prefix=api_prefix, tags=["Chat"])
    app.include_router(search.router, prefix=api_prefix, tags=["Search"])
    app.include_router(documents.router, prefix=api_prefix, tags=["Documents"])
    app.include_router(protocols.router, prefix=api_prefix, tags=["Protocols"])


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers.

    Args:
        app: FastAPI application instance.
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions.

        Args:
            request: Incoming request.
            exc: Exception that was raised.

        Returns:
            JSONResponse: Error response.
        """
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )

        # Don't expose internal errors in production
        detail = str(exc) if settings.debug else "Internal server error"

        return JSONResponse(
            status_code=500,
            content={"detail": detail, "type": "internal_error"},
        )


# Create default application instance
app = create_app()


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint.

    Returns:
        dict: Welcome message and API info.
    """
    return {
        "name": "ARIA - AI Research Intelligence Assistant",
        "version": "0.1.0",
        "docs": "/docs" if settings.debug else None,
        "health": "/api/v1/health",
    }
