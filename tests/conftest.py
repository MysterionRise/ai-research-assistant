"""Pytest configuration and shared fixtures."""

import os

# Set environment variables BEFORE any aria imports
# This is critical because settings are loaded at import time
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for API testing."""
    from aria.api.app import create_app

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def sample_document() -> dict[str, Any]:
    """Sample document for testing."""
    return {
        "title": "Test Scientific Paper",
        "abstract": "Test abstract for a scientific paper.",
        "content": "Full content here...",
    }


@pytest.fixture
def sample_query() -> str:
    """Sample search query."""
    return "What are the latest advances in glass fiber optics?"
