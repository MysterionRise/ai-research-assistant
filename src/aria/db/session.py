"""Database session management.

Provides async session factory and dependency injection for FastAPI.
"""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from aria.config.settings import settings

logger = structlog.get_logger(__name__)

# Create async engine
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope for a series of operations.

    Yields:
        AsyncSession: Database session for request scope.

    Example:
        ```python
        @router.get("/items")
        async def get_items(
            session: AsyncSession = Depends(get_async_session),
        ):
            result = await session.execute(select(Item))
            return result.scalars().all()
        ```
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database connection.

    Creates all tables if they don't exist.
    Should be called during application startup.
    """
    from aria.db.base import Base
    from aria.db.models import Chunk, Conversation, Document, Message  # noqa: F401

    logger.info("initializing_database", url=settings.database_url.split("@")[-1])

    async with engine.begin() as conn:
        # Create pgvector extension if it doesn't exist
        await conn.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"  # type: ignore[arg-type]
        )
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    logger.info("database_initialized")


async def close_db() -> None:
    """Close database connections.

    Should be called during application shutdown.
    """
    logger.info("closing_database_connections")
    await engine.dispose()
    logger.info("database_connections_closed")
