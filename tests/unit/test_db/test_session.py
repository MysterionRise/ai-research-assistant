"""Unit tests for database session management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDatabaseEngine:
    """Tests for database engine configuration."""

    def test_engine_exists(self) -> None:
        """Test that engine is created."""
        from aria.db.session import engine

        assert engine is not None

    def test_async_session_maker_exists(self) -> None:
        """Test that session maker is created."""
        from aria.db.session import async_session_maker

        assert async_session_maker is not None


class TestGetAsyncSession:
    """Tests for get_async_session function."""

    @pytest.mark.asyncio
    async def test_get_async_session_yields_session(self) -> None:
        """Test that get_async_session yields a session and commits."""
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_maker = MagicMock()
        mock_session_maker.return_value = mock_session

        with patch("aria.db.session.async_session_maker", mock_session_maker):
            from aria.db.session import get_async_session

            async for session in get_async_session():
                assert session is mock_session

    @pytest.mark.asyncio
    async def test_get_async_session_commits_on_success(self) -> None:
        """Test that session is committed on success."""
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_maker = MagicMock()
        mock_session_maker.return_value = mock_session

        with patch("aria.db.session.async_session_maker", mock_session_maker):
            from aria.db.session import get_async_session

            async for _ in get_async_session():
                pass  # Normal flow

            mock_session.commit.assert_called_once()
            mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_async_session_rollbacks_on_exception(self) -> None:
        """Test that session is rolled back on exception."""
        mock_session = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("DB Error"))
        mock_session.rollback = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_maker = MagicMock()
        mock_session_maker.return_value = mock_session

        with (
            patch("aria.db.session.async_session_maker", mock_session_maker),
            pytest.raises(Exception, match="DB Error"),
        ):
            from aria.db.session import get_async_session

            async for _ in get_async_session():
                pass

        mock_session.rollback.assert_called_once()


class TestInitDb:
    """Tests for init_db function."""

    @pytest.mark.asyncio
    async def test_init_db_creates_extension_and_tables(self) -> None:
        """Test that init_db creates pgvector extension and tables."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        with (
            patch("aria.db.session.engine", mock_engine),
            patch("aria.db.session.settings") as mock_settings,
        ):
            mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost/db"

            from aria.db.session import init_db

            await init_db()

            # Should have executed the CREATE EXTENSION command
            mock_conn.execute.assert_called_once()
            call_args = mock_conn.execute.call_args[0][0]
            assert "vector" in call_args

            # Should have run sync to create tables
            mock_conn.run_sync.assert_called_once()


class TestCloseDb:
    """Tests for close_db function."""

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self) -> None:
        """Test that close_db disposes the engine."""
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with patch("aria.db.session.engine", mock_engine):
            from aria.db.session import close_db

            await close_db()

            mock_engine.dispose.assert_called_once()


class TestModuleExports:
    """Tests for module exports."""

    def test_get_async_session_exported(self) -> None:
        """Test that get_async_session is exported."""
        from aria.db.session import get_async_session

        assert callable(get_async_session)

    def test_init_db_exported(self) -> None:
        """Test that init_db is exported."""
        from aria.db.session import init_db

        assert callable(init_db)

    def test_close_db_exported(self) -> None:
        """Test that close_db is exported."""
        from aria.db.session import close_db

        assert callable(close_db)
