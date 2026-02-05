"""Unit tests for main module."""

from unittest.mock import patch


class TestMainModule:
    """Tests for main module."""

    def test_app_export(self) -> None:
        """Test that app is exported from main."""
        from aria.main import app

        assert app is not None

    def test_main_function_exists(self) -> None:
        """Test that main function exists."""
        from aria.main import main

        assert callable(main)

    def test_main_calls_uvicorn_run(self) -> None:
        """Test that main calls uvicorn.run with correct arguments."""
        with (
            patch("aria.main.uvicorn") as mock_uvicorn,
            patch("aria.main.settings") as mock_settings,
        ):
            mock_settings.api_host = "0.0.0.0"  # noqa: S104
            mock_settings.api_port = 8000
            mock_settings.debug = False
            mock_settings.api_workers = 4
            mock_settings.log_level = "INFO"

            from aria.main import main

            main()

            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"  # noqa: S104
            assert call_kwargs["port"] == 8000
            assert call_kwargs["workers"] == 4
            assert call_kwargs["log_level"] == "info"

    def test_main_uses_single_worker_in_debug(self) -> None:
        """Test that main uses single worker in debug mode."""
        with (
            patch("aria.main.uvicorn") as mock_uvicorn,
            patch("aria.main.settings") as mock_settings,
        ):
            mock_settings.api_host = "localhost"
            mock_settings.api_port = 8000
            mock_settings.debug = True
            mock_settings.api_workers = 4  # Should be ignored
            mock_settings.log_level = "DEBUG"

            from aria.main import main

            main()

            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["workers"] == 1
            assert call_kwargs["reload"] is True

    def test_module_all_exports(self) -> None:
        """Test that __all__ includes expected exports."""
        from aria.main import __all__

        assert "app" in __all__
        assert "main" in __all__
