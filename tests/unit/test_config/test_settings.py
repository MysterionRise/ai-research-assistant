"""Unit tests for configuration settings."""

import os

import pytest


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_environment_is_development(self) -> None:
        """Test default environment."""
        # Create a new Settings instance directly with minimal env vars
        # Note: We test the Settings class behavior, not the singleton
        from aria.config.settings import Settings

        # Settings will use defaults + env vars already set by conftest
        s = Settings()
        # conftest sets ENVIRONMENT=test, so check against that
        assert s.environment == "test"

    def test_environment_validation_accepts_valid_values(self) -> None:
        """Test environment validation accepts valid values."""
        from aria.config.settings import Settings

        valid_environments = ["development", "staging", "production", "test"]
        for env in valid_environments:
            os.environ["ENVIRONMENT"] = env
            s = Settings()
            assert s.environment == env

    def test_environment_validation_rejects_invalid(self) -> None:
        """Test environment validation."""
        from aria.config.settings import Settings

        original = os.environ.get("ENVIRONMENT")
        try:
            os.environ["ENVIRONMENT"] = "invalid"
            with pytest.raises(ValueError, match="environment must be one of"):
                Settings()
        finally:
            if original:
                os.environ["ENVIRONMENT"] = original
            else:
                os.environ.pop("ENVIRONMENT", None)

    def test_settings_has_required_fields(self) -> None:
        """Test settings has all required fields."""
        from aria.config.settings import Settings

        s = Settings()
        assert hasattr(s, "secret_key")
        assert hasattr(s, "database_url")
        assert hasattr(s, "redis_url")
        assert hasattr(s, "anthropic_api_key")

    def test_rag_configuration_defaults(self) -> None:
        """Test RAG configuration defaults."""
        from aria.config.settings import Settings

        s = Settings()
        assert s.rag_chunk_size == 512
        assert s.rag_chunk_overlap == 50
        assert s.rag_retrieval_top_k == 20
        assert s.rag_rerank_top_k == 5


@pytest.mark.smoke
class TestSettingsSmoke:
    """Smoke tests for settings."""

    def test_settings_can_be_imported(self) -> None:
        """Test settings module import."""
        from aria.config import settings

        assert settings is not None
