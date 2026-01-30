"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Core
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    secret_key: SecretStr = Field(...)
    log_level: str = Field(default="INFO")

    # API
    api_host: str = Field(default="0.0.0.0")  # noqa: S104
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # Database
    database_url: str = Field(default="postgresql+asyncpg://aria:aria@localhost:5432/aria_db")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # AI Models
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    openai_api_key: SecretStr | None = Field(default=None)

    # Vector DB
    pinecone_api_key: SecretStr | None = Field(default=None)
    pinecone_index_name: str = Field(default="aria-documents")

    # Feature Flags
    feature_molecular_search: bool = Field(default=True)
    feature_audit_trail: bool = Field(default=True)

    # Embedding Configuration
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)

    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # Literature API Configuration
    pubmed_email: str = Field(default="aria@company.com")
    pubmed_api_key: str | None = Field(default=None)
    semantic_scholar_api_key: str | None = Field(default=None)

    # RAG Configuration
    rag_chunk_size: int = Field(default=512)
    rag_chunk_overlap: int = Field(default=50)
    rag_retrieval_top_k: int = Field(default=20)
    rag_rerank_top_k: int = Field(default=5)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        allowed = {"development", "staging", "production", "test"}
        if v.lower() not in allowed:
            msg = f"environment must be one of {allowed}"
            raise ValueError(msg)
        return v.lower()

    @property
    def is_development(self) -> bool:
        """Check if development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if production mode."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
