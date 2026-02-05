"""Unit tests for FastAPI dependencies."""

from unittest.mock import MagicMock, patch

import pytest


class TestDependencyFunctions:
    """Tests for dependency injection functions."""

    def test_get_embedder_returns_singleton(self) -> None:
        """Test that get_embedder returns a singleton."""
        with patch("aria.api.dependencies.OpenAIEmbedder") as mock_embedder:
            mock_embedder.return_value = MagicMock()

            from aria.api.dependencies import get_embedder

            # Clear cache if exists
            get_embedder.cache_clear()

            result1 = get_embedder()
            result2 = get_embedder()

            assert result1 is result2
            # Should only be called once due to caching
            mock_embedder.assert_called_once()

    def test_get_vector_store_returns_singleton(self) -> None:
        """Test that get_vector_store returns a singleton."""
        with patch("aria.api.dependencies.PgVectorStore") as mock_store:
            mock_store.return_value = MagicMock()

            from aria.api.dependencies import get_vector_store

            get_vector_store.cache_clear()

            result1 = get_vector_store()
            result2 = get_vector_store()

            assert result1 is result2
            mock_store.assert_called_once()

    def test_get_rag_pipeline_returns_singleton(self) -> None:
        """Test that get_rag_pipeline returns a singleton."""
        with patch("aria.api.dependencies.RAGPipeline") as mock_pipeline:
            mock_pipeline.return_value = MagicMock()

            from aria.api.dependencies import get_rag_pipeline

            get_rag_pipeline.cache_clear()

            result1 = get_rag_pipeline()
            result2 = get_rag_pipeline()

            assert result1 is result2
            mock_pipeline.assert_called_once()

    def test_get_literature_aggregator_returns_singleton(self) -> None:
        """Test that get_literature_aggregator returns a singleton."""
        with patch("aria.api.dependencies.LiteratureAggregator") as mock_aggregator:
            mock_aggregator.return_value = MagicMock()

            from aria.api.dependencies import get_literature_aggregator

            get_literature_aggregator.cache_clear()

            result1 = get_literature_aggregator()
            result2 = get_literature_aggregator()

            assert result1 is result2
            mock_aggregator.assert_called_once()

    def test_get_literature_qa_chain_returns_singleton(self) -> None:
        """Test that get_literature_qa_chain returns a singleton."""
        with (
            patch("aria.api.dependencies.LiteratureQAChain") as mock_chain,
            patch("aria.api.dependencies.get_rag_pipeline") as mock_get_pipeline,
            patch("aria.api.dependencies.get_literature_aggregator") as mock_get_aggregator,
        ):
            mock_chain.return_value = MagicMock()
            mock_get_pipeline.return_value = MagicMock()
            mock_get_aggregator.return_value = MagicMock()

            from aria.api.dependencies import get_literature_qa_chain

            get_literature_qa_chain.cache_clear()

            result1 = get_literature_qa_chain()
            result2 = get_literature_qa_chain()

            assert result1 is result2
            mock_chain.assert_called_once()


class TestDbSession:
    """Tests for database session dependency."""

    @pytest.mark.asyncio
    async def test_get_db_session_yields_session(self) -> None:
        """Test that get_db_session yields a session."""
        mock_session = MagicMock()

        async def mock_get_async_session():
            yield mock_session

        with patch(
            "aria.api.dependencies.get_async_session",
            return_value=mock_get_async_session(),
        ):
            from aria.api.dependencies import get_db_session

            async for session in get_db_session():
                assert session is mock_session


class TestDependencyTypes:
    """Tests for dependency type aliases."""

    def test_db_session_type_alias_exists(self) -> None:
        """Test that DBSession type alias is defined."""
        from aria.api.dependencies import DBSession

        assert DBSession is not None

    def test_rag_pipeline_dep_type_exists(self) -> None:
        """Test that RAGPipelineDep type alias is defined."""
        from aria.api.dependencies import RAGPipelineDep

        assert RAGPipelineDep is not None

    def test_embedder_dep_type_exists(self) -> None:
        """Test that EmbedderDep type alias is defined."""
        from aria.api.dependencies import EmbedderDep

        assert EmbedderDep is not None

    def test_vector_store_dep_type_exists(self) -> None:
        """Test that VectorStoreDep type alias is defined."""
        from aria.api.dependencies import VectorStoreDep

        assert VectorStoreDep is not None

    def test_literature_aggregator_dep_type_exists(self) -> None:
        """Test that LiteratureAggregatorDep type alias is defined."""
        from aria.api.dependencies import LiteratureAggregatorDep

        assert LiteratureAggregatorDep is not None

    def test_literature_qa_chain_dep_type_exists(self) -> None:
        """Test that LiteratureQAChainDep type alias is defined."""
        from aria.api.dependencies import LiteratureQAChainDep

        assert LiteratureQAChainDep is not None
