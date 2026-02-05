"""Unit tests for LLM chains."""

from unittest.mock import MagicMock, patch

import pytest

from aria.llm.chains.literature_qa import (
    LiteratureQAChain,
    LiteratureQAResult,
    LiteratureQAState,
)


class TestLiteratureQAState:
    """Tests for LiteratureQAState TypedDict."""

    def test_state_creation(self) -> None:
        """Test creating LiteratureQAState."""
        state: LiteratureQAState = {
            "query": "What is CRISPR?",
            "query_type": "factual",
            "internal_results": [],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        assert state["query"] == "What is CRISPR?"
        assert state["query_type"] == "factual"
        assert state["error"] is None


class TestLiteratureQAResult:
    """Tests for LiteratureQAResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating LiteratureQAResult."""
        result = LiteratureQAResult(
            answer="CRISPR is a gene editing technology.",
            citations=[],
            confidence=0.85,
            sources_used=3,
            internal_count=2,
            external_count=1,
        )

        assert result.answer == "CRISPR is a gene editing technology."
        assert result.confidence == 0.85
        assert result.sources_used == 3
        assert result.internal_count == 2
        assert result.external_count == 1

    def test_result_with_metadata(self) -> None:
        """Test LiteratureQAResult with metadata."""
        result = LiteratureQAResult(
            answer="Answer",
            citations=[],
            confidence=0.9,
            sources_used=0,
            internal_count=0,
            external_count=0,
            metadata={"latency_ms": 500, "model": "claude-3"},
        )

        assert result.metadata["latency_ms"] == 500
        assert result.metadata["model"] == "claude-3"

    def test_result_default_metadata(self) -> None:
        """Test that metadata defaults to empty dict."""
        result = LiteratureQAResult(
            answer="Answer",
            citations=[],
            confidence=0.9,
            sources_used=0,
            internal_count=0,
            external_count=0,
        )

        assert result.metadata == {}


class TestLiteratureQAChainInit:
    """Tests for LiteratureQAChain initialization."""

    def test_chain_init_defaults(self) -> None:
        """Test chain initialization with defaults."""
        with (
            patch("aria.llm.chains.literature_qa.RAGPipeline") as mock_rag,
            patch("aria.llm.chains.literature_qa.LiteratureAggregator") as mock_agg,
        ):
            mock_rag.return_value = MagicMock()
            mock_agg.return_value = MagicMock()

            chain = LiteratureQAChain()

            assert chain.include_external is True
            assert chain.rag_pipeline is not None
            assert chain.literature_aggregator is not None
            assert chain.graph is not None

    def test_chain_init_with_external_disabled(self) -> None:
        """Test chain initialization with external search disabled."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
            include_external=False,
        )

        assert chain.include_external is False


class TestLiteratureQAChainClassifyQuery:
    """Tests for _classify_query method."""

    @pytest.mark.asyncio
    async def test_classify_comparative_query(self) -> None:
        """Test classifying comparative query."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
        )

        state: LiteratureQAState = {
            "query": "Compare CRISPR versus traditional gene therapy",
            "query_type": "",
            "internal_results": [],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        result = await chain._classify_query(state)

        assert result["query_type"] == "comparative"

    @pytest.mark.asyncio
    async def test_classify_factual_query(self) -> None:
        """Test classifying factual query."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
        )

        state: LiteratureQAState = {
            "query": "What is the mechanism of CRISPR?",
            "query_type": "",
            "internal_results": [],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        result = await chain._classify_query(state)

        assert result["query_type"] == "factual"

    @pytest.mark.asyncio
    async def test_classify_exploratory_query(self) -> None:
        """Test classifying exploratory query."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
        )

        state: LiteratureQAState = {
            "query": "Recent advances in cancer immunotherapy",
            "query_type": "",
            "internal_results": [],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        result = await chain._classify_query(state)

        assert result["query_type"] == "exploratory"


class TestLiteratureQAChainShouldSearchExternal:
    """Tests for _should_search_external method."""

    def test_should_search_external_disabled(self) -> None:
        """Test that external search is skipped when disabled."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
            include_external=False,
        )

        state: LiteratureQAState = {
            "query": "Test query",
            "query_type": "factual",
            "internal_results": [],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        result = chain._should_search_external(state)

        assert result == "synthesize"

    def test_should_search_external_insufficient_internal(self) -> None:
        """Test that external search happens when internal results are insufficient."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
            include_external=True,
        )

        state: LiteratureQAState = {
            "query": "Test query",
            "query_type": "factual",
            "internal_results": [{"content": "result 1"}, {"content": "result 2"}],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        result = chain._should_search_external(state)

        assert result == "search_external"

    def test_should_search_external_for_exploratory(self) -> None:
        """Test that external search happens for exploratory queries."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
            include_external=True,
        )

        state: LiteratureQAState = {
            "query": "Test query",
            "query_type": "exploratory",
            "internal_results": [
                {"content": "result 1"},
                {"content": "result 2"},
                {"content": "result 3"},
                {"content": "result 4"},
            ],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        result = chain._should_search_external(state)

        assert result == "search_external"

    def test_should_skip_external_sufficient_internal(self) -> None:
        """Test that external search is skipped when internal results are sufficient."""
        mock_rag = MagicMock()
        mock_agg = MagicMock()

        chain = LiteratureQAChain(
            rag_pipeline=mock_rag,
            literature_aggregator=mock_agg,
            include_external=True,
        )

        state: LiteratureQAState = {
            "query": "Test query",
            "query_type": "factual",
            "internal_results": [
                {"content": "result 1"},
                {"content": "result 2"},
                {"content": "result 3"},
            ],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        result = chain._should_search_external(state)

        assert result == "synthesize"
