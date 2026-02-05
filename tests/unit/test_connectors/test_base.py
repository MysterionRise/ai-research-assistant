"""Unit tests for connector base classes."""

from typing import Any

import pytest

from aria.connectors.base import BaseConnector, LiteratureResult


class TestLiteratureResult:
    """Tests for LiteratureResult dataclass."""

    def test_create_literature_result_minimal(self) -> None:
        """Test creating LiteratureResult with required fields."""
        result = LiteratureResult(
            id="pmid:12345678",
            title="A Novel Study on Materials Science",
        )

        assert result.id == "pmid:12345678"
        assert result.title == "A Novel Study on Materials Science"
        assert result.abstract is None
        assert result.authors == []
        assert result.source == ""
        assert result.score == 0.0

    def test_create_literature_result_full(self) -> None:
        """Test creating LiteratureResult with all fields."""
        result = LiteratureResult(
            id="arxiv:2301.12345",
            title="Machine Learning for Drug Discovery",
            abstract="This paper presents a novel approach to...",
            authors=["John Doe", "Jane Smith", "Bob Wilson"],
            year=2024,
            journal="Nature Machine Intelligence",
            doi="10.1038/s42256-024-00123",
            url="https://arxiv.org/abs/2301.12345",
            source="arxiv",
            score=0.95,
            metadata={"citations": 42, "impact_factor": 25.5},
        )

        assert result.id == "arxiv:2301.12345"
        assert result.title == "Machine Learning for Drug Discovery"
        assert result.abstract == "This paper presents a novel approach to..."
        assert len(result.authors) == 3
        assert result.year == 2024
        assert result.journal == "Nature Machine Intelligence"
        assert result.doi == "10.1038/s42256-024-00123"
        assert result.url == "https://arxiv.org/abs/2301.12345"
        assert result.source == "arxiv"
        assert result.score == 0.95
        assert result.metadata["citations"] == 42

    def test_literature_result_default_authors(self) -> None:
        """Test that authors defaults to empty list."""
        result = LiteratureResult(id="test", title="Test")
        assert result.authors == []
        assert isinstance(result.authors, list)

    def test_literature_result_default_metadata(self) -> None:
        """Test that metadata defaults to empty dict."""
        result = LiteratureResult(id="test", title="Test")
        assert result.metadata == {}
        assert isinstance(result.metadata, dict)

    def test_literature_result_score_range(self) -> None:
        """Test that score can be any float."""
        result_zero = LiteratureResult(id="1", title="Test", score=0.0)
        assert result_zero.score == 0.0

        result_one = LiteratureResult(id="2", title="Test", score=1.0)
        assert result_one.score == 1.0

        result_mid = LiteratureResult(id="3", title="Test", score=0.567)
        assert result_mid.score == 0.567


class TestBaseConnector:
    """Tests for BaseConnector abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseConnector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseConnector()

    def test_subclass_must_implement_source_name(self) -> None:
        """Test that subclass must implement source_name property."""

        class IncompleteConnector(BaseConnector):
            async def search(
                self,
                query: str,
                limit: int = 10,
                **kwargs: Any,
            ) -> list[LiteratureResult]:
                return []

            async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
                return None

        with pytest.raises(TypeError):
            IncompleteConnector()

    def test_subclass_must_implement_search(self) -> None:
        """Test that subclass must implement search method."""

        class IncompleteConnector(BaseConnector):
            @property
            def source_name(self) -> str:
                return "test"

            async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
                return None

        with pytest.raises(TypeError):
            IncompleteConnector()

    def test_subclass_must_implement_get_by_id(self) -> None:
        """Test that subclass must implement get_by_id method."""

        class IncompleteConnector(BaseConnector):
            @property
            def source_name(self) -> str:
                return "test"

            async def search(
                self,
                query: str,
                limit: int = 10,
                **kwargs: Any,
            ) -> list[LiteratureResult]:
                return []

        with pytest.raises(TypeError):
            IncompleteConnector()

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that complete implementation can be instantiated."""

        class CompleteConnector(BaseConnector):
            @property
            def source_name(self) -> str:
                return "test_source"

            async def search(
                self,
                query: str,
                limit: int = 10,
                **kwargs: Any,
            ) -> list[LiteratureResult]:
                return []

            async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
                return None

        connector = CompleteConnector()
        assert isinstance(connector, BaseConnector)
        assert connector.source_name == "test_source"


class TestMockConnector:
    """Tests using a mock connector implementation."""

    class MockConnector(BaseConnector):
        """Mock implementation for testing."""

        def __init__(self) -> None:
            self._papers: dict[str, LiteratureResult] = {}

        @property
        def source_name(self) -> str:
            return "mock_source"

        def add_paper(self, paper: LiteratureResult) -> None:
            """Add a paper to the mock database."""
            self._papers[paper.id] = paper

        async def search(
            self,
            query: str,
            limit: int = 10,
            **kwargs: Any,
        ) -> list[LiteratureResult]:
            # Simple mock: return papers that have query in title
            results = [
                paper
                for paper in self._papers.values()
                if query.lower() in paper.title.lower()
            ]
            return results[:limit]

        async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
            return self._papers.get(paper_id)

    @pytest.fixture
    def connector(self) -> "TestMockConnector.MockConnector":
        """Create mock connector."""
        return self.MockConnector()

    @pytest.mark.asyncio
    async def test_search_returns_matching_papers(
        self, connector: "TestMockConnector.MockConnector"
    ) -> None:
        """Test that search returns papers matching query."""
        connector.add_paper(
            LiteratureResult(
                id="1",
                title="Machine Learning in Healthcare",
                source="mock_source",
            )
        )
        connector.add_paper(
            LiteratureResult(
                id="2",
                title="Deep Learning for Image Analysis",
                source="mock_source",
            )
        )

        results = await connector.search("Machine Learning")

        assert len(results) == 1
        assert results[0].id == "1"

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self, connector: "TestMockConnector.MockConnector"
    ) -> None:
        """Test that search respects limit parameter."""
        for i in range(10):
            connector.add_paper(
                LiteratureResult(
                    id=str(i),
                    title=f"Paper {i} about Science",
                    source="mock_source",
                )
            )

        results = await connector.search("Science", limit=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(
        self, connector: "TestMockConnector.MockConnector"
    ) -> None:
        """Test that search returns empty list for no matches."""
        connector.add_paper(
            LiteratureResult(id="1", title="Biology Paper", source="mock_source")
        )

        results = await connector.search("Quantum Physics")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_by_id_returns_paper(
        self, connector: "TestMockConnector.MockConnector"
    ) -> None:
        """Test that get_by_id returns the correct paper."""
        paper = LiteratureResult(
            id="test-123",
            title="Specific Paper",
            abstract="Abstract text",
            source="mock_source",
        )
        connector.add_paper(paper)

        result = await connector.get_by_id("test-123")

        assert result is not None
        assert result.id == "test-123"
        assert result.title == "Specific Paper"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_for_unknown(
        self, connector: "TestMockConnector.MockConnector"
    ) -> None:
        """Test that get_by_id returns None for unknown ID."""
        result = await connector.get_by_id("unknown-id")
        assert result is None

    def test_source_name_property(
        self, connector: "TestMockConnector.MockConnector"
    ) -> None:
        """Test that source_name property works correctly."""
        assert connector.source_name == "mock_source"
