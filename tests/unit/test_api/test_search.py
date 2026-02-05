"""Unit tests for search endpoints and models."""

import pytest
from pydantic import ValidationError


class TestSearchFilters:
    """Tests for SearchFilters model."""

    def test_search_filters_empty(self) -> None:
        """Test creating empty SearchFilters."""
        from aria.api.routes.search import SearchFilters

        filters = SearchFilters()
        assert filters.year_from is None
        assert filters.year_to is None
        assert filters.journals is None
        assert filters.authors is None
        assert filters.document_types is None

    def test_search_filters_with_years(self) -> None:
        """Test SearchFilters with year range."""
        from aria.api.routes.search import SearchFilters

        filters = SearchFilters(year_from=2020, year_to=2024)
        assert filters.year_from == 2020
        assert filters.year_to == 2024

    def test_search_filters_with_journals(self) -> None:
        """Test SearchFilters with journal filter."""
        from aria.api.routes.search import SearchFilters

        filters = SearchFilters(journals=["Nature", "Science"])
        assert len(filters.journals) == 2

    def test_search_filters_with_all_fields(self) -> None:
        """Test SearchFilters with all fields."""
        from aria.api.routes.search import SearchFilters

        filters = SearchFilters(
            year_from=2020,
            year_to=2024,
            journals=["Nature"],
            authors=["Smith", "Doe"],
            document_types=["paper", "patent"],
        )
        assert filters.authors == ["Smith", "Doe"]
        assert filters.document_types == ["paper", "patent"]


class TestLiteratureSearchRequest:
    """Tests for LiteratureSearchRequest model."""

    def test_minimal_search_request(self) -> None:
        """Test minimal LiteratureSearchRequest."""
        from aria.api.routes.search import LiteratureSearchRequest

        request = LiteratureSearchRequest(query="cancer treatment")
        assert request.query == "cancer treatment"
        assert request.limit == 20
        assert request.offset == 0
        assert "pubmed" in request.sources

    def test_search_request_with_filters(self) -> None:
        """Test LiteratureSearchRequest with filters."""
        from aria.api.routes.search import LiteratureSearchRequest, SearchFilters

        request = LiteratureSearchRequest(
            query="machine learning",
            filters=SearchFilters(year_from=2023),
            limit=50,
            sources=["pubmed"],
        )
        assert request.filters.year_from == 2023
        assert request.limit == 50
        assert request.sources == ["pubmed"]

    def test_search_request_query_validation(self) -> None:
        """Test that empty query is rejected."""
        from aria.api.routes.search import LiteratureSearchRequest

        with pytest.raises(ValidationError):
            LiteratureSearchRequest(query="")

    def test_search_request_limit_bounds(self) -> None:
        """Test limit validation bounds."""
        from aria.api.routes.search import LiteratureSearchRequest

        # Valid limit
        request = LiteratureSearchRequest(query="test", limit=100)
        assert request.limit == 100

        # Invalid: over 100
        with pytest.raises(ValidationError):
            LiteratureSearchRequest(query="test", limit=101)

        # Invalid: less than 1
        with pytest.raises(ValidationError):
            LiteratureSearchRequest(query="test", limit=0)


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_minimal_search_result(self) -> None:
        """Test minimal SearchResult."""
        from aria.api.routes.search import SearchResult

        result = SearchResult(
            id="doc-123",
            title="Test Paper",
            source="pubmed",
            relevance_score=0.85,
        )
        assert result.id == "doc-123"
        assert result.authors == []
        assert result.abstract is None

    def test_full_search_result(self) -> None:
        """Test full SearchResult."""
        from aria.api.routes.search import SearchResult

        result = SearchResult(
            id="pmid:12345678",
            title="Important Discovery",
            abstract="We found something interesting.",
            authors=["Smith, J.", "Doe, J."],
            year=2024,
            journal="Nature",
            doi="10.1038/nature12345",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678",
            source="pubmed",
            relevance_score=0.92,
        )
        assert len(result.authors) == 2
        assert result.year == 2024

    def test_search_result_score_validation(self) -> None:
        """Test relevance_score validation."""
        from aria.api.routes.search import SearchResult

        # Invalid: > 1
        with pytest.raises(ValidationError):
            SearchResult(
                id="test",
                title="Test",
                source="internal",
                relevance_score=1.5,
            )


class TestLiteratureSearchResponse:
    """Tests for LiteratureSearchResponse model."""

    def test_empty_response(self) -> None:
        """Test empty LiteratureSearchResponse."""
        from aria.api.routes.search import LiteratureSearchResponse

        response = LiteratureSearchResponse(
            query="test query",
            total_results=0,
            results=[],
        )
        assert response.query == "test query"
        assert response.total_results == 0
        assert response.metadata == {}

    def test_response_with_results(self) -> None:
        """Test LiteratureSearchResponse with results."""
        from aria.api.routes.search import LiteratureSearchResponse, SearchResult

        results = [
            SearchResult(
                id="1",
                title="Paper 1",
                source="pubmed",
                relevance_score=0.9,
            ),
            SearchResult(
                id="2",
                title="Paper 2",
                source="internal",
                relevance_score=0.8,
            ),
        ]
        response = LiteratureSearchResponse(
            query="test",
            total_results=100,
            results=results,
            metadata={"search_time_ms": 150},
        )
        assert len(response.results) == 2
        assert response.total_results == 100


class TestMolecularSearchRequest:
    """Tests for MolecularSearchRequest model."""

    def test_smiles_search_request(self) -> None:
        """Test MolecularSearchRequest with SMILES."""
        from aria.api.routes.search import MolecularSearchRequest

        request = MolecularSearchRequest(
            smiles="CCO",  # Ethanol
            similarity_threshold=0.8,
        )
        assert request.smiles == "CCO"
        assert request.similarity_threshold == 0.8
        assert request.limit == 20

    def test_name_search_request(self) -> None:
        """Test MolecularSearchRequest with name."""
        from aria.api.routes.search import MolecularSearchRequest

        request = MolecularSearchRequest(
            name="aspirin",
            limit=10,
        )
        assert request.name == "aspirin"
        assert request.smiles is None

    def test_similarity_threshold_validation(self) -> None:
        """Test similarity_threshold validation."""
        from aria.api.routes.search import MolecularSearchRequest

        # Valid threshold
        request = MolecularSearchRequest(smiles="C", similarity_threshold=0.5)
        assert request.similarity_threshold == 0.5

        # Invalid: > 1
        with pytest.raises(ValidationError):
            MolecularSearchRequest(smiles="C", similarity_threshold=1.5)


class TestMoleculeResult:
    """Tests for MoleculeResult model."""

    def test_molecule_result_creation(self) -> None:
        """Test MoleculeResult creation."""
        from aria.api.routes.search import MoleculeResult

        result = MoleculeResult(
            id="mol-123",
            name="Ethanol",
            smiles="CCO",
            similarity=0.95,
            properties={"molecular_weight": 46.07, "logP": -0.31},
        )
        assert result.name == "Ethanol"
        assert result.properties["molecular_weight"] == 46.07


class TestMolecularSearchResponse:
    """Tests for MolecularSearchResponse model."""

    def test_molecular_response_creation(self) -> None:
        """Test MolecularSearchResponse creation."""
        from aria.api.routes.search import MolecularSearchResponse, MoleculeResult

        results = [
            MoleculeResult(
                id="mol-1",
                name="Ethanol",
                smiles="CCO",
                similarity=0.95,
            ),
        ]
        response = MolecularSearchResponse(
            query_smiles="CCO",
            results=results,
        )
        assert response.query_smiles == "CCO"
        assert len(response.results) == 1


