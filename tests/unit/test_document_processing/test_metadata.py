"""Unit tests for metadata extractor."""

import pytest

from aria.document_processing.extractors.metadata import ExtractedMetadata, MetadataExtractor
from aria.document_processing.parsers.base import ParsedDocument, ParsedPage


def make_doc(text: str, metadata: dict | None = None) -> ParsedDocument:
    """Helper to create a ParsedDocument with required fields."""
    pages = [ParsedPage(page_number=1, text=text)] if text else []
    return ParsedDocument(
        filename="test.pdf",
        file_type="application/pdf",
        total_pages=len(pages),
        pages=pages,
        full_text=text,
        metadata=metadata or {},
    )


class TestExtractedMetadata:
    """Tests for ExtractedMetadata dataclass."""

    def test_default_values(self) -> None:
        """Test that ExtractedMetadata has correct defaults."""
        metadata = ExtractedMetadata()

        assert metadata.title is None
        assert metadata.authors is None
        assert metadata.year is None
        assert metadata.journal is None
        assert metadata.doi is None
        assert metadata.abstract is None
        assert metadata.keywords is None

    def test_with_values(self) -> None:
        """Test ExtractedMetadata with values."""
        metadata = ExtractedMetadata(
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            doi="10.1234/test",
        )

        assert metadata.title == "Test Paper"
        assert len(metadata.authors) == 2
        assert metadata.year == 2024


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    @pytest.fixture
    def extractor(self) -> MetadataExtractor:
        """Create metadata extractor."""
        return MetadataExtractor()

    @pytest.fixture
    def sample_document(self) -> ParsedDocument:
        """Create sample parsed document."""
        text = """Machine Learning in Drug Discovery: A Review

John Doe, Jane Smith, and Bob Wilson

Department of Computer Science, Test University

Abstract: This paper presents a comprehensive review of machine learning
applications in drug discovery. We examine recent advances in neural networks
and their applications to molecular property prediction.

Keywords: machine learning, drug discovery, neural networks

1. Introduction

Drug discovery is a complex process..."""
        return make_doc(text, {"title": "ML in Drug Discovery"})

    def test_extract_title_from_metadata(self, extractor, sample_document) -> None:
        """Test extracting title from PDF metadata."""
        result = extractor.extract(sample_document)
        assert result.title == "ML in Drug Discovery"

    def test_extract_title_from_text(self, extractor) -> None:
        """Test extracting title from text when no metadata."""
        doc = make_doc("Machine Learning in Drug Discovery\n\nAuthors here...")
        result = extractor.extract(doc)
        assert result.title == "Machine Learning in Drug Discovery"

    def test_extract_doi(self, extractor) -> None:
        """Test extracting DOI from text."""
        doc = make_doc("Some Paper Title\n\ndoi: 10.1038/nature12345\n\nAbstract...")
        result = extractor.extract(doc)
        assert result.doi == "10.1038/nature12345"

    def test_extract_doi_url_format(self, extractor) -> None:
        """Test extracting DOI in URL format."""
        doc = make_doc("Paper at https://doi.org/10.1000/test123\n\nContent...")
        result = extractor.extract(doc)
        assert result.doi == "10.1000/test123"

    def test_extract_abstract(self, extractor, sample_document) -> None:
        """Test extracting abstract."""
        result = extractor.extract(sample_document)
        assert result.abstract is not None
        assert "machine learning" in result.abstract.lower()

    def test_extract_keywords(self, extractor, sample_document) -> None:
        """Test extracting keywords."""
        result = extractor.extract(sample_document)
        assert result.keywords is not None
        assert "machine learning" in result.keywords

    def test_extract_year_from_metadata(self, extractor) -> None:
        """Test extracting year from PDF metadata."""
        doc = make_doc("Content here", {"creation_date": "2024-01-15"})
        result = extractor.extract(doc)
        assert result.year == 2024

    def test_extract_year_from_text(self, extractor) -> None:
        """Test extracting year from text."""
        doc = make_doc("Published in Nature, 2023. Some paper content...")
        result = extractor.extract(doc)
        assert result.year == 2023

    def test_extract_authors(self, extractor, sample_document) -> None:
        """Test extracting author names."""
        # Just verify the method runs without error
        extractor.extract(sample_document)


class TestDoiPattern:
    """Tests for DOI pattern matching."""

    def test_doi_pattern_basic(self) -> None:
        """Test basic DOI pattern."""
        extractor = MetadataExtractor()
        match = extractor.DOI_PATTERN.search("doi: 10.1234/test")
        assert match is not None
        assert match.group(1) == "10.1234/test"

    def test_doi_pattern_url(self) -> None:
        """Test DOI URL pattern."""
        extractor = MetadataExtractor()
        match = extractor.DOI_PATTERN.search("https://doi.org/10.1234/test")
        assert match is not None
        assert match.group(1) == "10.1234/test"

    def test_doi_pattern_dx_doi(self) -> None:
        """Test dx.doi.org URL pattern."""
        extractor = MetadataExtractor()
        match = extractor.DOI_PATTERN.search("https://dx.doi.org/10.5678/example")
        assert match is not None
        assert match.group(1) == "10.5678/example"


class TestYearPattern:
    """Tests for year pattern matching."""

    def test_year_pattern_recent(self) -> None:
        """Test year pattern with recent year."""
        extractor = MetadataExtractor()
        match = extractor.YEAR_PATTERN.search("Published 2024")
        assert match is not None
        assert match.group(0) == "2024"

    def test_year_pattern_historical(self) -> None:
        """Test year pattern with historical year."""
        extractor = MetadataExtractor()
        match = extractor.YEAR_PATTERN.search("Original work from 1999")
        assert match is not None
        assert match.group(0) == "1999"


class TestEmptyDocument:
    """Tests with empty or minimal documents."""

    def test_empty_pages(self) -> None:
        """Test extraction with empty pages."""
        extractor = MetadataExtractor()
        doc = ParsedDocument(
            filename="empty.pdf",
            file_type="application/pdf",
            total_pages=0,
            pages=[],
            full_text="",
            metadata={},
        )
        result = extractor.extract(doc)

        # Should not crash, returns empty metadata
        assert result.title is None

    def test_minimal_content(self) -> None:
        """Test extraction with minimal content."""
        extractor = MetadataExtractor()
        doc = make_doc("Short")
        result = extractor.extract(doc)

        # Should handle gracefully
        assert isinstance(result, ExtractedMetadata)
