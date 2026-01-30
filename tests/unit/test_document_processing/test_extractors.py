"""Tests for document extractors."""

import pytest

from aria.document_processing.extractors.metadata import MetadataExtractor
from aria.document_processing.extractors.sections import SectionExtractor


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    @pytest.fixture
    def extractor(self):
        return MetadataExtractor()

    def test_extract_doi(self, extractor):
        """Test DOI extraction."""
        text = "This paper discusses... doi: 10.1234/journal.123456"
        doi = extractor._extract_doi(text)

        assert doi == "10.1234/journal.123456"

    def test_extract_doi_with_url(self, extractor):
        """Test DOI extraction from URL format."""
        text = "Available at https://doi.org/10.1038/nature12373"
        doi = extractor._extract_doi(text)

        assert "10.1038/nature12373" in (doi or "")

    def test_extract_year(self, extractor):
        """Test year extraction."""
        text = "Published in 2023. Previous work from 2021 showed..."
        year = extractor._extract_year(text, {})

        # Should get most recent year
        assert year == 2023

    def test_extract_abstract(self, extractor):
        """Test abstract extraction."""
        text = """
        Title of the Paper

        Abstract: This study investigates the effects of treatment A on disease B.
        We found significant improvements in patient outcomes.

        Introduction
        """
        abstract = extractor._extract_abstract(text)

        assert abstract is not None
        assert "investigates" in abstract.lower()

    def test_extract_keywords(self, extractor):
        """Test keyword extraction."""
        text = """
        Keywords: machine learning, natural language processing, transformers, BERT
        """
        keywords = extractor._extract_keywords(text)

        assert keywords is not None
        assert len(keywords) >= 2
        assert "machine learning" in keywords or "transformers" in keywords


class TestSectionExtractor:
    """Tests for SectionExtractor."""

    @pytest.fixture
    def extractor(self):
        return SectionExtractor()

    def test_extract_numbered_sections(self, extractor):
        """Test extraction of numbered sections."""
        text = """
        1. Introduction
        This is the introduction content.

        2. Methods
        This describes the methods used.

        3. Results
        Here are the results.

        4. Discussion
        Discussion of findings.
        """

        result = extractor.extract(text)

        section_names = result.section_names
        assert "Introduction" in section_names
        assert "Methods" in section_names
        assert "Results" in section_names
        assert "Discussion" in section_names

    def test_extract_unnumbered_sections(self, extractor):
        """Test extraction of unnumbered sections."""
        text = """
        ABSTRACT
        This is the abstract content with enough text to be included.

        INTRODUCTION
        This is the introduction with sufficient content for detection.

        CONCLUSION
        Final conclusions drawn from this research study are presented.
        """

        result = extractor.extract(text)

        section_names = result.section_names
        assert "Abstract" in section_names
        assert "Introduction" in section_names
        assert "Conclusion" in section_names

    def test_get_section(self, extractor):
        """Test getting section by name."""
        text = """
        Introduction
        This is introduction content that is long enough to be detected.

        Methods
        These are the methods used in the study which are described here.
        """

        result = extractor.extract(text)
        intro = result.get_section("Introduction")

        assert intro is not None
        assert "content" in intro.content.lower() or "introduction" in intro.content.lower()

    def test_normalize_section_name(self, extractor):
        """Test section name normalization."""
        assert extractor._normalize_section_name("materials and methods") == "Methods"
        assert extractor._normalize_section_name("conclusions") == "Conclusion"
        assert extractor._normalize_section_name("acknowledgements") == "Acknowledgements"
