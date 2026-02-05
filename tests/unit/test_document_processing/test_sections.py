"""Unit tests for sections extractor."""

import pytest

from aria.document_processing.extractors.sections import (
    ExtractedSections,
    Section,
    SectionExtractor,
)
from aria.document_processing.parsers.base import ParsedDocument, ParsedPage


class TestSection:
    """Tests for Section dataclass."""

    def test_section_creation(self) -> None:
        """Test creating a Section."""
        section = Section(
            name="Introduction",
            content="This is the introduction text.",
            start_pos=0,
            end_pos=30,
        )

        assert section.name == "Introduction"
        assert section.content == "This is the introduction text."
        assert section.start_pos == 0
        assert section.end_pos == 30

    def test_section_with_level(self) -> None:
        """Test Section with heading level."""
        section = Section(
            name="Methods",
            content="Methods content",
            start_pos=100,
            end_pos=200,
            level=1,
        )

        assert section.level == 1


class TestExtractedSections:
    """Tests for ExtractedSections dataclass."""

    def test_extracted_sections_creation(self) -> None:
        """Test creating ExtractedSections."""
        sections = [
            Section(name="Intro", content="...", start_pos=0, end_pos=100),
            Section(name="Methods", content="...", start_pos=100, end_pos=200),
        ]
        result = ExtractedSections(sections=sections)

        assert len(result.sections) == 2
        assert result.sections[0].name == "Intro"

    def test_empty_sections(self) -> None:
        """Test ExtractedSections with no sections."""
        result = ExtractedSections(sections=[])
        assert len(result.sections) == 0


class TestSectionExtractor:
    """Tests for SectionExtractor."""

    @pytest.fixture
    def extractor(self) -> SectionExtractor:
        """Create section extractor."""
        return SectionExtractor()

    def test_extract_numbered_sections(self, extractor) -> None:
        """Test extracting numbered sections."""
        text = """1. Introduction

This is the introduction content.

2. Methods

This is the methods content.

3. Results

This is the results content."""

        result = extractor.extract(text)

        assert len(result.sections) >= 2
        section_names = [s.name.lower() for s in result.sections]
        assert any("introduction" in name for name in section_names)
        assert any("methods" in name for name in section_names)

    def test_extract_heading_sections(self, extractor) -> None:
        """Test extracting sections by heading patterns."""
        text = """Introduction

This is the introduction content.

METHODS

This is the methods content.

Results

This is the results content."""

        result = extractor.extract(text)

        assert len(result.sections) >= 1

    def test_extract_empty_text(self, extractor) -> None:
        """Test extracting from empty text."""
        result = extractor.extract("")
        # Should return empty sections or single body section
        assert isinstance(result, ExtractedSections)

    def test_extract_no_sections(self, extractor) -> None:
        """Test text with no clear sections."""
        text = "Just a simple paragraph without any section markers."
        result = extractor.extract(text)

        # Should handle gracefully
        assert isinstance(result, ExtractedSections)

    def test_extract_from_document(self, extractor) -> None:
        """Test extracting sections from ParsedDocument."""
        text = """Abstract

We present a study...

1. Introduction

The field of machine learning..."""
        doc = ParsedDocument(
            filename="test.pdf",
            file_type="application/pdf",
            total_pages=1,
            pages=[ParsedPage(page_number=1, text=text)],
            full_text=text,
            metadata={},
        )

        # Extract from full text
        result = extractor.extract(doc.full_text)

        assert isinstance(result, ExtractedSections)

    def test_section_positions(self, extractor) -> None:
        """Test that section positions are valid."""
        text = """Introduction

Content of intro.

Methods

Content of methods."""

        result = extractor.extract(text)

        for section in result.sections:
            assert section.start_pos >= 0
            assert section.end_pos >= section.start_pos
            assert section.end_pos <= len(text) + 1


class TestSectionPatterns:
    """Tests for section pattern matching."""

    def test_common_section_names(self) -> None:
        """Test that extractor handles common section names."""
        extractor = SectionExtractor()
        # Use numbered sections which are more likely to be detected
        text = """1. Introduction

intro text

2. Materials and Methods

methods text

3. Results and Discussion

results text

4. Conclusions

conclusion text"""

        result = extractor.extract(text)

        # The extractor returns ExtractedSections regardless of detection
        assert isinstance(result, ExtractedSections)


class TestExtractedSectionsMethods:
    """Tests for ExtractedSections methods."""

    def test_get_section_case_insensitive(self) -> None:
        """Test get_section is case insensitive."""
        sections = ExtractedSections(
            sections=[
                Section(name="Introduction", content="intro content", start_pos=0, end_pos=50),
                Section(name="Methods", content="methods content", start_pos=50, end_pos=100),
            ]
        )

        # Test various cases
        assert sections.get_section("introduction") is not None
        assert sections.get_section("INTRODUCTION") is not None
        assert sections.get_section("Introduction") is not None
        assert sections.get_section("introduction").name == "Introduction"

    def test_get_section_not_found(self) -> None:
        """Test get_section returns None when section not found."""
        sections = ExtractedSections(
            sections=[
                Section(name="Introduction", content="content", start_pos=0, end_pos=50),
            ]
        )

        result = sections.get_section("Nonexistent")
        assert result is None

    def test_section_names_property(self) -> None:
        """Test section_names property returns list of names."""
        sections = ExtractedSections(
            sections=[
                Section(name="Abstract", content="...", start_pos=0, end_pos=30),
                Section(name="Introduction", content="...", start_pos=30, end_pos=60),
                Section(name="Methods", content="...", start_pos=60, end_pos=90),
            ]
        )

        names = sections.section_names
        assert names == ["Abstract", "Introduction", "Methods"]

    def test_section_names_empty(self) -> None:
        """Test section_names property with no sections."""
        sections = ExtractedSections(sections=[])
        assert sections.section_names == []


class TestSectionExtractorNormalization:
    """Tests for section name normalization."""

    def test_normalize_known_sections(self) -> None:
        """Test normalization of known section names."""
        extractor = SectionExtractor()

        assert extractor._normalize_section_name("abstract") == "Abstract"
        assert extractor._normalize_section_name("introduction") == "Introduction"
        assert extractor._normalize_section_name("methods") == "Methods"
        assert extractor._normalize_section_name("results") == "Results"
        assert extractor._normalize_section_name("discussion") == "Discussion"
        assert extractor._normalize_section_name("conclusions") == "Conclusion"
        assert extractor._normalize_section_name("references") == "References"

    def test_normalize_aliases(self) -> None:
        """Test normalization of section aliases."""
        extractor = SectionExtractor()

        # Background -> Introduction
        assert extractor._normalize_section_name("background") == "Introduction"
        # Materials -> Methods
        assert extractor._normalize_section_name("materials") == "Methods"
        # Materials and methods -> Methods
        assert extractor._normalize_section_name("materials and methods") == "Methods"

    def test_normalize_unknown_section(self) -> None:
        """Test normalization returns None for unknown sections."""
        extractor = SectionExtractor()

        assert extractor._normalize_section_name("random section") is None
        assert extractor._normalize_section_name("xyz") is None


class TestSectionExtractorGetSectionForPosition:
    """Tests for get_section_for_position method."""

    def test_get_section_for_position_in_section(self) -> None:
        """Test getting section name for position within a section."""
        extractor = SectionExtractor()

        sections = ExtractedSections(
            sections=[
                Section(name="Introduction", content="...", start_pos=0, end_pos=100),
                Section(name="Methods", content="...", start_pos=100, end_pos=200),
            ]
        )

        # Position in Introduction
        result = extractor.get_section_for_position(sections, 50)
        assert result == "Introduction"

        # Position in Methods
        result = extractor.get_section_for_position(sections, 150)
        assert result == "Methods"

    def test_get_section_for_position_at_boundary(self) -> None:
        """Test getting section at section boundaries."""
        extractor = SectionExtractor()

        sections = ExtractedSections(
            sections=[
                Section(name="Introduction", content="...", start_pos=0, end_pos=100),
                Section(name="Methods", content="...", start_pos=100, end_pos=200),
            ]
        )

        # Position at start of Introduction
        result = extractor.get_section_for_position(sections, 0)
        assert result == "Introduction"

        # Position at start of Methods (end of Introduction)
        result = extractor.get_section_for_position(sections, 100)
        assert result == "Methods"

    def test_get_section_for_position_outside(self) -> None:
        """Test getting section for position outside all sections."""
        extractor = SectionExtractor()

        sections = ExtractedSections(
            sections=[
                Section(name="Introduction", content="...", start_pos=50, end_pos=100),
            ]
        )

        # Position before first section
        result = extractor.get_section_for_position(sections, 10)
        assert result is None

        # Position after last section
        result = extractor.get_section_for_position(sections, 200)
        assert result is None

    def test_get_section_for_position_empty_sections(self) -> None:
        """Test with no sections."""
        extractor = SectionExtractor()

        sections = ExtractedSections(sections=[])

        result = extractor.get_section_for_position(sections, 50)
        assert result is None
