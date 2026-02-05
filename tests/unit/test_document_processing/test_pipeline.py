"""Unit tests for document processing pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aria.document_processing.pipeline import DocumentProcessingPipeline
from aria.exceptions import UnsupportedFileTypeError


class TestDocumentProcessingPipelineInit:
    """Tests for DocumentProcessingPipeline initialization."""

    def test_pipeline_init(self) -> None:
        """Test pipeline initialization."""
        pipeline = DocumentProcessingPipeline()

        assert pipeline.pdf_parser is not None
        assert pipeline.metadata_extractor is not None
        assert pipeline.section_extractor is not None

    def test_supported_mime_types(self) -> None:
        """Test that PDF is in supported MIME types."""
        pipeline = DocumentProcessingPipeline()

        assert "application/pdf" in pipeline.SUPPORTED_MIME_TYPES


class TestDocumentProcessingPipelineProcess:
    """Tests for DocumentProcessingPipeline.process method."""

    @pytest.mark.asyncio
    async def test_process_unsupported_file_type(self) -> None:
        """Test that unsupported file types raise error."""
        pipeline = DocumentProcessingPipeline()

        with pytest.raises(UnsupportedFileTypeError):
            await pipeline.process(
                file_path=Path("/fake/path/doc.docx"),
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    @pytest.mark.asyncio
    async def test_process_with_mocked_parser(self) -> None:
        """Test process with mocked PDF parser."""
        from aria.document_processing.extractors.metadata import ExtractedMetadata
        from aria.document_processing.extractors.sections import ExtractedSections
        from aria.document_processing.parsers.base import ParsedDocument, ParsedPage

        pipeline = DocumentProcessingPipeline()

        # Mock the PDF parser
        mock_parsed = ParsedDocument(
            filename="test.pdf",
            file_type="application/pdf",
            total_pages=1,
            pages=[ParsedPage(page_number=1, text="Sample scientific paper content.")],
            full_text="Sample scientific paper content.",
        )

        pipeline.pdf_parser.parse = AsyncMock(return_value=mock_parsed)

        parsed, metadata, sections = await pipeline.process(
            file_path=Path("/fake/test.pdf"),
            mime_type="application/pdf",
        )

        assert parsed.filename == "test.pdf"
        assert isinstance(metadata, ExtractedMetadata)
        assert isinstance(sections, ExtractedSections)


class TestDocumentProcessingPipelineParseDocument:
    """Tests for _parse_document method."""

    @pytest.mark.asyncio
    async def test_parse_document_pdf(self) -> None:
        """Test _parse_document with PDF mime type."""
        from aria.document_processing.parsers.base import ParsedDocument, ParsedPage

        pipeline = DocumentProcessingPipeline()

        mock_parsed = ParsedDocument(
            filename="test.pdf",
            file_type="application/pdf",
            total_pages=2,
            pages=[
                ParsedPage(page_number=1, text="Page 1"),
                ParsedPage(page_number=2, text="Page 2"),
            ],
            full_text="Page 1\n\nPage 2",
        )

        pipeline.pdf_parser.parse = AsyncMock(return_value=mock_parsed)

        result = await pipeline._parse_document(
            file_path=Path("/test.pdf"),
            mime_type="application/pdf",
        )

        assert result.total_pages == 2

    @pytest.mark.asyncio
    async def test_parse_document_unsupported(self) -> None:
        """Test _parse_document with unsupported type."""
        pipeline = DocumentProcessingPipeline()

        with pytest.raises(UnsupportedFileTypeError):
            await pipeline._parse_document(
                file_path=Path("/test.txt"),
                mime_type="text/plain",
            )


class TestDocumentProcessingPipelineGetTextWithSections:
    """Tests for get_text_with_sections method."""

    def test_get_text_with_sections_basic(self) -> None:
        """Test get_text_with_sections with basic input."""
        from aria.document_processing.extractors.sections import (
            ExtractedSections,
            Section,
        )
        from aria.document_processing.parsers.base import ParsedDocument, ParsedPage

        pipeline = DocumentProcessingPipeline()

        parsed = ParsedDocument(
            filename="test.pdf",
            file_type="application/pdf",
            total_pages=2,
            pages=[
                ParsedPage(page_number=1, text="Introduction to the topic."),
                ParsedPage(page_number=2, text="Methods used in the study."),
            ],
            full_text="Introduction to the topic.\n\nMethods used in the study.",
        )

        sections = ExtractedSections(
            sections=[
                Section(
                    name="Introduction",
                    content="Introduction to the topic.",
                    start_pos=0,
                    end_pos=27,
                ),
                Section(
                    name="Methods",
                    content="Methods used in the study.",
                    start_pos=29,
                    end_pos=55,
                ),
            ]
        )

        results = pipeline.get_text_with_sections(parsed, sections)

        assert len(results) == 2
        # First result should be page 1
        assert results[0][0] == "Introduction to the topic."
        assert results[0][2] == 1  # Page number
        # Second result should be page 2
        assert results[1][0] == "Methods used in the study."
        assert results[1][2] == 2  # Page number

    def test_get_text_with_sections_no_match(self) -> None:
        """Test get_text_with_sections when sections don't match."""
        from aria.document_processing.extractors.sections import (
            ExtractedSections,
            Section,
        )
        from aria.document_processing.parsers.base import ParsedDocument, ParsedPage

        pipeline = DocumentProcessingPipeline()

        parsed = ParsedDocument(
            filename="test.pdf",
            file_type="application/pdf",
            total_pages=1,
            pages=[
                ParsedPage(page_number=1, text="Some random content."),
            ],
            full_text="Some random content.",
        )

        sections = ExtractedSections(
            sections=[
                Section(
                    name="Introduction",
                    content="Different content entirely.",
                    start_pos=0,
                    end_pos=27,
                ),
            ]
        )

        results = pipeline.get_text_with_sections(parsed, sections)

        assert len(results) == 1
        assert results[0][0] == "Some random content."
        assert results[0][1] is None  # No section match
        assert results[0][2] == 1

    def test_get_text_with_sections_empty_document(self) -> None:
        """Test get_text_with_sections with empty document."""
        from aria.document_processing.extractors.sections import ExtractedSections
        from aria.document_processing.parsers.base import ParsedDocument

        pipeline = DocumentProcessingPipeline()

        parsed = ParsedDocument(
            filename="empty.pdf",
            file_type="application/pdf",
            total_pages=0,
            pages=[],
            full_text="",
        )

        sections = ExtractedSections(sections=[])

        results = pipeline.get_text_with_sections(parsed, sections)

        assert len(results) == 0
