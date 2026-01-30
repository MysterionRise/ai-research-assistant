"""Main document processing pipeline."""

from pathlib import Path

import structlog

from aria.document_processing.extractors.metadata import ExtractedMetadata, MetadataExtractor
from aria.document_processing.extractors.sections import ExtractedSections, SectionExtractor
from aria.document_processing.parsers.base import ParsedDocument
from aria.document_processing.parsers.pdf import PDFParser
from aria.exceptions import UnsupportedFileTypeError

logger = structlog.get_logger(__name__)


class DocumentProcessingPipeline:
    """End-to-end document processing pipeline.

    Orchestrates:
    1. Document parsing (PDF, etc.)
    2. Metadata extraction
    3. Section detection
    4. Preparation for chunking and embedding
    """

    SUPPORTED_MIME_TYPES = {
        "application/pdf": PDFParser,
    }

    def __init__(self) -> None:
        """Initialize the processing pipeline."""
        self.pdf_parser = PDFParser()
        self.metadata_extractor = MetadataExtractor()
        self.section_extractor = SectionExtractor()

    async def process(
        self,
        file_path: Path,
        mime_type: str,
    ) -> tuple[ParsedDocument, ExtractedMetadata, ExtractedSections]:
        """Process a document through the full pipeline.

        Args:
            file_path: Path to the document file.
            mime_type: MIME type of the document.

        Returns:
            Tuple of (ParsedDocument, ExtractedMetadata, ExtractedSections).

        Raises:
            UnsupportedFileTypeError: If file type is not supported.
            DocumentParsingError: If parsing fails.
        """
        logger.info(
            "processing_document",
            file_path=str(file_path),
            mime_type=mime_type,
        )

        # Validate file type
        if mime_type not in self.SUPPORTED_MIME_TYPES:
            raise UnsupportedFileTypeError(
                file_type=mime_type,
                supported_types=list(self.SUPPORTED_MIME_TYPES.keys()),
            )

        # Parse document
        parsed = await self._parse_document(file_path, mime_type)

        # Extract metadata
        metadata = self.metadata_extractor.extract(parsed)

        # Extract sections
        sections = self.section_extractor.extract(parsed.full_text)

        logger.info(
            "document_processed",
            filename=parsed.filename,
            pages=parsed.total_pages,
            sections=len(sections.sections),
            has_title=bool(metadata.title),
            has_abstract=bool(metadata.abstract),
        )

        return parsed, metadata, sections

    async def _parse_document(
        self,
        file_path: Path,
        mime_type: str,
    ) -> ParsedDocument:
        """Parse document based on MIME type.

        Args:
            file_path: Path to document.
            mime_type: Document MIME type.

        Returns:
            ParsedDocument: Parsed document.
        """
        if mime_type == "application/pdf":
            return await self.pdf_parser.parse(file_path)

        raise UnsupportedFileTypeError(
            file_type=mime_type,
            supported_types=list(self.SUPPORTED_MIME_TYPES.keys()),
        )

    def get_text_with_sections(
        self,
        parsed: ParsedDocument,
        sections: ExtractedSections,
    ) -> list[tuple[str, str | None, int | None]]:
        """Get text chunks with section and page annotations.

        Args:
            parsed: Parsed document.
            sections: Extracted sections.

        Returns:
            List of (text, section_name, page_number) tuples.
        """
        results: list[tuple[str, str | None, int | None]] = []

        for page in parsed.pages:
            # Determine which section this page belongs to
            section_name = None
            for section in sections.sections:
                if section.content[:100] in page.text:
                    section_name = section.name
                    break

            results.append((page.text, section_name, page.page_number))

        return results
