"""PDF parser using pdfplumber with fallback chain."""

import re
from pathlib import Path

import structlog

from aria.document_processing.parsers.base import BaseParser, ParsedDocument, ParsedPage
from aria.exceptions import DocumentParsingError

logger = structlog.get_logger(__name__)


class PDFParser(BaseParser):
    """PDF parser using pdfplumber with pypdf fallback.

    Implements a fallback chain for robust PDF parsing:
    1. pdfplumber (best for text + tables)
    2. pypdf (fallback for problematic PDFs)
    """

    @property
    def supported_types(self) -> list[str]:
        """Return supported MIME types."""
        return ["application/pdf"]

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument: Parsed document with text and metadata.

        Raises:
            DocumentParsingError: If all parsing methods fail.
        """
        logger.info("parsing_pdf", file_path=str(file_path))

        # Try pdfplumber first
        try:
            return await self._parse_with_pdfplumber(file_path)
        except Exception as e:
            logger.warning(
                "pdfplumber_failed_trying_fallback",
                file_path=str(file_path),
                error=str(e),
            )

        # Fallback to pypdf
        try:
            return await self._parse_with_pypdf(file_path)
        except Exception as e:
            logger.error(
                "all_pdf_parsers_failed",
                file_path=str(file_path),
                error=str(e),
            )
            raise DocumentParsingError(
                filename=file_path.name,
                reason=f"All PDF parsing methods failed: {e}",
            ) from e

    async def _parse_with_pdfplumber(self, file_path: Path) -> ParsedDocument:
        """Parse PDF using pdfplumber.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument: Parsed document.
        """
        import pdfplumber

        pages: list[ParsedPage] = []
        all_text: list[str] = []
        metadata: dict = {}

        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            if pdf.metadata:
                metadata = {
                    "title": pdf.metadata.get("Title"),
                    "author": pdf.metadata.get("Author"),
                    "subject": pdf.metadata.get("Subject"),
                    "creator": pdf.metadata.get("Creator"),
                    "creation_date": pdf.metadata.get("CreationDate"),
                }
                # Clean None values
                metadata = {k: v for k, v in metadata.items() if v}

            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                page_num = i + 1

                # Extract text
                text = page.extract_text() or ""
                text = self._clean_text(text)
                all_text.append(text)

                # Extract tables
                tables: list[list[list[str]]] = []
                try:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            # Convert None values to empty strings
                            cleaned_table = [
                                [str(cell) if cell else "" for cell in row] for row in table if row
                            ]
                            if cleaned_table:
                                tables.append(cleaned_table)
                except Exception as e:
                    logger.debug(
                        "table_extraction_failed",
                        page=page_num,
                        error=str(e),
                    )

                pages.append(
                    ParsedPage(
                        page_number=page_num,
                        text=text,
                        tables=tables,
                        metadata={"width": page.width, "height": page.height},
                    )
                )

        return ParsedDocument(
            filename=file_path.name,
            file_type="application/pdf",
            total_pages=total_pages,
            pages=pages,
            full_text="\n\n".join(all_text),
            metadata=metadata,
        )

    async def _parse_with_pypdf(self, file_path: Path) -> ParsedDocument:
        """Parse PDF using pypdf as fallback.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument: Parsed document.
        """
        from pypdf import PdfReader

        pages: list[ParsedPage] = []
        all_text: list[str] = []
        metadata: dict = {}

        reader = PdfReader(file_path)

        # Extract metadata
        if reader.metadata:
            metadata = {
                "title": reader.metadata.title,
                "author": reader.metadata.author,
                "subject": reader.metadata.subject,
                "creator": reader.metadata.creator,
            }
            metadata = {k: v for k, v in metadata.items() if v}

        total_pages = len(reader.pages)

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            text = page.extract_text() or ""
            text = self._clean_text(text)
            all_text.append(text)

            pages.append(
                ParsedPage(
                    page_number=page_num,
                    text=text,
                    tables=[],  # pypdf doesn't extract tables
                    metadata={},
                )
            )

        return ParsedDocument(
            filename=file_path.name,
            file_type="application/pdf",
            total_pages=total_pages,
            pages=pages,
            full_text="\n\n".join(all_text),
            metadata=metadata,
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r"\n\d+\n", "\n", text)
        # Normalize line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
