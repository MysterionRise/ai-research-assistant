"""Base parser interface for document parsing."""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    """Represents a parsed page from a document."""

    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Extracted text content")
    tables: list[list[list[str]]] = Field(
        default_factory=list,
        description="Tables as list of rows",
    )
    metadata: dict = Field(default_factory=dict, description="Page metadata")


class ParsedDocument(BaseModel):
    """Represents a fully parsed document."""

    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="MIME type")
    total_pages: int = Field(..., description="Total page count")
    pages: list[ParsedPage] = Field(..., description="Parsed pages")
    full_text: str = Field(..., description="Concatenated full text")
    metadata: dict = Field(default_factory=dict, description="Document metadata")

    @property
    def text_by_page(self) -> dict[int, str]:
        """Get text content indexed by page number."""
        return {page.page_number: page.text for page in self.pages}


class BaseParser(ABC):
    """Abstract base class for document parsers.

    All document parsers should inherit from this class and implement
    the parse method.
    """

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Return list of supported MIME types."""
        pass

    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document from file path.

        Args:
            file_path: Path to the document file.

        Returns:
            ParsedDocument: Parsed document with text and metadata.

        Raises:
            DocumentParsingError: If parsing fails.
        """
        pass

    def supports(self, mime_type: str) -> bool:
        """Check if this parser supports the given MIME type.

        Args:
            mime_type: MIME type to check.

        Returns:
            True if supported, False otherwise.
        """
        return mime_type in self.supported_types
