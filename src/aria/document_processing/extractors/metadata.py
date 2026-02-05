"""Metadata extractor for scientific documents."""

import re
from dataclasses import dataclass

import structlog

from aria.document_processing.parsers.base import ParsedDocument

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedMetadata:
    """Extracted document metadata."""

    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    abstract: str | None = None
    keywords: list[str] | None = None


class MetadataExtractor:
    """Extract metadata from parsed scientific documents.

    Uses heuristics and pattern matching to extract:
    - Title (from PDF metadata or first page)
    - Authors
    - DOI
    - Abstract
    - Keywords
    - Publication year
    """

    # DOI pattern
    DOI_PATTERN = re.compile(
        r"(?:doi[:\s]*)?(?:https?://(?:dx\.)?doi\.org/)?"
        r"(10\.\d{4,}/[^\s]+)",
        re.IGNORECASE,
    )

    # Year pattern (1900-2099)
    YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

    # Abstract pattern
    ABSTRACT_PATTERNS = [
        re.compile(
            r"abstract[:\s]*(.{100,2000}?)(?=\n\n|\bintroduction\b|\b1\.\s)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"summary[:\s]*(.{100,2000}?)(?=\n\n|\bintroduction\b|\b1\.\s)",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    # Keywords pattern
    KEYWORDS_PATTERN = re.compile(
        r"(?:key\s*words?|keywords?)[:\s]*([^\n]+)",
        re.IGNORECASE,
    )

    def extract(self, document: ParsedDocument) -> ExtractedMetadata:
        """Extract metadata from a parsed document.

        Args:
            document: Parsed document.

        Returns:
            ExtractedMetadata: Extracted metadata fields.
        """
        logger.info("extracting_metadata", filename=document.filename)

        text = document.full_text
        first_page_text = document.pages[0].text if document.pages else ""

        # Extract each field
        title = self._extract_title(document, first_page_text)
        authors = self._extract_authors(first_page_text)
        doi = self._extract_doi(text)
        abstract = self._extract_abstract(text)
        keywords = self._extract_keywords(text)
        year = self._extract_year(text, document.metadata)

        metadata = ExtractedMetadata(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            keywords=keywords,
        )

        logger.info(
            "metadata_extracted",
            filename=document.filename,
            has_title=bool(title),
            has_authors=bool(authors),
            has_doi=bool(doi),
            has_abstract=bool(abstract),
        )

        return metadata

    def _extract_title(
        self,
        document: ParsedDocument,
        first_page_text: str,
    ) -> str | None:
        """Extract document title.

        Args:
            document: Parsed document.
            first_page_text: Text from first page.

        Returns:
            Extracted title or None.
        """
        # First try PDF metadata
        if document.metadata.get("title"):
            return document.metadata["title"]

        # Try to extract from first page
        # Usually the title is in the first few lines, often in larger font
        lines = first_page_text.split("\n")
        for line in lines[:10]:
            line = line.strip()
            # Title heuristics: non-empty, reasonable length, not all caps header
            if (
                line
                and 10 < len(line) < 300
                and not line.isupper()
                and not line.startswith("http")
                and not re.match(r"^\d", line)  # Doesn't start with number
            ):
                return line

        return None

    def _extract_authors(self, first_page_text: str) -> list[str] | None:
        """Extract author names.

        Args:
            first_page_text: Text from first page.

        Returns:
            List of author names or None.
        """
        # Look for common author patterns
        # Pattern: "Author1, Author2, and Author3"
        # Pattern: "Author1 1, Author2 2" (with affiliations)

        # Simple heuristic: look for lines with multiple commas or "and"
        lines = first_page_text.split("\n")
        for line in lines[1:15]:  # Skip title, look in next 15 lines
            line = line.strip()
            if not line:
                continue

            # Check for author-like pattern
            if (
                ("," in line or " and " in line.lower())
                and len(line) < 500
                and not any(
                    word in line.lower() for word in ["abstract", "introduction", "keywords"]
                )
            ):
                # Try to split by common separators
                authors = re.split(r",\s*(?:and\s*)?|\s+and\s+", line)
                # Clean up author names
                authors = [re.sub(r"\d+|\*|†|‡|§", "", a).strip() for a in authors if a.strip()]
                # Filter out unlikely author names
                authors = [a for a in authors if len(a) > 2 and len(a) < 100 and not a.isupper()]
                if authors and len(authors) <= 20:
                    return authors

        return None

    def _extract_doi(self, text: str) -> str | None:
        """Extract DOI from text.

        Args:
            text: Full document text.

        Returns:
            DOI string or None.
        """
        match = self.DOI_PATTERN.search(text[:5000])  # Search first part
        if match:
            return match.group(1)
        return None

    def _extract_abstract(self, text: str) -> str | None:
        """Extract abstract from text.

        Args:
            text: Full document text.

        Returns:
            Abstract text or None.
        """
        for pattern in self.ABSTRACT_PATTERNS:
            match = pattern.search(text[:10000])  # Search beginning
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r"\s+", " ", abstract)
                if len(abstract) > 50:
                    return abstract
        return None

    def _extract_keywords(self, text: str) -> list[str] | None:
        """Extract keywords from text.

        Args:
            text: Full document text.

        Returns:
            List of keywords or None.
        """
        match = self.KEYWORDS_PATTERN.search(text[:10000])
        if match:
            keywords_str = match.group(1)
            # Split by common separators
            keywords = re.split(r"[,;•·]", keywords_str)
            keywords = [k.strip() for k in keywords if k.strip()]
            # Filter out unlikely keywords
            keywords = [k for k in keywords if len(k) > 2 and len(k) < 50]
            if keywords:
                return keywords
        return None

    def _extract_year(
        self,
        text: str,
        pdf_metadata: dict,
    ) -> int | None:
        """Extract publication year.

        Args:
            text: Full document text.
            pdf_metadata: PDF metadata dict.

        Returns:
            Publication year or None.
        """
        # Try PDF metadata first
        creation_date = pdf_metadata.get("creation_date", "")
        if creation_date:
            match = self.YEAR_PATTERN.search(str(creation_date))
            if match:
                return int(match.group(0))

        # Search in text (first part only)
        matches = self.YEAR_PATTERN.findall(text[:5000])
        if matches:
            # Return the most recent reasonable year
            years = [int(f"{m[0]}{m[1]}") if len(m) == 2 else int(m) for m in matches]
            # Assuming it's an older format, reconstruct properly
            years = []
            for match in self.YEAR_PATTERN.finditer(text[:5000]):
                year = int(match.group(0))
                if 1900 <= year <= 2030:
                    years.append(year)
            if years:
                return max(years)  # Most recent year

        return None
