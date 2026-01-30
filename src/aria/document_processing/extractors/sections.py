"""Section extractor for scientific documents."""

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Section:
    """Represents a document section."""

    name: str
    content: str
    start_pos: int
    end_pos: int
    level: int = 1  # Section nesting level


@dataclass
class ExtractedSections:
    """Extracted document sections."""

    sections: list[Section] = field(default_factory=list)

    def get_section(self, name: str) -> Section | None:
        """Get section by name (case-insensitive)."""
        name_lower = name.lower()
        for section in self.sections:
            if section.name.lower() == name_lower:
                return section
        return None

    @property
    def section_names(self) -> list[str]:
        """Get list of section names."""
        return [s.name for s in self.sections]


class SectionExtractor:
    """Extract sections from scientific documents.

    Detects common scientific paper sections:
    - Abstract
    - Introduction
    - Methods/Materials
    - Results
    - Discussion
    - Conclusion
    - References
    """

    # Common section heading patterns
    SECTION_PATTERNS = [
        # Numbered sections: "1. Introduction", "2 Methods"
        re.compile(
            r"^\s*(\d+\.?\s*)(abstract|introduction|background|methods?|materials?"
            r"|results?|discussion|conclusions?|references?|acknowledgements?"
            r"|supplementary|appendix)",
            re.IGNORECASE | re.MULTILINE,
        ),
        # Unnumbered sections: "INTRODUCTION", "Methods"
        re.compile(
            r"^\s*(abstract|introduction|background|methods?|materials?\s*(?:and\s*methods?)?"
            r"|results?(?:\s*and\s*discussion)?|discussion|conclusions?"
            r"|references?|acknowledgements?|supplementary|appendix)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ]

    # Section name normalization mapping
    SECTION_NORMALIZATION = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "background": "Introduction",
        "method": "Methods",
        "methods": "Methods",
        "materials": "Methods",
        "materials and methods": "Methods",
        "result": "Results",
        "results": "Results",
        "results and discussion": "Results and Discussion",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "reference": "References",
        "references": "References",
        "acknowledgement": "Acknowledgements",
        "acknowledgements": "Acknowledgements",
        "supplementary": "Supplementary",
        "appendix": "Appendix",
    }

    def extract(self, text: str) -> ExtractedSections:
        """Extract sections from document text.

        Args:
            text: Full document text.

        Returns:
            ExtractedSections: Extracted sections with content.
        """
        logger.info("extracting_sections")

        # Find all section headings
        headings: list[tuple[int, str, str]] = []  # (position, raw_heading, normalized_name)

        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(text):
                pos = match.start()
                raw_heading = match.group(0).strip()

                # Extract the section name (remove numbers)
                name = re.sub(r"^\d+\.?\s*", "", raw_heading).strip().lower()

                # Normalize section name
                normalized = self._normalize_section_name(name)
                if normalized:
                    headings.append((pos, raw_heading, normalized))

        # Remove duplicates and sort by position
        headings = sorted(set(headings), key=lambda x: x[0])

        # Extract section content
        sections: list[Section] = []
        for i, (pos, raw_heading, name) in enumerate(headings):
            # Content starts after the heading
            content_start = pos + len(raw_heading)

            # Content ends at next section or end of text
            if i + 1 < len(headings):
                content_end = headings[i + 1][0]
            else:
                content_end = len(text)

            content = text[content_start:content_end].strip()

            # Only include sections with meaningful content (>20 chars to filter noise)
            if len(content) > 20:
                sections.append(
                    Section(
                        name=name,
                        content=content,
                        start_pos=pos,
                        end_pos=content_end,
                        level=1,
                    )
                )

        logger.info(
            "sections_extracted",
            count=len(sections),
            names=[s.name for s in sections],
        )

        return ExtractedSections(sections=sections)

    def _normalize_section_name(self, name: str) -> str | None:
        """Normalize section name to standard form.

        Args:
            name: Raw section name.

        Returns:
            Normalized name or None if not recognized.
        """
        name_lower = name.lower().strip()

        # Direct match
        if name_lower in self.SECTION_NORMALIZATION:
            return self.SECTION_NORMALIZATION[name_lower]

        # Partial match
        for key, normalized in self.SECTION_NORMALIZATION.items():
            if key in name_lower or name_lower in key:
                return normalized

        return None

    def get_section_for_position(
        self,
        extracted: ExtractedSections,
        char_position: int,
    ) -> str | None:
        """Get section name for a character position.

        Args:
            extracted: Extracted sections.
            char_position: Character position in original text.

        Returns:
            Section name or None if not in any section.
        """
        for section in extracted.sections:
            if section.start_pos <= char_position < section.end_pos:
                return section.name
        return None
