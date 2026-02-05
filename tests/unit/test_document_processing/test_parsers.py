"""Unit tests for document parsers."""


import pytest

from aria.document_processing.parsers.base import BaseParser, ParsedDocument, ParsedPage


class TestParsedPage:
    """Tests for ParsedPage model."""

    def test_parsed_page_creation(self) -> None:
        """Test creating ParsedPage."""
        page = ParsedPage(
            page_number=1,
            text="This is the page content.",
        )

        assert page.page_number == 1
        assert page.text == "This is the page content."
        assert page.tables == []
        assert page.metadata == {}

    def test_parsed_page_with_tables(self) -> None:
        """Test ParsedPage with tables."""
        page = ParsedPage(
            page_number=2,
            text="Table here",
            tables=[[["A", "B"], ["1", "2"]]],
        )

        assert len(page.tables) == 1
        assert page.tables[0][0] == ["A", "B"]

    def test_parsed_page_with_metadata(self) -> None:
        """Test ParsedPage with metadata."""
        page = ParsedPage(
            page_number=1,
            text="Content",
            metadata={"width": 612, "height": 792},
        )

        assert page.metadata["width"] == 612


class TestParsedDocument:
    """Tests for ParsedDocument model."""

    def test_parsed_document_creation(self) -> None:
        """Test creating ParsedDocument."""
        doc = ParsedDocument(
            filename="test.pdf",
            file_type="application/pdf",
            total_pages=2,
            pages=[
                ParsedPage(page_number=1, text="Page 1"),
                ParsedPage(page_number=2, text="Page 2"),
            ],
            full_text="Page 1\n\nPage 2",
        )

        assert doc.filename == "test.pdf"
        assert doc.file_type == "application/pdf"
        assert doc.total_pages == 2
        assert len(doc.pages) == 2

    def test_text_by_page_property(self) -> None:
        """Test text_by_page property."""
        doc = ParsedDocument(
            filename="test.pdf",
            file_type="application/pdf",
            total_pages=2,
            pages=[
                ParsedPage(page_number=1, text="First page content"),
                ParsedPage(page_number=2, text="Second page content"),
            ],
            full_text="First page content\n\nSecond page content",
        )

        by_page = doc.text_by_page

        assert by_page[1] == "First page content"
        assert by_page[2] == "Second page content"

    def test_empty_document(self) -> None:
        """Test empty document."""
        doc = ParsedDocument(
            filename="empty.pdf",
            file_type="application/pdf",
            total_pages=0,
            pages=[],
            full_text="",
        )

        assert doc.total_pages == 0
        assert doc.text_by_page == {}


class TestBaseParser:
    """Tests for BaseParser abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseParser cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseParser()

    def test_subclass_must_implement_supported_types(self) -> None:
        """Test that subclass must implement supported_types."""

        class IncompleteParser(BaseParser):
            async def parse(self, file_path) -> ParsedDocument:
                return ParsedDocument(
                    filename="test",
                    file_type="text/plain",
                    total_pages=0,
                    pages=[],
                    full_text="",
                )

        with pytest.raises(TypeError):
            IncompleteParser()

    def test_subclass_must_implement_parse(self) -> None:
        """Test that subclass must implement parse."""

        class IncompleteParser(BaseParser):
            @property
            def supported_types(self) -> list[str]:
                return ["text/plain"]

        with pytest.raises(TypeError):
            IncompleteParser()

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that complete implementation can be instantiated."""
        from pathlib import Path

        class MockParser(BaseParser):
            @property
            def supported_types(self) -> list[str]:
                return ["text/plain"]

            async def parse(self, file_path: Path) -> ParsedDocument:
                return ParsedDocument(
                    filename=str(file_path),
                    file_type="text/plain",
                    total_pages=1,
                    pages=[ParsedPage(page_number=1, text="content")],
                    full_text="content",
                )

        parser = MockParser()
        assert isinstance(parser, BaseParser)

    def test_supports_method(self) -> None:
        """Test the supports() method."""
        from pathlib import Path

        class MockParser(BaseParser):
            @property
            def supported_types(self) -> list[str]:
                return ["text/plain", "text/markdown"]

            async def parse(self, file_path: Path) -> ParsedDocument:
                return ParsedDocument(
                    filename=str(file_path),
                    file_type="text/plain",
                    total_pages=1,
                    pages=[ParsedPage(page_number=1, text="content")],
                    full_text="content",
                )

        parser = MockParser()

        assert parser.supports("text/plain") is True
        assert parser.supports("text/markdown") is True
        assert parser.supports("application/pdf") is False
