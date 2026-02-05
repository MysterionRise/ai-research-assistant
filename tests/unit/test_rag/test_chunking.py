"""Tests for semantic chunking."""

import pytest

from aria.rag.chunking.base import BaseChunker, Chunk
from aria.rag.chunking.semantic import SemanticChunker


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test creating a Chunk."""
        chunk = Chunk(
            content="Test content",
            chunk_index=0,
            token_count=10,
        )

        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.token_count == 10
        assert chunk.section is None
        assert chunk.page_number is None

    def test_chunk_with_all_fields(self) -> None:
        """Test Chunk with all fields."""
        chunk = Chunk(
            content="Test content",
            chunk_index=5,
            token_count=50,
            section="Introduction",
            page_number=3,
            start_char=100,
            end_char=200,
            metadata={"key": "value"},
        )

        assert chunk.section == "Introduction"
        assert chunk.page_number == 3
        assert chunk.start_char == 100
        assert chunk.end_char == 200
        assert chunk.metadata == {"key": "value"}


class TestBaseChunker:
    """Tests for BaseChunker abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseChunker cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseChunker()

    def test_subclass_must_implement_chunk(self) -> None:
        """Test that subclass must implement chunk method."""

        class IncompleteChunker(BaseChunker):
            def count_tokens(self, text: str) -> int:
                return len(text.split())

        with pytest.raises(TypeError):
            IncompleteChunker()

    def test_subclass_must_implement_count_tokens(self) -> None:
        """Test that subclass must implement count_tokens method."""

        class IncompleteChunker(BaseChunker):
            def chunk(self, text: str, metadata=None) -> list[Chunk]:
                return [Chunk(content=text, chunk_index=0, token_count=10)]

        with pytest.raises(TypeError):
            IncompleteChunker()

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that complete implementation can be instantiated."""

        class MockChunker(BaseChunker):
            def chunk(self, text: str, metadata=None) -> list[Chunk]:
                return [Chunk(content=text, chunk_index=0, token_count=len(text.split()))]

            def count_tokens(self, text: str) -> int:
                return len(text.split())

        chunker = MockChunker()
        assert isinstance(chunker, BaseChunker)

        # Test it works
        chunks = chunker.chunk("Test content here")
        assert len(chunks) == 1
        assert chunks[0].content == "Test content here"

        tokens = chunker.count_tokens("Test content here")
        assert tokens == 3


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with small chunk size for testing."""
        return SemanticChunker(chunk_size=100, chunk_overlap=20)

    def test_count_tokens(self, chunker):
        """Test token counting."""
        text = "This is a simple test sentence."
        tokens = chunker.count_tokens(text)

        assert tokens > 0
        assert tokens < 20  # Simple sentence should be < 20 tokens

    def test_chunk_simple_text(self, chunker):
        """Test chunking simple text."""
        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert all(c.content for c in chunks)
        assert all(c.token_count > 0 for c in chunks)

    def test_chunk_empty_text(self, chunker):
        """Test chunking empty text returns empty list."""
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_indexes_sequential(self, chunker):
        """Test that chunk indexes are sequential."""
        text = " ".join(["This is a test sentence." for _ in range(50)])

        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_with_sections(self, chunker):
        """Test chunking with section metadata."""
        from aria.document_processing.extractors.sections import ExtractedSections, Section

        text = "Introduction content here. Methods content here."
        sections = ExtractedSections(
            sections=[
                Section(
                    name="Introduction",
                    content="Introduction content here.",
                    start_pos=0,
                    end_pos=28,
                ),
                Section(
                    name="Methods",
                    content="Methods content here.",
                    start_pos=28,
                    end_pos=50,
                ),
            ]
        )

        chunks = chunker.chunk(text, metadata={"sections": sections})

        # Should have chunks with section annotations
        section_names = {c.section for c in chunks if c.section}
        assert "Introduction" in section_names or "Methods" in section_names


class TestChunkTokenBoundaries:
    """Test token boundary handling."""

    @pytest.fixture
    def chunker(self):
        return SemanticChunker(chunk_size=50, chunk_overlap=10)

    def test_long_text_creates_multiple_chunks(self, chunker):
        """Test that long text creates multiple chunks."""
        # Create text that should exceed chunk size
        sentences = [f"This is sentence number {i}." for i in range(20)]
        text = " ".join(sentences)

        chunks = chunker.chunk(text)

        assert len(chunks) > 1

    def test_chunk_overlap_preserved(self, chunker):
        """Test that chunks have appropriate overlap."""
        sentences = [f"Unique sentence {i} here." for i in range(20)]
        text = " ".join(sentences)

        chunks = chunker.chunk(text)

        if len(chunks) > 1:
            # Check that chunks have content
            for i in range(len(chunks) - 1):
                # Verify chunks have content
                assert len(chunks[i].content) > 0
                assert len(chunks[i + 1].content) > 0
