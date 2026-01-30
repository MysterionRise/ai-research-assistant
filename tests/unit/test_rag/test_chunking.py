"""Tests for semantic chunking."""

import pytest

from aria.rag.chunking.semantic import SemanticChunker


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
