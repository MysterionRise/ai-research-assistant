"""Unit tests for RAG synthesis module."""

from dataclasses import field
from unittest.mock import MagicMock, patch

import pytest


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_synthesis_result_creation(self) -> None:
        """Test creating SynthesisResult."""
        from aria.rag.synthesis.citation_aware import SynthesisResult

        result = SynthesisResult(
            answer="The answer based on [1].",
            sources_used=1,
            confidence=0.85,
            tokens_used=100,
        )
        assert result.answer == "The answer based on [1]."
        assert result.sources_used == 1
        assert result.confidence == 0.85
        assert result.tokens_used == 100

    def test_synthesis_result_defaults(self) -> None:
        """Test SynthesisResult default values."""
        from aria.rag.synthesis.citation_aware import SynthesisResult

        result = SynthesisResult(answer="Answer")
        assert result.citations == []
        assert result.sources_used == 0
        assert result.confidence == 0.0
        assert result.tokens_used == 0
        assert result.metadata == {}


class TestCitationAwareSynthesizerInit:
    """Tests for CitationAwareSynthesizer initialization."""

    def test_synthesizer_init_default_model(self) -> None:
        """Test initialization with default model."""
        with patch("aria.rag.synthesis.citation_aware.settings") as mock_settings:
            mock_settings.anthropic_model = "claude-3-sonnet"

            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer()
            assert synthesizer.model == "claude-3-sonnet"

    def test_synthesizer_init_custom_model(self) -> None:
        """Test initialization with custom model."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="claude-3-opus")
            assert synthesizer.model == "claude-3-opus"


class TestFormatContext:
    """Tests for _format_context method."""

    def test_format_context_with_chunks(self) -> None:
        """Test formatting context with chunks."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="First chunk content.",
                    score=0.9,
                    document_title="Paper 1",
                ),
                RetrievalResult(
                    chunk_id="c2",
                    document_id="d2",
                    content="Second chunk content.",
                    score=0.85,
                    document_title="Paper 2",
                    section="Methods",
                    page_number=10,
                ),
            ]

            formatted, citation_map = synthesizer._format_context(context)

            assert "[1]" in formatted
            assert "[2]" in formatted
            assert "Paper 1" in formatted
            assert "Paper 2" in formatted
            assert "Methods" in formatted
            assert "p. 10" in formatted
            assert 1 in citation_map
            assert 2 in citation_map

    def test_format_context_without_title(self) -> None:
        """Test formatting context when title is missing."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="Content without title.",
                    score=0.9,
                    document_title=None,
                ),
            ]

            formatted, _ = synthesizer._format_context(context)

            # Should use "Unknown Document" as fallback
            assert "Unknown Document" in formatted


class TestBuildPrompt:
    """Tests for _build_prompt method."""

    def test_build_prompt(self) -> None:
        """Test building the synthesis prompt."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            prompt = synthesizer._build_prompt(
                query="What is photosynthesis?",
                context="[1] Paper about plants:\nPhotosynthesis is...",
            )

            assert "What is photosynthesis?" in prompt
            assert "Photosynthesis is" in prompt
            assert "[1]" in prompt or "citation" in prompt.lower()


class TestExtractCitations:
    """Tests for _extract_citations method."""

    def test_extract_citations_from_answer(self) -> None:
        """Test extracting citations from answer text."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="Content for citation 1.",
                    score=0.9,
                    document_title="Paper 1",
                ),
                RetrievalResult(
                    chunk_id="c2",
                    document_id="d2",
                    content="Content for citation 2.",
                    score=0.85,
                    document_title="Paper 2",
                ),
            ]

            citation_map = {1: context[0], 2: context[1]}

            answer = "Based on [1] and [2], the answer is clear."

            citations = synthesizer._extract_citations(answer, citation_map, context)

            assert len(citations) == 2
            assert citations[0].citation_id == 1
            assert citations[0].document_id == "d1"
            assert citations[1].citation_id == 2

    def test_extract_citations_only_used(self) -> None:
        """Test that only used citations are extracted."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="Content 1.",
                    score=0.9,
                    document_title="Paper 1",
                ),
                RetrievalResult(
                    chunk_id="c2",
                    document_id="d2",
                    content="Content 2.",
                    score=0.8,
                    document_title="Paper 2",
                ),
            ]

            citation_map = {1: context[0], 2: context[1]}

            # Only [1] is used
            answer = "Based on [1], the result is..."

            citations = synthesizer._extract_citations(answer, citation_map, context)

            assert len(citations) == 1
            assert citations[0].citation_id == 1

    def test_extract_citations_none_used(self) -> None:
        """Test extracting when no citations are used."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="Content.",
                    score=0.9,
                    document_title="Paper",
                ),
            ]

            citation_map = {1: context[0]}

            answer = "I don't have enough information."

            citations = synthesizer._extract_citations(answer, citation_map, context)

            assert len(citations) == 0


class TestSynthesizerModuleExports:
    """Tests for synthesis module exports."""

    def test_synthesizer_exported(self) -> None:
        """Test that CitationAwareSynthesizer is exported."""
        from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

        assert CitationAwareSynthesizer is not None

    def test_synthesis_result_exported(self) -> None:
        """Test that SynthesisResult is exported."""
        from aria.rag.synthesis.citation_aware import SynthesisResult

        assert SynthesisResult is not None


class TestSynthesizerWithCitations:
    """Additional tests for citation extraction."""

    def test_extract_citations_with_long_content(self) -> None:
        """Test extracting citations with content longer than 200 chars."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            # Create content longer than 200 chars
            long_content = "A" * 250  # 250 characters

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content=long_content,
                    score=0.9,
                    document_title="Long Paper",
                ),
            ]

            citation_map = {1: context[0]}

            answer = "Based on [1], the answer is..."

            citations = synthesizer._extract_citations(answer, citation_map, context)

            assert len(citations) == 1
            # Excerpt should be truncated with "..."
            assert len(citations[0].excerpt) == 203  # 200 + 3 for "..."
            assert citations[0].excerpt.endswith("...")

    def test_extract_citations_short_content(self) -> None:
        """Test extracting citations with content shorter than 200 chars."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            short_content = "Short content here."

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content=short_content,
                    score=0.9,
                    document_title="Paper",
                ),
            ]

            citation_map = {1: context[0]}

            answer = "Based on [1], the answer..."

            citations = synthesizer._extract_citations(answer, citation_map, context)

            assert len(citations) == 1
            # Excerpt should be the full content without "..."
            assert citations[0].excerpt == short_content
            assert not citations[0].excerpt.endswith("...")

    def test_extract_citations_invalid_number(self) -> None:
        """Test extracting citations with invalid citation numbers."""
        with patch("aria.rag.synthesis.citation_aware.settings"):
            from aria.rag.retrieval.base import RetrievalResult
            from aria.rag.synthesis.citation_aware import CitationAwareSynthesizer

            synthesizer = CitationAwareSynthesizer(model="test-model")

            context = [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="Content.",
                    score=0.9,
                    document_title="Paper",
                ),
            ]

            citation_map = {1: context[0]}

            # Answer references [5] which doesn't exist
            answer = "Based on [5], the answer..."

            citations = synthesizer._extract_citations(answer, citation_map, context)

            # Should return empty list since [5] doesn't exist
            assert len(citations) == 0
