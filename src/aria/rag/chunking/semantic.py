"""Semantic chunking strategy for scientific documents."""

import re

import structlog
import tiktoken

from aria.config.settings import settings
from aria.document_processing.extractors.sections import ExtractedSections
from aria.rag.chunking.base import BaseChunker, Chunk

logger = structlog.get_logger(__name__)


class SemanticChunker(BaseChunker):
    """Section-aware semantic chunker for scientific documents.

    Features:
    - Preserves section boundaries
    - Uses sliding window with overlap
    - Token-based sizing using tiktoken
    - Tries to split on sentence boundaries
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        model: str = "cl100k_base",  # GPT-4/Claude compatible encoding
    ) -> None:
        """Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
            model: Tiktoken encoding model name.
        """
        self.chunk_size = chunk_size or settings.rag_chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag_chunk_overlap
        self.encoding = tiktoken.get_encoding(model)

        logger.info(
            "semantic_chunker_initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        return len(self.encoding.encode(text))

    def chunk(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into semantic chunks.

        Args:
            text: Text to chunk.
            metadata: Optional metadata (can include 'sections').

        Returns:
            List of Chunk objects.
        """
        metadata = metadata or {}
        sections: ExtractedSections | None = metadata.get("sections")

        if sections and sections.sections:
            # Use section-aware chunking
            return self._chunk_with_sections(text, sections)
        else:
            # Fall back to simple chunking
            return self._chunk_simple(text)

    def _chunk_with_sections(
        self,
        text: str,
        sections: ExtractedSections,
    ) -> list[Chunk]:
        """Chunk text while respecting section boundaries.

        Args:
            text: Full document text.
            sections: Extracted sections.

        Returns:
            List of chunks with section annotations.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for section in sections.sections:
            section_chunks = self._chunk_text(
                section.content,
                section_name=section.name,
                start_offset=section.start_pos,
            )

            for chunk in section_chunks:
                chunk.chunk_index = chunk_index
                chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "chunked_with_sections",
            section_count=len(sections.sections),
            chunk_count=len(chunks),
        )

        return chunks

    def _chunk_simple(self, text: str) -> list[Chunk]:
        """Simple chunking without section awareness.

        Args:
            text: Text to chunk.

        Returns:
            List of chunks.
        """
        return self._chunk_text(text)

    def _chunk_text(
        self,
        text: str,
        section_name: str | None = None,
        start_offset: int = 0,
    ) -> list[Chunk]:
        """Core chunking logic with sliding window.

        Args:
            text: Text to chunk.
            section_name: Optional section name to annotate.
            start_offset: Character offset in original document.

        Returns:
            List of chunks.
        """
        chunks: list[Chunk] = []

        # Split into sentences for cleaner boundaries
        sentences = self._split_sentences(text)

        current_chunk: list[str] = []
        current_tokens = 0
        current_start = start_offset
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        token_count=current_tokens,
                        section=section_name,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                    )
                )
                chunk_index += 1

                # Calculate overlap
                overlap_text, overlap_tokens = self._get_overlap(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = overlap_tokens
                current_start = current_start + len(chunk_text) - len(overlap_text)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    section=section_name,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting - handles common cases
        # Could be enhanced with nltk or spacy for better accuracy
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_endings.split(text)

        # Clean up and filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _get_overlap(
        self,
        sentences: list[str],
    ) -> tuple[str, int]:
        """Get overlap text from end of chunk.

        Args:
            sentences: List of sentences in current chunk.

        Returns:
            Tuple of (overlap_text, overlap_tokens).
        """
        if not sentences:
            return "", 0

        overlap_sentences: list[str] = []
        overlap_tokens = 0

        # Take sentences from end until we hit overlap limit
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return " ".join(overlap_sentences), overlap_tokens
