"""Citation-aware answer synthesis."""

import re
from dataclasses import dataclass, field

import structlog

from aria.config.settings import settings
from aria.rag.retrieval.base import RetrievalResult
from aria.types import Citation

logger = structlog.get_logger(__name__)


@dataclass
class SynthesisResult:
    """Result from answer synthesis."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources_used: int = 0
    confidence: float = 0.0
    tokens_used: int = 0
    metadata: dict = field(default_factory=dict)


class CitationAwareSynthesizer:
    """Synthesizes answers with inline citations.

    Features:
    - Generates answers grounded in retrieved context
    - Adds inline citations [1], [2], etc.
    - Tracks source usage for transparency
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialize synthesizer.

        Args:
            model: LLM model name (default: from settings).
        """
        self.model = model or settings.anthropic_model
        self._client = None

        logger.info("citation_aware_synthesizer_initialized", model=self.model)

    def _get_client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic
            api_key = settings.anthropic_api_key
            if api_key:
                self._client = AsyncAnthropic(api_key=api_key.get_secret_value())
            else:
                raise ValueError("Anthropic API key not configured")
        return self._client

    async def synthesize(
        self,
        query: str,
        context: list[RetrievalResult],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> SynthesisResult:
        """Synthesize an answer with citations.

        Args:
            query: User's question.
            context: Retrieved context chunks.
            max_tokens: Maximum response tokens.
            temperature: LLM temperature (lower = more focused).

        Returns:
            SynthesisResult with answer and citations.
        """
        if not context:
            return SynthesisResult(
                answer="I don't have enough information to answer this question. Please try rephrasing or provide more context.",
                citations=[],
                sources_used=0,
                confidence=0.0,
            )

        logger.info(
            "synthesizing_answer",
            query=query[:100],
            context_chunks=len(context),
        )

        # Build context with citation markers
        formatted_context, citation_map = self._format_context(context)

        # Build prompt
        prompt = self._build_prompt(query, formatted_context)

        # Call LLM
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text

        # Extract citations used
        citations = self._extract_citations(answer, citation_map, context)

        # Calculate confidence based on citation coverage
        confidence = min(1.0, len(citations) / max(1, len(context) // 2))

        result = SynthesisResult(
            answer=answer,
            citations=citations,
            sources_used=len(citations),
            confidence=confidence,
            tokens_used=response.usage.output_tokens,
            metadata={
                "model": self.model,
                "input_tokens": response.usage.input_tokens,
            },
        )

        logger.info(
            "synthesis_completed",
            answer_length=len(answer),
            citations_count=len(citations),
            confidence=confidence,
        )

        return result

    def _format_context(
        self,
        context: list[RetrievalResult],
    ) -> tuple[str, dict[int, RetrievalResult]]:
        """Format context chunks with citation markers.

        Args:
            context: Retrieved chunks.

        Returns:
            Tuple of (formatted_context, citation_map).
        """
        formatted_parts = []
        citation_map = {}

        for i, chunk in enumerate(context, 1):
            citation_map[i] = chunk

            title = chunk.document_title or "Unknown Document"
            section = f" - {chunk.section}" if chunk.section else ""
            page = f" (p. {chunk.page_number})" if chunk.page_number else ""

            formatted_parts.append(
                f"[{i}] {title}{section}{page}:\n{chunk.content}\n"
            )

        return "\n".join(formatted_parts), citation_map

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the synthesis prompt.

        Args:
            query: User's question.
            context: Formatted context with citations.

        Returns:
            Complete prompt for LLM.
        """
        return f"""You are a scientific research assistant. Answer the user's question based ONLY on the provided context. Follow these rules:

1. Use inline citations [1], [2], etc. to reference your sources
2. If the context doesn't contain enough information, say so clearly
3. Be precise and scientific - avoid speculation
4. Synthesize information from multiple sources when relevant
5. Use direct quotes sparingly, preferring paraphrased summaries

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with citations):"""

    def _extract_citations(
        self,
        answer: str,
        citation_map: dict[int, RetrievalResult],
        context: list[RetrievalResult],
    ) -> list[Citation]:
        """Extract citations used in the answer.

        Args:
            answer: Generated answer with citation markers.
            citation_map: Mapping of citation numbers to chunks.
            context: Original context chunks.

        Returns:
            List of Citation objects.
        """
        # Find all citation numbers in the answer
        citation_pattern = re.compile(r"\[(\d+)\]")
        used_citations = set(int(m) for m in citation_pattern.findall(answer))

        citations = []
        for num in sorted(used_citations):
            if num in citation_map:
                chunk = citation_map[num]
                # Create excerpt (first 200 chars of content)
                excerpt = chunk.content[:200]
                if len(chunk.content) > 200:
                    excerpt += "..."

                citations.append(
                    Citation(
                        citation_id=num,
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        title=chunk.document_title or "Unknown",
                        excerpt=excerpt,
                        page=chunk.page_number,
                        confidence=chunk.score,
                    )
                )

        return citations
