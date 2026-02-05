"""Unit tests for LLM prompts."""

from aria.llm.prompts import RAG_SYSTEM_PROMPT, SYSTEM_PROMPT, build_rag_prompt
from aria.llm.prompts.rag import (
    build_answer_verification_prompt,
    build_query_expansion_prompt,
)
from aria.llm.prompts.system import HALLUCINATION_GUARD_PROMPT, SCIENCE_ASSISTANT_PROMPT
from aria.rag.retrieval.base import RetrievalResult


class TestSystemPrompts:
    """Tests for system prompts."""

    def test_system_prompt_exists(self) -> None:
        """Test that SYSTEM_PROMPT is defined."""
        assert SYSTEM_PROMPT is not None
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_aria(self) -> None:
        """Test that SYSTEM_PROMPT mentions ARIA."""
        assert "ARIA" in SYSTEM_PROMPT

    def test_system_prompt_mentions_citations(self) -> None:
        """Test that SYSTEM_PROMPT mentions citation format."""
        assert "[1]" in SYSTEM_PROMPT or "citation" in SYSTEM_PROMPT.lower()

    def test_science_assistant_prompt_exists(self) -> None:
        """Test that SCIENCE_ASSISTANT_PROMPT is defined."""
        assert SCIENCE_ASSISTANT_PROMPT is not None
        assert isinstance(SCIENCE_ASSISTANT_PROMPT, str)

    def test_hallucination_guard_prompt_exists(self) -> None:
        """Test that HALLUCINATION_GUARD_PROMPT is defined."""
        assert HALLUCINATION_GUARD_PROMPT is not None
        assert isinstance(HALLUCINATION_GUARD_PROMPT, str)
        # Should mention context
        assert "context" in HALLUCINATION_GUARD_PROMPT.lower()


class TestRAGSystemPrompt:
    """Tests for RAG system prompt."""

    def test_rag_system_prompt_exists(self) -> None:
        """Test that RAG_SYSTEM_PROMPT is defined."""
        assert RAG_SYSTEM_PROMPT is not None
        assert isinstance(RAG_SYSTEM_PROMPT, str)

    def test_rag_prompt_mentions_citations(self) -> None:
        """Test that RAG prompt mentions citation format."""
        assert "[1]" in RAG_SYSTEM_PROMPT
        assert "[2]" in RAG_SYSTEM_PROMPT

    def test_rag_prompt_warns_against_hallucination(self) -> None:
        """Test that RAG prompt warns against making up facts."""
        assert "make up" in RAG_SYSTEM_PROMPT.lower() or "never" in RAG_SYSTEM_PROMPT.lower()


class TestBuildRAGPrompt:
    """Tests for build_rag_prompt function."""

    def test_build_rag_prompt_basic(self) -> None:
        """Test building RAG prompt with basic inputs."""
        context = [
            RetrievalResult(
                chunk_id="c1",
                document_id="d1",
                content="This is test content about materials.",
                score=0.9,
                document_title="Test Paper",
            ),
        ]

        prompt = build_rag_prompt("What is the topic?", context)

        assert "What is the topic?" in prompt
        assert "Test Paper" in prompt
        assert "materials" in prompt
        assert "[1]" in prompt

    def test_build_rag_prompt_multiple_chunks(self) -> None:
        """Test building RAG prompt with multiple chunks."""
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
            ),
        ]

        prompt = build_rag_prompt("Question?", context)

        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Paper 1" in prompt
        assert "Paper 2" in prompt

    def test_build_rag_prompt_with_section(self) -> None:
        """Test building RAG prompt with section metadata."""
        context = [
            RetrievalResult(
                chunk_id="c1",
                document_id="d1",
                content="Content here.",
                score=0.9,
                document_title="Paper",
                section="Methods",
            ),
        ]

        prompt = build_rag_prompt("Query?", context)

        assert "Methods" in prompt

    def test_build_rag_prompt_with_page_number(self) -> None:
        """Test building RAG prompt with page number."""
        context = [
            RetrievalResult(
                chunk_id="c1",
                document_id="d1",
                content="Content here.",
                score=0.9,
                document_title="Paper",
                page_number=42,
            ),
        ]

        prompt = build_rag_prompt("Query?", context)

        assert "42" in prompt
        assert "p." in prompt

    def test_build_rag_prompt_with_conversation_history(self) -> None:
        """Test building RAG prompt with conversation history."""
        context = [
            RetrievalResult(
                chunk_id="c1",
                document_id="d1",
                content="Content.",
                score=0.9,
            ),
        ]
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        prompt = build_rag_prompt("New question?", context, history)

        assert "Hello" in prompt
        assert "Hi there!" in prompt
        assert "CONVERSATION HISTORY" in prompt

    def test_build_rag_prompt_without_title(self) -> None:
        """Test building RAG prompt when document title is missing."""
        context = [
            RetrievalResult(
                chunk_id="c1",
                document_id="d1",
                content="Content.",
                score=0.9,
                document_title=None,
            ),
        ]

        prompt = build_rag_prompt("Query?", context)

        # Should use "Document" as fallback
        assert "Document" in prompt


class TestBuildQueryExpansionPrompt:
    """Tests for build_query_expansion_prompt function."""

    def test_build_query_expansion_prompt(self) -> None:
        """Test building query expansion prompt."""
        prompt = build_query_expansion_prompt("cancer treatment")

        assert "cancer treatment" in prompt
        assert "scientific" in prompt.lower() or "terminology" in prompt.lower()
        assert "alternative" in prompt.lower()

    def test_query_expansion_prompt_requests_multiple(self) -> None:
        """Test that query expansion requests multiple alternatives."""
        prompt = build_query_expansion_prompt("test query")

        assert "3" in prompt  # Should request 3 alternatives


class TestBuildAnswerVerificationPrompt:
    """Tests for build_answer_verification_prompt function."""

    def test_build_answer_verification_prompt(self) -> None:
        """Test building answer verification prompt."""
        prompt = build_answer_verification_prompt(
            query="What is X?",
            answer="X is Y based on [1].",
            context="Source says X is Y.",
        )

        assert "What is X?" in prompt
        assert "X is Y based on [1]" in prompt
        assert "Source says X is Y" in prompt

    def test_verification_prompt_checks_accuracy(self) -> None:
        """Test that verification prompt checks for accuracy."""
        prompt = build_answer_verification_prompt(
            query="Q",
            answer="A",
            context="C",
        )

        assert "accuracy" in prompt.lower() or "faithful" in prompt.lower()

    def test_verification_prompt_checks_citations(self) -> None:
        """Test that verification prompt checks citations."""
        prompt = build_answer_verification_prompt(
            query="Q",
            answer="A",
            context="C",
        )

        assert "citation" in prompt.lower()

    def test_verification_prompt_requests_score(self) -> None:
        """Test that verification prompt requests a score."""
        prompt = build_answer_verification_prompt(
            query="Q",
            answer="A",
            context="C",
        )

        # Should request a 0-1 score
        assert "0" in prompt and "1" in prompt


class TestPromptsModule:
    """Tests for prompts module exports."""

    def test_module_exports_rag_system_prompt(self) -> None:
        """Test that module exports RAG_SYSTEM_PROMPT."""
        from aria.llm.prompts import RAG_SYSTEM_PROMPT

        assert RAG_SYSTEM_PROMPT is not None

    def test_module_exports_system_prompt(self) -> None:
        """Test that module exports SYSTEM_PROMPT."""
        from aria.llm.prompts import SYSTEM_PROMPT

        assert SYSTEM_PROMPT is not None

    def test_module_exports_build_rag_prompt(self) -> None:
        """Test that module exports build_rag_prompt."""
        from aria.llm.prompts import build_rag_prompt

        assert callable(build_rag_prompt)
