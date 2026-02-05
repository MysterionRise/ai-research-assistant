"""RAG-specific prompts for ARIA."""

from aria.rag.retrieval.base import RetrievalResult

RAG_SYSTEM_PROMPT = """You are ARIA, a scientific research assistant. Answer questions using ONLY the provided context.

CRITICAL RULES:
1. Use inline citations [1], [2], etc. to reference sources
2. If the context doesn't contain the answer, say "I don't have enough information"
3. Never make up facts or citations
4. Be precise and scientific
5. Synthesize information from multiple sources when relevant

For scientific claims:
- Quote specific data points when available
- Note confidence levels or uncertainty if mentioned
- Distinguish between established facts and hypotheses"""


def build_rag_prompt(
    query: str,
    context: list[RetrievalResult],
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Build a RAG prompt with context and optional history.

    Args:
        query: User's question.
        context: Retrieved context chunks.
        conversation_history: Optional previous messages.

    Returns:
        Formatted prompt string.
    """
    # Format context with citation markers
    context_parts = []
    for i, chunk in enumerate(context, 1):
        title = chunk.document_title or "Document"
        section = f" - {chunk.section}" if chunk.section else ""
        page = f" (p. {chunk.page_number})" if chunk.page_number else ""

        context_parts.append(f"[{i}] {title}{section}{page}:\n{chunk.content}\n")

    formatted_context = "\n".join(context_parts)

    # Build prompt
    if conversation_history:
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-4:]  # Last 4 messages
        )
        prompt = f"""CONVERSATION HISTORY:
{history_text}

CONTEXT:
{formatted_context}

CURRENT QUESTION: {query}

Answer the current question using the context. Use [1], [2] citations."""
    else:
        prompt = f"""CONTEXT:
{formatted_context}

QUESTION: {query}

Answer using the context above. Use [1], [2] citations to reference sources."""

    return prompt


def build_query_expansion_prompt(query: str) -> str:
    """Build prompt for query expansion/rewriting.

    Args:
        query: Original user query.

    Returns:
        Prompt for query expansion.
    """
    return f"""Rewrite this search query to improve retrieval from scientific literature.

Original query: {query}

Generate 3 alternative phrasings that:
1. Use scientific terminology
2. Include relevant synonyms
3. Are more specific if the original is vague

Return only the alternative queries, one per line."""


def build_answer_verification_prompt(
    query: str,
    answer: str,
    context: str,
) -> str:
    """Build prompt for answer verification.

    Args:
        query: Original question.
        answer: Generated answer to verify.
        context: Source context used.

    Returns:
        Prompt for verification.
    """
    return f"""Verify this answer against the source context.

QUESTION: {query}

ANSWER: {answer}

CONTEXT: {context}

Check for:
1. Factual accuracy - does the answer match the context?
2. Citation correctness - are citations used properly?
3. Unsupported claims - is anything stated without source support?
4. Missing information - are key context points omitted?

Rate the answer's faithfulness from 0.0 to 1.0 and explain any issues."""
