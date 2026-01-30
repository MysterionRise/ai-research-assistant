"""Keyword-based retrieval using BM25."""

import re
from collections import Counter
from math import log
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aria.db.models import Chunk, Document
from aria.db.session import async_session_maker
from aria.rag.retrieval.base import BaseRetriever, RetrievalResult

logger = structlog.get_logger(__name__)


class KeywordRetriever(BaseRetriever):
    """BM25-based keyword retriever.

    Uses BM25 algorithm for keyword matching, which is effective
    for exact term matching and domain-specific terminology.
    """

    # BM25 parameters
    K1 = 1.2  # Term frequency saturation
    B = 0.75  # Length normalization

    def __init__(self, session: AsyncSession | None = None) -> None:
        """Initialize keyword retriever.

        Args:
            session: Optional database session.
        """
        self._session = session
        logger.info("keyword_retriever_initialized")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using BM25 keyword matching.

        Args:
            query: Search query.
            top_k: Number of results to return.
            filters: Optional filters.

        Returns:
            List of retrieval results sorted by BM25 score.
        """
        logger.info("keyword_retrieval", query=query[:100], top_k=top_k)

        # Tokenize query
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        session = self._session or async_session_maker()

        try:
            # Build query
            stmt = (
                select(
                    Chunk.id,
                    Chunk.document_id,
                    Chunk.content,
                    Chunk.section,
                    Chunk.page_number,
                    Chunk.token_count,
                    Document.title.label("document_title"),
                )
                .join(Document, Chunk.document_id == Document.id)
            )

            # Apply filters
            if filters:
                if filters.get("document_id"):
                    stmt = stmt.where(Chunk.document_id == filters["document_id"])
                if filters.get("document_ids"):
                    stmt = stmt.where(Chunk.document_id.in_(filters["document_ids"]))

            result = await session.execute(stmt)
            rows = result.fetchall()

            if not rows:
                return []

            # Calculate BM25 scores
            scored_results = []
            avg_doc_len = sum(r.token_count for r in rows) / len(rows)

            # Calculate IDF for each query term
            doc_count = len(rows)
            term_doc_freq = Counter()
            for row in rows:
                content_terms = set(self._tokenize(row.content))
                for term in query_terms:
                    if term in content_terms:
                        term_doc_freq[term] += 1

            idfs = {}
            for term in query_terms:
                df = term_doc_freq.get(term, 0)
                idfs[term] = log((doc_count - df + 0.5) / (df + 0.5) + 1)

            # Score each document
            for row in rows:
                content_terms = self._tokenize(row.content)
                term_freq = Counter(content_terms)
                doc_len = row.token_count

                score = 0.0
                for term in query_terms:
                    tf = term_freq.get(term, 0)
                    if tf > 0:
                        idf = idfs[term]
                        tf_component = (
                            (tf * (self.K1 + 1))
                            / (tf + self.K1 * (1 - self.B + self.B * doc_len / avg_doc_len))
                        )
                        score += idf * tf_component

                if score > 0:
                    scored_results.append(
                        (
                            score,
                            RetrievalResult(
                                chunk_id=row.id,
                                document_id=row.document_id,
                                content=row.content,
                                score=score,
                                section=row.section,
                                page_number=row.page_number,
                                document_title=row.document_title,
                            ),
                        )
                    )

            # Sort by score and return top_k
            scored_results.sort(key=lambda x: x[0], reverse=True)
            results = [r for _, r in scored_results[:top_k]]

            # Normalize scores to 0-1 range
            if results:
                max_score = max(r.score for r in results)
                if max_score > 0:
                    for r in results:
                        r.score = r.score / max_score

            logger.info(
                "keyword_retrieval_completed",
                results_count=len(results),
            )

            return results

        finally:
            if not self._session:
                await session.close()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase terms.
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        # Filter short tokens and stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "to", "of", "in", "for", "on", "with", "at", "by",
                     "from", "as", "into", "through", "during", "before", "after",
                     "above", "below", "between", "under", "again", "further",
                     "then", "once", "here", "there", "when", "where", "why",
                     "how", "all", "each", "few", "more", "most", "other", "some",
                     "such", "no", "nor", "not", "only", "own", "same", "so",
                     "than", "too", "very", "just", "and", "but", "if", "or",
                     "because", "until", "while", "this", "that", "these", "those"}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]
