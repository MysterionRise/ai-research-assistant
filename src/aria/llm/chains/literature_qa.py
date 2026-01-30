"""LangGraph-based literature QA chain."""

from dataclasses import dataclass, field
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from aria.connectors.aggregator import LiteratureAggregator
from aria.rag.pipeline import RAGPipeline
from aria.types import Citation

logger = structlog.get_logger(__name__)


class LiteratureQAState(TypedDict):
    """State for literature QA chain."""

    query: str
    query_type: str  # "factual", "exploratory", "comparative"
    internal_results: list[dict]
    external_results: list[dict]
    answer: str
    citations: list[dict]
    confidence: float
    error: str | None


@dataclass
class LiteratureQAResult:
    """Result from literature QA chain."""

    answer: str
    citations: list[Citation]
    confidence: float
    sources_used: int
    internal_count: int
    external_count: int
    metadata: dict = field(default_factory=dict)


class LiteratureQAChain:
    """LangGraph chain for literature question answering.

    Implements a multi-step workflow:
    1. Classify query type
    2. Search internal knowledge base
    3. Search external sources (PubMed, arXiv, Semantic Scholar)
    4. Synthesize answer with citations
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
        literature_aggregator: LiteratureAggregator | None = None,
        include_external: bool = True,
    ) -> None:
        """Initialize the QA chain.

        Args:
            rag_pipeline: RAG pipeline for internal search.
            literature_aggregator: Aggregator for external search.
            include_external: Whether to search external sources.
        """
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.literature_aggregator = literature_aggregator or LiteratureAggregator()
        self.include_external = include_external

        # Build the graph
        self.graph = self._build_graph()

        logger.info(
            "literature_qa_chain_initialized",
            include_external=include_external,
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(LiteratureQAState)

        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("search_internal", self._search_internal)
        workflow.add_node("search_external", self._search_external)
        workflow.add_node("synthesize", self._synthesize)

        # Add edges
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "search_internal")

        # Conditional edge for external search
        workflow.add_conditional_edges(
            "search_internal",
            self._should_search_external,
            {
                "search_external": "search_external",
                "synthesize": "synthesize",
            },
        )

        workflow.add_edge("search_external", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _classify_query(
        self,
        state: LiteratureQAState,
    ) -> dict[str, Any]:
        """Classify the query type.

        Args:
            state: Current state.

        Returns:
            Updated state with query_type.
        """
        query = state["query"].lower()

        # Simple heuristic classification
        if any(word in query for word in ["compare", "difference", "versus", "vs"]):
            query_type = "comparative"
        elif any(word in query for word in ["what is", "define", "how does"]):
            query_type = "factual"
        else:
            query_type = "exploratory"

        logger.info("query_classified", query_type=query_type)

        return {"query_type": query_type}

    async def _search_internal(
        self,
        state: LiteratureQAState,
    ) -> dict[str, Any]:
        """Search internal knowledge base.

        Args:
            state: Current state.

        Returns:
            Updated state with internal_results.
        """
        logger.info("searching_internal")

        result = await self.rag_pipeline.query(state["query"])

        internal_results = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "score": chunk.score,
                "title": chunk.document_title,
                "section": chunk.section,
            }
            for chunk in result.reranked_chunks
        ]

        return {"internal_results": internal_results}

    def _should_search_external(self, state: LiteratureQAState) -> str:
        """Decide whether to search external sources.

        Args:
            state: Current state.

        Returns:
            Next node name.
        """
        if not self.include_external:
            return "synthesize"

        # Search external if internal results are insufficient
        internal_count = len(state.get("internal_results", []))
        if internal_count < 3:
            return "search_external"

        # Also search for exploratory queries
        if state.get("query_type") == "exploratory":
            return "search_external"

        return "synthesize"

    async def _search_external(
        self,
        state: LiteratureQAState,
    ) -> dict[str, Any]:
        """Search external literature sources.

        Args:
            state: Current state.

        Returns:
            Updated state with external_results.
        """
        logger.info("searching_external")

        results = await self.literature_aggregator.search(
            query=state["query"],
            limit=10,
        )

        external_results = [
            {
                "id": r.id,
                "title": r.title,
                "abstract": r.abstract,
                "authors": r.authors,
                "year": r.year,
                "source": r.source,
                "score": r.score,
                "doi": r.doi,
                "url": r.url,
            }
            for r in results
        ]

        return {"external_results": external_results}

    async def _synthesize(
        self,
        state: LiteratureQAState,
    ) -> dict[str, Any]:
        """Synthesize final answer.

        Args:
            state: Current state.

        Returns:
            Updated state with answer and citations.
        """
        logger.info("synthesizing_answer")

        # Use RAG pipeline's synthesis for internal results
        result = await self.rag_pipeline.query(state["query"])

        # Format citations
        citations = [
            {
                "citation_id": c.citation_id,
                "document_id": c.document_id,
                "title": c.title,
                "excerpt": c.excerpt,
                "page": c.page,
                "confidence": c.confidence,
            }
            for c in result.citations
        ]

        # Add external references to answer if available
        answer = result.answer
        external = state.get("external_results", [])

        if external:
            answer += "\n\n**Related Literature:**\n"
            for i, ext in enumerate(external[:5], 1):
                authors = ", ".join(ext["authors"][:2]) if ext["authors"] else "Unknown"
                year = f" ({ext['year']})" if ext["year"] else ""
                answer += f"- {ext['title']} - {authors}{year}\n"

        return {
            "answer": answer,
            "citations": citations,
            "confidence": result.confidence,
        }

    async def run(self, query: str) -> LiteratureQAResult:
        """Run the QA chain.

        Args:
            query: User's question.

        Returns:
            LiteratureQAResult with answer and citations.
        """
        logger.info("running_literature_qa_chain", query=query[:100])

        initial_state: LiteratureQAState = {
            "query": query,
            "query_type": "",
            "internal_results": [],
            "external_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
        }

        final_state = await self.graph.ainvoke(initial_state)

        # Convert citations to Citation objects
        citations = [
            Citation(
                citation_id=c["citation_id"],
                document_id=c["document_id"],
                chunk_id=None,
                title=c["title"],
                excerpt=c["excerpt"],
                page=c["page"],
                confidence=c["confidence"],
            )
            for c in final_state.get("citations", [])
        ]

        result = LiteratureQAResult(
            answer=final_state.get("answer", ""),
            citations=citations,
            confidence=final_state.get("confidence", 0.0),
            sources_used=len(citations),
            internal_count=len(final_state.get("internal_results", [])),
            external_count=len(final_state.get("external_results", [])),
        )

        logger.info(
            "literature_qa_chain_completed",
            confidence=result.confidence,
            sources_used=result.sources_used,
        )

        return result
