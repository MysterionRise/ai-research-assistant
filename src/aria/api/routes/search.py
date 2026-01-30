"""Search endpoints.

Provides literature and molecular search capabilities.
"""

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from aria.api.dependencies import EmbedderDep, LiteratureAggregatorDep, VectorStoreDep

router = APIRouter(prefix="/search")
logger = structlog.get_logger(__name__)


# =========================
# Request/Response Models
# =========================


class SearchFilters(BaseModel):
    """Filters for search queries."""

    year_from: int | None = Field(default=None, description="Start year filter")
    year_to: int | None = Field(default=None, description="End year filter")
    journals: list[str] | None = Field(default=None, description="Journal filter")
    authors: list[str] | None = Field(default=None, description="Author filter")
    document_types: list[str] | None = Field(
        default=None,
        description="Document type filter (paper, patent, report)",
    )


class LiteratureSearchRequest(BaseModel):
    """Request for literature search."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    filters: SearchFilters | None = Field(default=None, description="Search filters")
    limit: int = Field(default=20, ge=1, le=100, description="Max results")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")
    sources: list[str] = Field(
        default=["pubmed", "semantic_scholar", "internal"],
        description="Data sources to search",
    )


class SearchResult(BaseModel):
    """A single search result."""

    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    abstract: str | None = Field(default=None, description="Abstract or summary")
    authors: list[str] = Field(default_factory=list, description="Author list")
    year: int | None = Field(default=None, description="Publication year")
    journal: str | None = Field(default=None, description="Journal or source")
    doi: str | None = Field(default=None, description="DOI if available")
    url: str | None = Field(default=None, description="Source URL")
    source: str = Field(..., description="Data source (pubmed, internal, etc.)")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class LiteratureSearchResponse(BaseModel):
    """Response from literature search."""

    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total matching results")
    results: list[SearchResult] = Field(..., description="Search results")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Search metadata")


class MolecularSearchRequest(BaseModel):
    """Request for molecular similarity search."""

    smiles: str | None = Field(default=None, description="SMILES string to search")
    name: str | None = Field(default=None, description="Molecule name to search")
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max results")


class MoleculeResult(BaseModel):
    """A single molecule search result."""

    id: str = Field(..., description="Molecule ID")
    name: str = Field(..., description="Molecule name")
    smiles: str = Field(..., description="SMILES representation")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Computed properties",
    )


class MolecularSearchResponse(BaseModel):
    """Response from molecular search."""

    query_smiles: str = Field(..., description="Query SMILES")
    results: list[MoleculeResult] = Field(..., description="Similar molecules")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Search metadata")


# =========================
# Endpoints
# =========================


@router.post(
    "/literature",
    response_model=LiteratureSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search scientific literature",
    description="Search across PubMed, Semantic Scholar, and internal documents.",
)
async def search_literature(
    request: LiteratureSearchRequest,
    aggregator: LiteratureAggregatorDep,
    vector_store: VectorStoreDep,
    embedder: EmbedderDep,
) -> LiteratureSearchResponse:
    """Search scientific literature.

    Args:
        request: Search request with query and filters.
        aggregator: Literature aggregator for external search.
        vector_store: Vector store for internal search.
        embedder: Embedder for semantic search.

    Returns:
        LiteratureSearchResponse: Search results with relevance scores.
    """
    start_time = time.time()
    logger.info("literature_search", query=request.query, sources=request.sources)

    results: list[SearchResult] = []

    # Search internal documents if requested
    if "internal" in request.sources:
        query_embedding = await embedder.embed(request.query)
        internal_results = await vector_store.search_with_document_info(
            query_embedding=query_embedding,
            top_k=request.limit,
        )

        for r in internal_results:
            results.append(
                SearchResult(
                    id=r.chunk_id,
                    title=(
                        r.metadata.get("document_title", "Internal Document")
                        if r.metadata
                        else "Internal Document"
                    ),
                    abstract=r.content[:500] if r.content else None,
                    authors=[],
                    year=None,
                    journal=None,
                    doi=None,
                    url=None,
                    source="internal",
                    relevance_score=r.score,
                )
            )

    # Search external sources
    external_sources = [s for s in request.sources if s != "internal"]
    if external_sources:
        kwargs = {}
        if request.filters:
            if request.filters.year_from:
                kwargs["year_from"] = request.filters.year_from
            if request.filters.year_to:
                kwargs["year_to"] = request.filters.year_to

        external_results = await aggregator.search(
            query=request.query,
            limit=request.limit,
            sources=external_sources,
            **kwargs,
        )

        for r in external_results:
            results.append(
                SearchResult(
                    id=r.id,
                    title=r.title,
                    abstract=r.abstract,
                    authors=r.authors,
                    year=r.year,
                    journal=r.journal,
                    doi=r.doi,
                    url=r.url,
                    source=r.source,
                    relevance_score=r.score,
                )
            )

    # Sort by relevance
    results.sort(key=lambda x: x.relevance_score, reverse=True)

    # Apply pagination
    paginated = results[request.offset : request.offset + request.limit]

    search_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        "literature_search_completed",
        query=request.query,
        results_count=len(paginated),
        search_time_ms=search_time_ms,
    )

    return LiteratureSearchResponse(
        query=request.query,
        total_results=len(results),
        results=paginated,
        metadata={
            "sources_queried": request.sources,
            "search_time_ms": search_time_ms,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
    )


@router.get(
    "/literature/quick",
    response_model=LiteratureSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Quick literature search",
    description="Simple GET endpoint for quick searches.",
)
async def quick_search_literature(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Max results"),
    aggregator: LiteratureAggregatorDep = None,  # type: ignore[assignment]
    vector_store: VectorStoreDep = None,  # type: ignore[assignment]
    embedder: EmbedderDep = None,  # type: ignore[assignment]
) -> LiteratureSearchResponse:
    """Quick literature search via GET.

    Args:
        q: Search query string.
        limit: Maximum results to return.
        aggregator: Literature aggregator.
        vector_store: Vector store.
        embedder: Embedder.

    Returns:
        LiteratureSearchResponse: Search results.
    """
    request = LiteratureSearchRequest(query=q, limit=limit)
    return await search_literature(request, aggregator, vector_store, embedder)


@router.post(
    "/molecules",
    response_model=MolecularSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search similar molecules",
    description="Find molecules similar to a query structure.",
)
async def search_molecules(request: MolecularSearchRequest) -> MolecularSearchResponse:
    """Search for similar molecules.

    Args:
        request: Molecular search request with SMILES or name.

    Returns:
        MolecularSearchResponse: Similar molecules with similarity scores.
    """
    # TODO: Implement molecular similarity search with ChemBERTa
    # For now, return placeholder response

    query_smiles = request.smiles or ""

    logger.info("molecular_search", smiles=query_smiles[:50])

    return MolecularSearchResponse(
        query_smiles=query_smiles,
        results=[],
        metadata={
            "similarity_threshold": request.similarity_threshold,
            "search_time_ms": 0,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "note": "Molecular search not yet implemented",
        },
    )


@router.get(
    "/suggestions",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    summary="Get search suggestions",
    description="Get autocomplete suggestions for search queries.",
)
async def get_search_suggestions(
    q: str = Query(  # noqa: ARG001
        ..., min_length=2, max_length=100, description="Partial query"
    ),
    limit: int = Query(  # noqa: ARG001
        default=5, ge=1, le=20, description="Max suggestions"
    ),
) -> list[str]:
    """Get search suggestions.

    Args:
        q: Partial query string (not yet implemented).
        limit: Maximum suggestions to return (not yet implemented).

    Returns:
        List of suggested queries.
    """
    # TODO: Implement search suggestions from index using q and limit
    # For now, return empty list
    return []
