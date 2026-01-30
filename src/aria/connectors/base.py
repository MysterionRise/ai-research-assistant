"""Base connector interface for literature APIs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LiteratureResult:
    """A literature search result from an external source."""

    id: str
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    url: str | None = None
    source: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base class for literature connectors.

    Connectors provide integration with external literature databases
    and APIs (PubMed, arXiv, Semantic Scholar, etc.).
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this source."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[LiteratureResult]:
        """Search for literature.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            **kwargs: Additional source-specific parameters.

        Returns:
            List of literature results.
        """
        pass

    @abstractmethod
    async def get_by_id(self, paper_id: str) -> LiteratureResult | None:
        """Get a paper by its ID.

        Args:
            paper_id: Paper identifier (PMID, arXiv ID, etc.).

        Returns:
            LiteratureResult or None if not found.
        """
        pass
