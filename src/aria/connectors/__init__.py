"""External literature connectors for ARIA.

Provides integration with PubMed, arXiv, and Semantic Scholar APIs.
"""

from aria.connectors.aggregator import LiteratureAggregator
from aria.connectors.arxiv import ArxivConnector
from aria.connectors.base import BaseConnector, LiteratureResult
from aria.connectors.pubmed import PubMedConnector
from aria.connectors.semantic_scholar import SemanticScholarConnector

__all__ = [
    "ArxivConnector",
    "BaseConnector",
    "LiteratureAggregator",
    "LiteratureResult",
    "PubMedConnector",
    "SemanticScholarConnector",
]
