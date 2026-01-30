"""Storage layer for ARIA.

Provides vector storage and file storage capabilities.
"""

from aria.storage.vector.pgvector import PgVectorStore

__all__ = ["PgVectorStore"]
