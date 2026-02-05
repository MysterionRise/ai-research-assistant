"""Unit tests for vector store base classes."""

from typing import Any

import pytest

from aria.storage.vector.base import BaseVectorStore, VectorSearchResult
from aria.types import Embedding


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_create_vector_search_result(self) -> None:
        """Test creating VectorSearchResult with required fields."""
        result = VectorSearchResult(
            chunk_id="chunk-123",
            document_id="doc-456",
            content="This is the content of the chunk.",
            score=0.92,
        )

        assert result.chunk_id == "chunk-123"
        assert result.document_id == "doc-456"
        assert result.content == "This is the content of the chunk."
        assert result.score == 0.92

    def test_vector_search_result_with_optional_fields(self) -> None:
        """Test VectorSearchResult with all fields."""
        result = VectorSearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Content here",
            score=0.88,
            section="Methods",
            page_number=15,
            metadata={"embedding_model": "text-embedding-3-large"},
        )

        assert result.section == "Methods"
        assert result.page_number == 15
        assert result.metadata == {"embedding_model": "text-embedding-3-large"}

    def test_vector_search_result_defaults(self) -> None:
        """Test VectorSearchResult default values."""
        result = VectorSearchResult(
            chunk_id="c1",
            document_id="d1",
            content="Content",
            score=0.5,
        )

        assert result.section is None
        assert result.page_number is None
        assert result.metadata is None

    def test_vector_search_result_score_can_be_float(self) -> None:
        """Test that score can be any float value."""
        # High score
        result_high = VectorSearchResult(
            chunk_id="c1",
            document_id="d1",
            content="High relevance",
            score=0.99,
        )
        assert result_high.score == 0.99

        # Low score
        result_low = VectorSearchResult(
            chunk_id="c2",
            document_id="d2",
            content="Low relevance",
            score=0.01,
        )
        assert result_low.score == 0.01


class TestBaseVectorStore:
    """Tests for BaseVectorStore abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseVectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_subclass_must_implement_search(self) -> None:
        """Test that subclass must implement search method."""

        class IncompleteStore(BaseVectorStore):
            async def insert(
                self,
                chunk_id: str,
                document_id: str,
                content: str,
                embedding: Embedding,
                metadata: dict | None = None,
            ) -> None:
                pass

            async def delete(self, chunk_id: str) -> None:
                pass

            async def delete_by_document(self, document_id: str) -> int:
                return 0

        with pytest.raises(TypeError):
            IncompleteStore()

    def test_subclass_must_implement_insert(self) -> None:
        """Test that subclass must implement insert method."""

        class IncompleteStore(BaseVectorStore):
            async def search(
                self,
                query_embedding: Embedding,
                top_k: int = 10,
                filters: dict[str, Any] | None = None,
                min_score: float = 0.0,
            ) -> list[VectorSearchResult]:
                return []

            async def delete(self, chunk_id: str) -> None:
                pass

            async def delete_by_document(self, document_id: str) -> int:
                return 0

        with pytest.raises(TypeError):
            IncompleteStore()

    def test_subclass_must_implement_delete(self) -> None:
        """Test that subclass must implement delete method."""

        class IncompleteStore(BaseVectorStore):
            async def search(
                self,
                query_embedding: Embedding,
                top_k: int = 10,
                filters: dict[str, Any] | None = None,
                min_score: float = 0.0,
            ) -> list[VectorSearchResult]:
                return []

            async def insert(
                self,
                chunk_id: str,
                document_id: str,
                content: str,
                embedding: Embedding,
                metadata: dict | None = None,
            ) -> None:
                pass

            async def delete_by_document(self, document_id: str) -> int:
                return 0

        with pytest.raises(TypeError):
            IncompleteStore()

    def test_subclass_must_implement_delete_by_document(self) -> None:
        """Test that subclass must implement delete_by_document method."""

        class IncompleteStore(BaseVectorStore):
            async def search(
                self,
                query_embedding: Embedding,
                top_k: int = 10,
                filters: dict[str, Any] | None = None,
                min_score: float = 0.0,
            ) -> list[VectorSearchResult]:
                return []

            async def insert(
                self,
                chunk_id: str,
                document_id: str,
                content: str,
                embedding: Embedding,
                metadata: dict | None = None,
            ) -> None:
                pass

            async def delete(self, chunk_id: str) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteStore()

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """Test that complete implementation can be instantiated."""

        class CompleteStore(BaseVectorStore):
            async def search(
                self,
                query_embedding: Embedding,
                top_k: int = 10,
                filters: dict[str, Any] | None = None,
                min_score: float = 0.0,
            ) -> list[VectorSearchResult]:
                return []

            async def insert(
                self,
                chunk_id: str,
                document_id: str,
                content: str,
                embedding: Embedding,
                metadata: dict | None = None,
            ) -> None:
                pass

            async def delete(self, chunk_id: str) -> None:
                pass

            async def delete_by_document(self, document_id: str) -> int:
                return 0

        store = CompleteStore()
        assert isinstance(store, BaseVectorStore)


class TestMockVectorStore:
    """Tests using a mock vector store implementation."""

    class MockVectorStore(BaseVectorStore):
        """Mock implementation for testing."""

        def __init__(self) -> None:
            self.vectors: dict[str, dict] = {}

        async def search(
            self,
            query_embedding: Embedding,
            top_k: int = 10,
            filters: dict[str, Any] | None = None,
            min_score: float = 0.0,
        ) -> list[VectorSearchResult]:
            # Simple mock: return all stored vectors up to top_k
            results = []
            for chunk_id, data in list(self.vectors.items())[:top_k]:
                results.append(
                    VectorSearchResult(
                        chunk_id=chunk_id,
                        document_id=data["document_id"],
                        content=data["content"],
                        score=0.9,
                    )
                )
            return results

        async def insert(
            self,
            chunk_id: str,
            document_id: str,
            content: str,
            embedding: Embedding,
            metadata: dict | None = None,
        ) -> None:
            self.vectors[chunk_id] = {
                "document_id": document_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {},
            }

        async def delete(self, chunk_id: str) -> None:
            if chunk_id in self.vectors:
                del self.vectors[chunk_id]

        async def delete_by_document(self, document_id: str) -> int:
            to_delete = [
                cid for cid, data in self.vectors.items() if data["document_id"] == document_id
            ]
            for cid in to_delete:
                del self.vectors[cid]
            return len(to_delete)

    @pytest.fixture
    def store(self) -> "TestMockVectorStore.MockVectorStore":
        """Create mock store."""
        return self.MockVectorStore()

    @pytest.mark.asyncio
    async def test_insert_and_search(self, store: "TestMockVectorStore.MockVectorStore") -> None:
        """Test inserting and searching vectors."""
        embedding = [0.1, 0.2, 0.3]
        await store.insert(
            chunk_id="c1",
            document_id="d1",
            content="Test content",
            embedding=embedding,
        )

        results = await store.search(query_embedding=embedding)

        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_delete_chunk(self, store: "TestMockVectorStore.MockVectorStore") -> None:
        """Test deleting a chunk."""
        await store.insert(
            chunk_id="c1",
            document_id="d1",
            content="Content",
            embedding=[0.1],
        )
        assert "c1" in store.vectors

        await store.delete("c1")
        assert "c1" not in store.vectors

    @pytest.mark.asyncio
    async def test_delete_by_document(self, store: "TestMockVectorStore.MockVectorStore") -> None:
        """Test deleting all chunks for a document."""
        await store.insert("c1", "d1", "Content 1", [0.1])
        await store.insert("c2", "d1", "Content 2", [0.2])
        await store.insert("c3", "d2", "Content 3", [0.3])

        deleted = await store.delete_by_document("d1")

        assert deleted == 2
        assert "c1" not in store.vectors
        assert "c2" not in store.vectors
        assert "c3" in store.vectors

    @pytest.mark.asyncio
    async def test_search_with_top_k(self, store: "TestMockVectorStore.MockVectorStore") -> None:
        """Test search respects top_k limit."""
        for i in range(10):
            await store.insert(f"c{i}", f"d{i}", f"Content {i}", [float(i)])

        results = await store.search(query_embedding=[0.5], top_k=3)
        assert len(results) == 3
