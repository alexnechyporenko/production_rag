"""Unit tests for the embedding provider + storage layer."""

from __future__ import annotations

import pytest
from app.domain import DocumentChunk
from app.embeddings import MockEmbeddingProvider
from app.storage import tokenize

# ============================================================================
# Embeddings
# ============================================================================


class TestMockEmbeddingProvider:
    def test_deterministic(self, embedder: MockEmbeddingProvider):
        a = embedder.embed_texts(["hello world"])[0]
        b = embedder.embed_texts(["hello world"])[0]
        assert a == b

    def test_dimension(self, embedder: MockEmbeddingProvider):
        v = embedder.embed_texts(["foo"])[0]
        assert len(v) == embedder.dimension == 64

    def test_normalized(self, embedder: MockEmbeddingProvider):
        import math

        v = embedder.embed_texts(["some meaningful text content here"])[0]
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-5

    def test_different_text_different_vector(self, embedder: MockEmbeddingProvider):
        a = embedder.embed_texts(["apples"])[0]
        b = embedder.embed_texts(["oranges"])[0]
        assert a != b

    def test_batch(self, embedder: MockEmbeddingProvider):
        out = embedder.embed_texts(["a", "b", "c"])
        assert len(out) == 3
        assert all(len(v) == 64 for v in out)


# ============================================================================
# In-memory vector store
# ============================================================================


def _chunk(i: int, content: str, document_id: str = "doc-1") -> DocumentChunk:
    return DocumentChunk(
        id=f"chunk-{i}",
        document_id=document_id,
        content=content,
        sequence=i,
        char_start=0,
        char_end=len(content),
    )


class TestInMemoryVectorStore:
    def test_add_and_count(self, in_memory_store, embedder):
        chunks = [_chunk(i, f"content {i}") for i in range(3)]
        vecs = embedder.embed_texts([c.content for c in chunks])
        added = in_memory_store.add(chunks, vecs, model=embedder.model)
        assert added == 3
        assert in_memory_store.count() == 3

    def test_dimension_mismatch_raises(self, in_memory_store):
        with pytest.raises(ValueError):
            in_memory_store.add([_chunk(0, "x")], [[0.1] * 32], model="m")

    def test_search_returns_top_k(self, in_memory_store, embedder):
        chunks = [
            _chunk(0, "PostgreSQL database pgvector"),
            _chunk(1, "FastAPI Python web framework"),
            _chunk(2, "Hybrid search dense lexical"),
        ]
        vecs = embedder.embed_texts([c.content for c in chunks])
        in_memory_store.add(chunks, vecs, model=embedder.model)
        q = embedder.embed_texts(["pgvector database"])[0]
        results = in_memory_store.search(q, top_k=2)
        assert len(results) == 2
        assert results[0].rank == 0
        assert results[0].retrieval_method == "dense"

    def test_search_filters_by_document_id(self, in_memory_store, embedder):
        chunks = [_chunk(0, "alpha", "doc-a"), _chunk(1, "beta", "doc-b")]
        vecs = embedder.embed_texts([c.content for c in chunks])
        in_memory_store.add(chunks, vecs, model=embedder.model)
        q = embedder.embed_texts(["alpha"])[0]
        r = in_memory_store.search(q, top_k=10, filters={"document_id": "doc-b"})
        assert all(c.document_id == "doc-b" for c in r)

    def test_min_similarity_threshold(self, in_memory_store, embedder):
        chunks = [_chunk(0, "alpha"), _chunk(1, "totally different zzz")]
        vecs = embedder.embed_texts([c.content for c in chunks])
        in_memory_store.add(chunks, vecs, model=embedder.model)
        q = embedder.embed_texts(["alpha"])[0]
        r = in_memory_store.search(q, top_k=10, min_similarity=0.99)
        # The mock embedder is sparse, so most candidates should be filtered.
        assert all(c.score >= 0.99 for c in r)

    def test_get_returns_chunk(self, in_memory_store, embedder):
        c = _chunk(0, "hello")
        v = embedder.embed_texts(["hello"])[0]
        in_memory_store.add([c], [v], model=embedder.model)
        got = in_memory_store.get("chunk-0")
        assert got is not None
        assert got.content == "hello"
        assert in_memory_store.get("missing") is None

    def test_delete_document(self, in_memory_store, embedder):
        chunks = [_chunk(0, "a", "doc-1"), _chunk(1, "b", "doc-1"), _chunk(2, "c", "doc-2")]
        vecs = embedder.embed_texts([c.content for c in chunks])
        in_memory_store.add(chunks, vecs, model=embedder.model)
        n = in_memory_store.delete_document("doc-1")
        assert n == 2
        assert in_memory_store.count() == 1

    def test_list_documents(self, in_memory_store, embedder):
        chunks = [_chunk(0, "a", "doc-1"), _chunk(1, "b", "doc-2")]
        vecs = embedder.embed_texts([c.content for c in chunks])
        in_memory_store.add(chunks, vecs, model=embedder.model)
        docs = in_memory_store.list_documents()
        ids = {d["document_id"] for d in docs}
        assert ids == {"doc-1", "doc-2"}

    def test_reset(self, in_memory_store, embedder):
        in_memory_store.add([_chunk(0, "a")], embedder.embed_texts(["a"]), model="m")
        in_memory_store.reset()
        assert in_memory_store.count() == 0

    def test_upsert_does_not_duplicate_doc_index(self, in_memory_store, embedder):
        """Re-ingestion with the same chunk_id must NOT append a duplicate
        reference to `_doc_to_chunks` — that would corrupt `list_documents()`
        and `count()` semantics (spec §06 AC-06.3).
        """
        c = _chunk(0, "alpha", "doc-1")
        v = embedder.embed_texts(["alpha"])[0]
        # Ingest twice
        in_memory_store.add([c], [v], model="m")
        in_memory_store.add([c], [v], model="m")
        # count must still be 1, not 2
        assert in_memory_store.count() == 1
        # list_documents must report exactly 1 chunk for doc-1
        docs = in_memory_store.list_documents()
        assert len(docs) == 1
        assert docs[0]["chunk_count"] == 1
        assert docs[0]["document_id"] == "doc-1"

    def test_upsert_with_new_content_updates_chunk(self, in_memory_store, embedder):
        c1 = _chunk(0, "alpha", "doc-1")
        c2 = DocumentChunk(
            id="chunk-0", document_id="doc-1", content="beta",
            sequence=0, char_start=0, char_end=4,
        )
        v1 = embedder.embed_texts(["alpha"])[0]
        v2 = embedder.embed_texts(["beta"])[0]
        in_memory_store.add([c1], [v1], model="m")
        in_memory_store.add([c2], [v2], model="m")
        got = in_memory_store.get("chunk-0")
        assert got is not None
        assert got.content == "beta"
        assert in_memory_store.count() == 1


# ============================================================================
# Lexical index (BM25)
# ============================================================================


class TestLexicalIndex:
    def test_empty_returns_empty(self, lexical_index):
        r = lexical_index.search("anything", top_k=5)
        assert r == []

    def test_basic_retrieval(self, lexical_index):
        chunks = [
            _chunk(0, "PostgreSQL is a database"),
            _chunk(1, "FastAPI is a framework"),
            _chunk(2, "Hybrid search combines dense and lexical"),
        ]
        lexical_index.add(chunks)
        r = lexical_index.search("PostgreSQL database", top_k=2)
        assert r[0].chunk_id == "chunk-0"
        assert r[0].retrieval_method == "lexical"

    def test_count(self, lexical_index):
        lexical_index.add([_chunk(0, "alpha"), _chunk(1, "beta")])
        assert lexical_index.count() == 2

    def test_reset(self, lexical_index):
        lexical_index.add([_chunk(0, "alpha")])
        lexical_index.reset()
        assert lexical_index.count() == 0

    def test_upsert_does_not_duplicate(self, lexical_index):
        """Re-ingestion of the same chunk_id must NOT duplicate entries —
        the index must update in place (spec §06 AC-06.3).
        """
        c = _chunk(0, "alpha")
        lexical_index.add([c])
        lexical_index.add([c])  # same id, same content
        assert lexical_index.count() == 1

    def test_upsert_with_new_content_replaces(self, lexical_index):
        # Seed with a few unrelated chunks so BM25 has reasonable IDF statistics.
        seed = [_chunk(i, f"other chunk {i}") for i in range(3)]
        lexical_index.add(seed)
        c1 = _chunk(99, "alpha")
        c2 = _chunk(99, "completely different content")
        lexical_index.add([c1])
        lexical_index.add([c2])
        assert lexical_index.count() == 4
        r = lexical_index.search("completely different", top_k=3)
        # The index should reflect the updated content, not the old one.
        assert any(c.chunk_id == "chunk-99" for c in r)
        match = next(c for c in r if c.chunk_id == "chunk-99")
        assert "completely different" in match.content

    def test_remove(self, lexical_index):
        lexical_index.add([_chunk(0, "alpha"), _chunk(1, "beta")])
        n = lexical_index.remove(["chunk-0"])
        assert n == 1
        assert lexical_index.count() == 1
        r = lexical_index.search("alpha", top_k=1)
        assert r == []

    def test_remove_missing_returns_zero(self, lexical_index):
        n = lexical_index.remove(["nonexistent"])
        assert n == 0


def test_tokenize():
    assert tokenize("Hello, World!") == ["hello", "world"]


class TestVectorStoreFactory:
    """The factory must NEVER silently fall back to in-memory in production
    mode — that would put production data in RAM and lose it on restart."""

    def test_sqlite_url_uses_in_memory(self, monkeypatch):
        from app.config import reset_settings_cache
        from app.storage import InMemoryVectorStore, get_vector_store

        reset_settings_cache()
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "32")
        store = get_vector_store()
        assert isinstance(store, InMemoryVectorStore)

    def test_force_in_memory_store_overrides_postgres(self, monkeypatch):
        from app.config import reset_settings_cache
        from app.storage import InMemoryVectorStore, get_vector_store

        reset_settings_cache()
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:y@localhost:5432/z")
        monkeypatch.setenv("FORCE_IN_MEMORY_STORE", "true")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "32")
        store = get_vector_store()
        assert isinstance(store, InMemoryVectorStore)

    def test_postgres_url_without_force_raises(self, monkeypatch):
        """Production mode: factory must raise VectorStoreError on failure,
        NEVER silently fall back to in-memory."""
        from app.config import reset_settings_cache
        from app.domain import VectorStoreError
        from app.storage import get_vector_store

        reset_settings_cache()
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:y@localhost:5432/z")
        monkeypatch.setenv("FORCE_IN_MEMORY_STORE", "false")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "32")
        # PgVectorStore constructor will fail to connect → must raise, not fall back.
        with pytest.raises(VectorStoreError):
            get_vector_store()

    def test_unknown_scheme_raises(self, monkeypatch):
        from app.config import reset_settings_cache
        from app.domain import VectorStoreError
        from app.storage import get_vector_store

        reset_settings_cache()
        monkeypatch.setenv("DATABASE_URL", "mysql+asyncmy://user:pwd@localhost/db")
        monkeypatch.setenv("FORCE_IN_MEMORY_STORE", "false")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "32")
        with pytest.raises(VectorStoreError):
            get_vector_store()
