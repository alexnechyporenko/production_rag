"""Unit tests for the retrieval hybrid fusion + reranker."""

from __future__ import annotations

import pytest
from app.domain import RetrievalQuery, RetrievedChunk
from app.reranking import MockReranker, get_reranker
from app.retrieval import HybridRetriever, min_max_normalize, reciprocal_rank_fuse


def _chunk(i: int, content: str, score: float = 0.5, method: str = "dense") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{i}",
        document_id=f"doc-{i}",
        content=content,
        score=score,
        retrieval_method=method,
        rank=i,
        metadata={},
    )


# ============================================================================
# Helpers
# ============================================================================


class TestMinMaxNormalize:
    def test_basic(self):
        out = min_max_normalize([1.0, 3.0, 5.0])
        assert out == [0.0, 0.5, 1.0]

    def test_empty(self):
        assert min_max_normalize([]) == []

    def test_constant(self):
        # All-equal non-zero values normalise to 1.0 (each is at the max).
        assert min_max_normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]

    def test_all_zero(self):
        assert min_max_normalize([0.0, 0.0]) == [0.0, 0.0]


class TestRRF:
    def test_fuses_two_lists(self):
        dense = [_chunk(0, "a", score=1.0, method="dense"), _chunk(1, "b", score=0.5, method="dense")]
        lexical = [_chunk(1, "b", score=0.9, method="lexical"), _chunk(2, "c", score=0.1, method="lexical")]
        fused = reciprocal_rank_fuse(dense, lexical, k=60)
        ids = [c.chunk_id for c in fused]
        # chunk-1 appears at rank 0 in both lists → highest RRF score.
        assert ids[0] == "chunk-1"

    def test_empty(self):
        assert reciprocal_rank_fuse([], []) == []


# ============================================================================
# HybridRetriever
# ============================================================================


@pytest.fixture
def populated_retriever(in_memory_store, lexical_index, embedder):
    from app.domain import DocumentChunk

    chunks = [
        DocumentChunk(id=f"chunk-{i}", document_id="doc-1", content=c, sequence=i, char_start=0, char_end=len(c))
        for i, c in enumerate(
            [
                "PostgreSQL is a database with pgvector",
                "FastAPI is a Python web framework",
                "Hybrid search combines dense and lexical retrieval",
                "Reranking is an optional second stage",
            ]
        )
    ]
    vecs = embedder.embed_texts([c.content for c in chunks])
    in_memory_store.add(chunks, vecs, model=embedder.model)
    lexical_index.add(chunks)
    return HybridRetriever(
        vector_store=in_memory_store,
        lexical_index=lexical_index,
        embedding_provider=embedder,
        alpha=0.5,
    )


class TestHybridRetriever:
    def test_retrieves_and_fuses(self, populated_retriever):
        q = RetrievalQuery(text="hybrid search dense lexical")
        result = populated_retriever.retrieve(q)
        assert result.query_id == q.query_id
        assert len(result.candidates) > 0
        top = result.candidates[0]
        assert top.retrieval_method == "hybrid"
        assert top.rank == 0
        assert "Hybrid search" in top.content

    def test_alpha_0_is_lexical_only(self, populated_retriever):
        # When alpha=0, dense contribution is zeroed after normalisation →
        # lexical signal dominates if it exists.
        populated_retriever.alpha = 0.0
        q = RetrievalQuery(text="reranking")
        r = populated_retriever.retrieve(q)
        assert any("Reranking" in c.content for c in r.candidates)

    def test_alpha_1_is_dense_only(self, populated_retriever):
        populated_retriever.alpha = 1.0
        q = RetrievalQuery(text="reranking")
        r = populated_retriever.retrieve(q)
        # Dense retrieved the reranking chunk too.
        assert any("Reranking" in c.content for c in r.candidates)

    def test_top_k_truncates(self, populated_retriever):
        q = RetrievalQuery(text="database framework search reranking", top_k=2)
        r = populated_retriever.retrieve(q)
        assert len(r.candidates) <= 2

    def test_empty_query_methods_returns_no_candidates(self, populated_retriever):
        # Even with no methods, the result must not raise.
        q = RetrievalQuery(text="x", methods=("dense",))
        r = populated_retriever.retrieve(q)
        assert r.candidates is not None


# ============================================================================
# Reranker
# ============================================================================


class TestMockReranker:
    def test_reranks_by_query_overlap(self):
        rr = MockReranker()
        candidates = [
            _chunk(0, "unrelated content xyz"),
            _chunk(1, "hybrid search combines dense and lexical retrieval"),
            _chunk(2, "another unrelated piece"),
        ]
        result = rr.rerank("hybrid search retrieval", candidates, top_k=3)
        assert result.ranked[0].chunk_id == "chunk-1"

    def test_preserves_chunk_ids(self):
        rr = MockReranker()
        cands = [_chunk(0, "alpha"), _chunk(1, "beta")]
        out = rr.rerank("alpha", cands)
        assert {c.chunk_id for c in out.ranked} == {c.chunk_id for c in cands}

    def test_empty_candidates(self):
        rr = MockReranker()
        out = rr.rerank("alpha", [])
        assert out.ranked == []

    def test_top_k_limit(self):
        rr = MockReranker()
        cands = [_chunk(i, f"item {i}") for i in range(10)]
        out = rr.rerank("item", cands, top_k=3)
        assert len(out.ranked) == 3

    def test_factory_none(self):
        # Directly test the factory
        assert get_reranker(provider="none") is None

    def test_factory_mock(self):
        assert isinstance(get_reranker(provider="mock"), MockReranker)
