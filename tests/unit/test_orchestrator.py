"""Unit tests for the orchestrator (end-to-end pipeline)."""

from __future__ import annotations

import pytest
from app.domain import RetrievalQuery, ValidationError
from app.pipeline import OrchestratorComponents, RAGOrchestrator


@pytest.fixture
def populated_orchestrator(in_memory_store, lexical_index, embedder):
    """An orchestrator with a few small documents pre-ingested."""
    from app.ingestion import IngestionService

    IngestionService()
    orch = RAGOrchestrator(
        components=OrchestratorComponents(
            vector_store=in_memory_store,
            lexical_index=lexical_index,
            embedding_provider=embedder,
        )
    )
    docs = [
        "PostgreSQL is a database with pgvector for vector similarity search.",
        "FastAPI is a Python web framework with Pydantic validation.",
        "Hybrid search combines dense vector retrieval with lexical BM25 retrieval.",
        "Reranking is an optional second stage that scores query-chunk pairs.",
    ]
    for d in docs:
        items = [{"kind": "text", "text": d}]
        orch.ingest_documents(items)
    return orch


class TestIngestion:
    def test_ingest_returns_chunk_ids(self, populated_orchestrator):
        # Pre-populated by fixture
        assert populated_orchestrator.vector_store.count() >= 4

    def test_ingest_failure_isolation(self, in_memory_store, lexical_index, embedder):
        orch = RAGOrchestrator(
            components=OrchestratorComponents(
                vector_store=in_memory_store,
                lexical_index=lexical_index,
                embedding_provider=embedder,
            )
        )
        items = [
            {"kind": "text", "text": "good doc"},
            {"kind": "text", "text": ""},  # invalid
            {"kind": "bogus"},
        ]
        results = orch.ingest_documents(items)
        assert len(results) == 3
        assert results[0]["error"] is None
        assert results[1]["error"] is not None
        assert results[2]["error"] is not None


class TestQuery:
    def test_query_returns_grounded_answer(self, populated_orchestrator):
        q = RetrievalQuery(text="What is hybrid search?")
        ans = populated_orchestrator.query(q)
        assert ans.query_id == q.query_id
        assert ans.evidence_sufficient is True
        assert any("hybrid" in c.content.lower() for c in ans.citations if hasattr(c, "content")) or any(
            "hybrid" in (c.snippet or "").lower() for c in ans.citations
        )
        assert ans.latency_ms >= 0
        assert ans.retrieval is not None
        assert ans.rerank is not None

    def test_query_no_evidence_returns_insufficient(self, in_memory_store, lexical_index, embedder):
        orch = RAGOrchestrator(
            components=OrchestratorComponents(
                vector_store=in_memory_store,
                lexical_index=lexical_index,
                embedding_provider=embedder,
            )
        )
        ans = orch.query(RetrievalQuery(text="anything"))
        assert ans.evidence_sufficient is False
        assert ans.citations == []
        assert "insufficient" in ans.answer.lower()

    def test_query_too_long_rejected(self, populated_orchestrator):
        with pytest.raises(ValidationError):
            populated_orchestrator.query(RetrievalQuery(text="x" * 10000))


class TestSearchOnly:
    def test_search_returns_candidates_only(self, populated_orchestrator):
        q = RetrievalQuery(text="reranking")
        r = populated_orchestrator.search(q)
        assert r.query_id == q.query_id
        assert len(r.candidates) > 0
        assert all(c.retrieval_method == "hybrid" for c in r.candidates)


class TestCaching:
    def test_repeated_search_uses_cache(self, populated_orchestrator):
        # Enable cache for this test by recreating registry.
        from app.caching import get_cache_registry

        get_cache_registry().enabled = True
        q = RetrievalQuery(text="hybrid search")
        r1 = populated_orchestrator.search(q)
        # Cache hit returns the exact same result (same query_id).
        r2 = populated_orchestrator.search(q)
        assert r1.query_id == r2.query_id  # cached result returned verbatim
        assert len(r1.candidates) == len(r2.candidates)
        # Verify the cache recorded a hit on the second call.
        snap = get_cache_registry().snapshot()
        assert any(s["hits"] >= 1 for s in snap.values())
