"""Unit tests for evaluation metrics."""

from __future__ import annotations

from app.domain import Citation, CostEstimate, RAGAnswer, TokenUsage
from app.evaluation import (
    Evaluator,
    answer_completeness,
    answer_relevance,
    citation_correctness,
    groundedness,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
)


class TestRetrievalMetrics:
    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0
        assert recall_at_k(["a", "b", "c"], {"d"}, k=3) == 0.0
        assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == 0.5
        assert recall_at_k([], {"a"}, k=3) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k(["a", "b"], {"a"}, k=2) == 0.5
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0
        assert precision_at_k([], {"a"}, k=0) == 0.0

    def test_hit_rate_at_k(self):
        assert hit_rate_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
        assert hit_rate_at_k(["a", "b", "c"], {"d"}, k=3) == 0.0

    def test_mrr(self):
        assert mrr(["a", "b"], {"b"}) == 0.5
        assert mrr(["a", "b"], {"c"}) == 0.0
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_ndcg_at_k(self):
        # Perfect ranking: relevant at top.
        v = ndcg_at_k(["a", "b", "c"], {"a"}, k=3)
        assert v == 1.0
        # All relevant at the bottom.
        v = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
        assert 0 < v < 1
        # No relevant
        assert ndcg_at_k(["x", "y"], {"z"}, k=2) == 0.0

    def test_retrieval_metrics_dict(self):
        m = retrieval_metrics(["a", "b"], {"a"}, k=2)
        assert set(m.keys()) == {"recall@k", "precision@k", "hit_rate@k", "mrr", "ndcg@k"}
        assert m["recall@k"] == 1.0
        assert m["precision@k"] == 0.5


class TestAnswerMetrics:
    def _ans(self, text: str, chunk_ids: list[str], evidence_sufficient: bool = True) -> RAGAnswer:
        return RAGAnswer(
            query_id="q1",
            answer=text,
            citations=[Citation(chunk_id=c, document_id=f"doc-{c}", snippet=c) for c in chunk_ids],
            confidence=1.0,
            used_chunks=chunk_ids,
            token_usage=TokenUsage.from_pair(input_tokens=10, output_tokens=10, model="m"),
            cost_estimate=CostEstimate.unknown(),
            latency_ms=1.0,
            evidence_sufficient=evidence_sufficient,
        )

    def test_citation_correctness_all_correct(self):
        ans = self._ans("x", ["a", "b"])
        assert citation_correctness(ans.citations, {"a", "b"}) == 1.0

    def test_citation_correctness_partial(self):
        ans = self._ans("x", ["a", "z"])
        assert citation_correctness(ans.citations, {"a", "b"}) == 0.5

    def test_citation_correctness_no_relevant(self):
        ans = self._ans("x", ["a"])
        assert citation_correctness(ans.citations, set()) == 0.0

    def test_groundedness_with_citations(self):
        ans = self._ans("x", ["a", "b"])
        assert groundedness(ans, {"a", "b"}) == 1.0

    def test_groundedness_no_citations(self):
        ans = self._ans("x", [])
        assert groundedness(ans, {"a"}) == 0.0

    def test_groundedness_no_relevant_falls_back_to_evidence(self):
        ans = self._ans("x", ["a"], evidence_sufficient=True)
        assert groundedness(ans, set()) == 1.0
        ans2 = self._ans("x", ["a"], evidence_sufficient=False)
        assert groundedness(ans2, set()) == 0.0

    def test_answer_relevance_overlap(self):
        ans = self._ans("hybrid search dense lexical", ["a"])
        # All query tokens are in the answer → 1.0
        assert answer_relevance(ans, "hybrid search dense lexical") == 1.0
        # Partial overlap.
        ans2 = self._ans("hybrid search", ["a"])
        assert 0 < answer_relevance(ans2, "hybrid search dense lexical") < 1

    def test_answer_completeness(self):
        ans = self._ans("hybrid search dense lexical retrieval", ["a"])
        v = answer_completeness(ans, "hybrid search dense lexical retrieval")
        assert v == 1.0
        ans2 = self._ans("hybrid search", ["a"])
        v = answer_completeness(ans2, "hybrid search dense lexical retrieval")
        assert 0 < v < 1

    def test_answer_completeness_no_expected(self):
        ans = self._ans("x", [])
        assert answer_completeness(ans, None) == 0.0


class TestEvaluator:
    def test_evaluator_runs_against_orchestrator(self):
        # Build a tiny in-memory orchestrator with mock everything.
        from app.embeddings import MockEmbeddingProvider
        from app.ingestion import IngestionService
        from app.pipeline import OrchestratorComponents, RAGOrchestrator
        from app.storage import InMemoryVectorStore, LexicalIndex

        emb = MockEmbeddingProvider(dimension=64)
        store = InMemoryVectorStore(dimension=64)
        lex = LexicalIndex()

        ingest = IngestionService()
        # Ingest one doc; capture chunk ids so we can use them as relevant.
        r = ingest.ingest_text("Hybrid search combines dense and lexical retrieval with alpha fusion")
        chunks = r.chunks
        store.add(chunks, emb.embed_texts([c.content for c in chunks]), model=emb.model)
        lex.add(chunks)

        orch = RAGOrchestrator(
            components=OrchestratorComponents(
                vector_store=store,
                lexical_index=lex,
                embedding_provider=emb,
            )
        )
        ev = Evaluator(orchestrator=orch, top_k=3)
        cases = [
            {
                "case_id": "c1",
                "query": "What is hybrid search?",
                "relevant_chunk_ids": [chunks[0].id],
            }
        ]
        result = ev.evaluate(cases)
        assert result["n_cases"] == 1
        assert result["retrieval_metrics_mean"]["recall@k"] == 1.0
        assert result["latency_ms_mean"] > 0
        assert result["cases"][0]["error"] is None
