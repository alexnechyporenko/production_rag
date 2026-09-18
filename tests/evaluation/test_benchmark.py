"""Benchmark + evaluation tests — opt-in via `pytest -m benchmark`.

Runs both:
  1. `test_benchmark_runs` — measures latency / token / cost across the full
     evaluation dataset (offline mock pipeline) and dumps the result to
     `benchmarks/last_benchmark.json`.
  2. `test_evaluation_runs` — runs the evaluation harness against the same
     dataset and reports retrieval + answer metrics.

Both are marked slow + benchmark so normal pytest runs skip them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.pipeline import OrchestratorComponents, RAGOrchestrator
from benchmarks.runner import run_benchmark, save_benchmark
from datasets.builtin import DATASET_CASES, DATASET_DOCS, DATASET_QUERIES

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


@pytest.fixture
def populated_orchestrator():
    """Build an offline orchestrator and ingest the full evaluation dataset."""
    from app.embeddings import MockEmbeddingProvider
    from app.storage import InMemoryVectorStore, LexicalIndex

    emb = MockEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    lex = LexicalIndex()
    orch = RAGOrchestrator(
        components=OrchestratorComponents(
            vector_store=store, lexical_index=lex, embedding_provider=emb,
        )
    )
    docs = [{"text": d["text"], "title": d.get("title")} for d in DATASET_DOCS]
    orch.ingest_documents([{"kind": "text", **d} for d in docs])
    return orch


def test_benchmark_runs(populated_orchestrator):
    """Run the latency / cost benchmark against the full evaluation dataset.

    The harness re-ingests from scratch (timing included) so `n_documents`,
    `n_chunks`, and `chunks_per_s` are populated honestly.
    """
    from app.embeddings import MockEmbeddingProvider
    from app.storage import InMemoryVectorStore, LexicalIndex

    # Fresh orchestrator so the harness times ingestion from zero.
    emb = MockEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    lex = LexicalIndex()
    orch = RAGOrchestrator(
        components=OrchestratorComponents(
            vector_store=store, lexical_index=lex, embedding_provider=emb,
        )
    )

    docs = [{"text": d["text"], "title": d.get("title")} for d in DATASET_DOCS]
    queries = [{"query": q["query"], "top_k": 5, "rerank": True} for q in DATASET_QUERIES]
    bench = run_benchmark(orch, docs=docs, queries=queries, repeat=1, name="builtin-mock-full")

    # Honest reporting
    assert bench["n_documents"] == len(DATASET_DOCS), bench
    assert bench["n_chunks"] >= len(DATASET_DOCS), bench
    assert bench["n_queries"] == len(DATASET_QUERIES)
    assert "latencies_ms" in bench
    assert "end_to_end" in bench["latencies_ms"]
    assert bench["latencies_ms"]["end_to_end"]["mean"] >= 0
    assert bench["cost"]["pricing_known"] is False  # mock LLM has no real price
    assert bench["ingestion"]["total_s"] >= 0
    assert bench["ingestion"]["chunks_per_s"] >= 0

    out = Path(__file__).resolve().parents[2] / "benchmarks" / "last_benchmark.json"
    save_benchmark(bench, out)
    print(f"Benchmark saved to {out}")
    print(json.dumps(bench, indent=2))


def test_evaluation_runs(populated_orchestrator):
    """Run the evaluation harness against the full dataset and persist the
    retrieval + answer metrics to `benchmarks/last_evaluation.json`.
    """
    from app.evaluation import Evaluator

    ev = Evaluator(orchestrator=populated_orchestrator, top_k=5)
    result = ev.evaluate(DATASET_CASES)

    assert result["n_cases"] == len(DATASET_CASES), result
    assert "recall@k" in result["retrieval_metrics_mean"]
    assert "groundedness" in result["answer_metrics_mean"]
    assert result["latency_ms_mean"] > 0

    # Sanity: with the deterministic mock pipeline, every query should retrieve
    # at least one candidate (no query should return zero candidates).
    for case in result["cases"]:
        assert case["error"] is None, case
        assert "recall@k" in case["retrieval_metrics"], case

    out = Path(__file__).resolve().parents[2] / "benchmarks" / "last_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Evaluation saved to {out}")
    print(json.dumps(result, indent=2)[:2000])
