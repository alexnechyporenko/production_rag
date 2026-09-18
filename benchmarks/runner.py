"""Benchmark harness.

Runs the full pipeline against an evaluation dataset and reports:
  * ingestion throughput (docs/s, chunks/s)
  * embedding throughput (chunks/s)
  * retrieval latency (mean / p50 / p95)
  * reranking latency (mean / p50 / p95)
  * generation latency (mean / p50 / p95)
  * end-to-end latency (mean / p50 / p95)
  * token usage (input / output / total)
  * estimated cost

The harness is intentionally simple so the numbers are reproducible (P-09,
NFR-03). All results are dumped as JSON; the EVALUATION.md doc walks through
what each number means.
"""

from __future__ import annotations

import json
import time
from datetime import UTC
from pathlib import Path
from typing import Any

from app.domain import now_ms


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_benchmark(
    orchestrator: Any,
    docs: list[dict[str, str]],
    queries: list[dict[str, Any]],
    *,
    repeat: int = 1,
    name: str = "default",
) -> dict[str, Any]:
    """Run the benchmark. `docs` and `queries` are dicts matching the
    IngestionService and RetrievalQuery contracts.
    """
    # Ingestion timing
    t0 = time.perf_counter()
    chunks_total = 0
    for d in docs:
        r = orchestrator.ingest_documents([{"kind": "text", **d}])[0]
        chunks_total += len(r["chunk_ids"])
    t_ingest = time.perf_counter() - t0
    docs_per_s = len(docs) / t_ingest if t_ingest > 0 else 0.0
    chunks_per_s = chunks_total / t_ingest if t_ingest > 0 else 0.0

    # Query timing
    from app.domain import RetrievalQuery

    retrieval_lats: list[float] = []
    rerank_lats: list[float] = []
    gen_lats: list[float] = []
    e2e_lats: list[float] = []
    in_toks: list[int] = []
    out_toks: list[int] = []
    total_cost = 0.0
    pricing_known = True
    n = 0
    for _ in range(repeat):
        for q in queries:
            rq = RetrievalQuery(
                text=q["query"], top_k=q.get("top_k", 10), rerank=q.get("rerank", True)
            )
            t_start = now_ms()
            ans = orchestrator.query(rq)
            t_end = now_ms()
            e2e_lats.append(t_end - t_start)
            if ans.retrieval:
                retrieval_lats.append(ans.retrieval.latency_ms)
            if ans.rerank:
                rerank_lats.append(ans.rerank.latency_ms)
            gen_lats.append(ans.latency_ms)
            in_toks.append(ans.token_usage.input_tokens)
            out_toks.append(ans.token_usage.output_tokens)
            if ans.cost_estimate.pricing_known:
                total_cost += ans.cost_estimate.total_cost
            else:
                pricing_known = False
            n += 1

    bench = {
        "name": name,
        "n_documents": len(docs),
        "n_chunks": chunks_total,
        "n_queries": len(queries),
        "repeat": repeat,
        "ingestion": {
            "total_s": round(t_ingest, 4),
            "docs_per_s": round(docs_per_s, 2),
            "chunks_per_s": round(chunks_per_s, 2),
        },
        "latencies_ms": {
            "retrieval": {
                "mean": round(sum(retrieval_lats) / max(1, len(retrieval_lats)), 3),
                "p50": round(percentile(retrieval_lats, 50), 3),
                "p95": round(percentile(retrieval_lats, 95), 3),
            },
            "rerank": {
                "mean": round(sum(rerank_lats) / max(1, len(rerank_lats)), 3),
                "p50": round(percentile(rerank_lats, 50), 3),
                "p95": round(percentile(rerank_lats, 95), 3),
            },
            "generation": {
                "mean": round(sum(gen_lats) / max(1, len(gen_lats)), 3),
                "p50": round(percentile(gen_lats, 50), 3),
                "p95": round(percentile(gen_lats, 95), 3),
            },
            "end_to_end": {
                "mean": round(sum(e2e_lats) / max(1, len(e2e_lats)), 3),
                "p50": round(percentile(e2e_lats, 50), 3),
                "p95": round(percentile(e2e_lats, 95), 3),
            },
        },
        "token_usage": {
            "input_tokens_total": sum(in_toks),
            "output_tokens_total": sum(out_toks),
            "input_tokens_mean": round(sum(in_toks) / max(1, len(in_toks)), 2),
            "output_tokens_mean": round(sum(out_toks) / max(1, len(out_toks)), 2),
        },
        "cost": {
            "total_estimated_usd": round(total_cost, 8),
            "pricing_known": pricing_known,
        },
        "timestamp": _now_iso(),
    }
    return bench


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def save_benchmark(bench: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bench, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["percentile", "run_benchmark", "save_benchmark"]
