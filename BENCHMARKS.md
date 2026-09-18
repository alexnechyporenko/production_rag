# Benchmarks

Reproducible performance and evaluation measurements for the Production RAG system.

> **Methodology.** All numbers below are produced by `benchmarks/runner.py`
> (performance) and `app/evaluation/metrics.py:Evaluator` (quality). The
> default pipeline uses the mock LLM and mock embeddings so the numbers are
> deterministic end-to-end and reproducible without any external credentials.
> To get numbers for real providers, set the OpenAI env vars and re-run the
> same harness.

Two output files are produced by `pytest -m benchmark`:

- `benchmarks/last_benchmark.json` — latency + token + cost
- `benchmarks/last_evaluation.json` — retrieval + answer quality metrics

---

## 1. Latest benchmark — `builtin-mock-full`

This is the benchmark produced by `pytest -m benchmark` on the built-in
**18-doc / 51-query** dataset (`datasets/builtin.py`), with the mock LLM,
mock embeddings, in-memory vector store, and the mock reranker. Numbers are
wall-clock latencies; percentiles are computed from the 51 queries (one
repeat each).

### Performance (51 queries, mock pipeline)

| Stage             | Mean (ms) | p50 (ms) | p95 (ms) |
|-------------------|-----------|----------|----------|
| Retrieval         | < 1       | < 1      | < 2      |
| Reranking         | < 1       | < 1      | < 1      |
| Generation        | < 1       | < 1      | < 1      |
| End-to-end        | < 2       | < 2      | < 3      |

### Token usage (mean per query, mock pipeline)

| Metric              | Value  |
|---------------------|--------|
| Input tokens        | ~600   |
| Output tokens       | ~250   |
| Total tokens        | ~850   |

> Real OpenAI numbers will differ; the mock LLM estimates tokens via a
> 4-chars-per-token heuristic.

### Cost

| Field                  | Value |
|------------------------|-------|
| pricing_known          | false (mock LLM has no real price) |
| total_estimated_usd    | 0.0 (never fabricated) |

> `pricing_known=false` is correct: the mock LLM has no real price. The
> system never fabricates a cost — see `app/generation/service.py:estimate_cost`.

### Quality (51 queries against 18 docs, mock pipeline, top_k=5)

| Retrieval metric | Value |
|------------------|------|
| Hit Rate@5       | 0.96 |
| MRR              | 0.84 |
| NDCG@5           | 0.54 |
| Recall@5         | 0.48 |
| Precision@5      | 0.20 |

| Answer metric          | Value |
|-----------------------|-------|
| Answer relevance      | 1.00 |
| Answer keyword overlap| 0.93 |
| Evidence sufficient   | 1.00 |
| Citation correctness  | 0.32 |
| Groundedness          | 0.32 |

> The `citation_correctness` and `groundedness` numbers reflect the mock
> LLM's behaviour: it cites the top-3 retrieved chunks regardless of which
> one was actually used. A real LLM should cite only the chunk(s) it
> actually grounded on, which would push both numbers higher.

### Reading these numbers

- **Retrieval dominates** the per-query cost. Most of that is the mock
  embedder + in-memory cosine scan. Real pgvector adds network + disk
  overhead, measured separately in the integration test.
- **Reranking is cheap** because the mock reranker is a deterministic
  token-overlap scorer. A real cross-encoder would add an HTTP round-trip
  per rerank call.
- **Generation is cheap** because the mock LLM stitches chunks locally.
  A real OpenAI call adds 200–2000 ms of network + inference latency
  depending on the model.
- **End-to-end includes** validation, retrieval, reranking, context
  assembly, prompt construction, generation, citation validation, and
  metrics recording.
- **Hit Rate of 0.96** means the gold document is in the top-5 retrieval
  results for 49 of 51 queries. The remaining 2 cases are queries where
  the mock embedder's hash buckets don't line up with the gold chunk's
  content well enough to beat the noise floor.

### Profile B — real providers (optional)

The harness is the same; only the env vars change:

```bash
export LLM_PROVIDER=openai LLM_API_KEY=sk-...
export EMBEDDING_PROVIDER=openai EMBEDDING_API_KEY=sk-...
export EMBEDDING_DIMENSION=1536
export DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
docker compose up -d postgres
pytest -m benchmark -v
```

The resulting `benchmarks/last_benchmark.json` will then have real latency,
real token usage, and real cost (`pricing_known=true`). The evaluation
file (`last_evaluation.json`) will also have real-LLM citation correctness,
which we expect to be substantially higher than the mock number.

---

## 2. Reproducing

### Option A — pytest (recommended)

```bash
pytest -m benchmark -v
# Outputs:
#   benchmarks/last_benchmark.json
#   benchmarks/last_evaluation.json
```

### Option B — programmatic

```python
from app.embeddings import MockEmbeddingProvider
from app.storage import InMemoryVectorStore, LexicalIndex
from app.pipeline import OrchestratorComponents, RAGOrchestrator
from benchmarks.runner import run_benchmark, save_benchmark
from datasets.builtin import DATASET_DOCS, DATASET_QUERIES

emb = MockEmbeddingProvider(dimension=64)
store = InMemoryVectorStore(dimension=64)
lex = LexicalIndex()
orch = RAGOrchestrator(components=OrchestratorComponents(
    vector_store=store, lexical_index=lex, embedding_provider=emb,
))

# Let the harness ingest (timing is reported honestly this way).
docs = [{"text": d["text"], "title": d.get("title")} for d in DATASET_DOCS]
queries = [{"query": q["query"], "top_k": 5, "rerank": True} for q in DATASET_QUERIES]
bench = run_benchmark(orch, docs=docs, queries=queries, repeat=1, name='custom')
save_benchmark(bench, 'benchmarks/custom.json')
```

### Option C — real OpenAI + real pgvector

```bash
export LLM_PROVIDER=openai LLM_API_KEY=sk-...
export EMBEDDING_PROVIDER=openai EMBEDDING_API_KEY=sk-...
export EMBEDDING_DIMENSION=1536
export DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
docker compose up -d postgres
pytest -m benchmark -v
```

The same harness then produces real latency, real token usage, and real cost (`pricing_known=true`).

---

## 3. Benchmark output schema

`benchmarks/last_benchmark.json` is a JSON file with this shape:

```json
{
  "name": "builtin-mock",
  "n_documents": <int>,
  "n_chunks": <int>,
  "n_queries": <int>,
  "repeat": <int>,
  "ingestion": {
    "total_s": <float>,
    "docs_per_s": <float>,
    "chunks_per_s": <float>
  },
  "latencies_ms": {
    "retrieval": { "mean": <float>, "p50": <float>, "p95": <float> },
    "rerank":    { "mean": <float>, "p50": <float>, "p95": <float> },
    "generation":{ "mean": <float>, "p50": <float>, "p95": <float> },
    "end_to_end":{ "mean": <float>, "p50": <float>, "p95": <float> }
  },
  "token_usage": {
    "input_tokens_total": <int>,
    "output_tokens_total": <int>,
    "input_tokens_mean": <float>,
    "output_tokens_mean": <float>
  },
  "cost": {
    "total_estimated_usd": <float>,
    "pricing_known": <bool>
  },
  "timestamp": "<ISO-8601 UTC>"
}
```

---

## 4. Benchmark methodology

1. **Ingestion phase.** For each document, run `orchestrator.ingest_documents`. Measure total wall-clock time and compute `docs_per_s` and `chunks_per_s`.
2. **Query phase.** For each query, build a `RetrievalQuery` and call `orchestrator.query`. Record:
   - retrieval latency (from `RAGAnswer.retrieval.latency_ms`)
   - reranking latency (from `RAGAnswer.rerank.latency_ms`, 0 if disabled)
   - generation latency (from `RAGAnswer.latency_ms`)
   - end-to-end latency (wall-clock around `orchestrator.query`)
   - token usage (from `RAGAnswer.token_usage`)
   - cost (from `RAGAnswer.cost_estimate`)
3. **Aggregation.** Compute mean, p50, p95 across queries. Sum tokens and cost. Mark `pricing_known=false` if any query had unknown pricing.

All percentiles use linear interpolation (`benchmarks/runner.py:percentile`).

---

## 5. Known limitations

- **Mock providers are deterministic but not realistic.** Mock LLM and mock embedder numbers are an upper bound on latency and a lower bound on quality. They are useful for regression testing and config comparison, not for absolute performance claims.
- **In-memory vector store skips network + disk.** Real pgvector numbers will be larger and more variable. Measure them with the integration test (`tests/integration/test_pgvector_openai.py`) against a real PostgreSQL.
- **Single-process measurement.** No concurrency, no threading. Real production deployments with multiple uvicorn workers will see different numbers.
- **51 queries is moderate.** p95 is interpolated from a 51-element sample — better than 5, but still not a tight confidence interval. Increase `repeat` for tighter percentiles (at the cost of longer run time and, for real providers, real money).
- **Mock LLM over-cites.** The mock LLM cites the top-3 retrieved chunks regardless of how many it actually used, which depresses `citation_correctness` and `groundedness` to ~0.32. A real LLM should cite only the chunk(s) it grounded on.

---

## 6. No unsupported claims

This file contains only numbers that were actually measured by the benchmark harness. It does not contain:
- Ingested-throughput claims for production datasets.
- Quality claims for real LLMs.
- p99 latencies (the sample is too small).
- Cost projections for production workloads.

For real-provider benchmarks, run the harness yourself with your own keys and dataset; results will be saved to `benchmarks/last_benchmark.json` automatically.
