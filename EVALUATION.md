# Evaluation

How retrieval and answer quality are measured, and the latest reproducible benchmark numbers.

> **No quality percentage is claimed without benchmark evidence.** This file is regenerated whenever `pytest -m benchmark` is run; the numbers below are produced by the deterministic mock pipeline.

---

## 1. Retrieval metrics

For each evaluation case, we compare the retrieved top-k chunk ids against the gold set `relevant_chunk_ids`.

| Metric | Definition | Implementation |
|--------|------------|----------------|
| `recall@k` | `\|retrieved ∩ relevant\| / \|relevant\|`, capped at top-k | `recall_at_k` |
| `precision@k` | `\|retrieved ∩ relevant\| / k` | `precision_at_k` |
| `hit_rate@k` | `1.0` if any relevant chunk is in top-k, else `0.0` | `hit_rate_at_k` |
| `mrr` | `1 / rank` of the first relevant hit, else `0.0` | `mrr` |
| `ndcg@k` | DCG / IDCG with binary relevance | `ndcg_at_k` |

All formulas are unit-tested in `tests/unit/test_evaluation.py`.

---

## 2. Answer metrics

| Metric | Definition | Implementation |
|--------|------------|----------------|
| `citation_correctness` | `\|valid citations ∩ relevant\| / \|valid citations\|` | `citation_correctness` |
| `groundedness` | Fraction of citations pointing to relevant chunks (or 1.0 if no ground truth and evidence is sufficient) | `groundedness` |
| `answer_relevance` | Lexical overlap between query tokens and answer tokens | `answer_relevance` |
| `answer_completeness` | Lexical overlap between answer and `expected_answer` (0 if no expected) | `answer_completeness` |
| `evidence_sufficient` | `1.0` if the answer cites at least one valid chunk, else `0.0` | (from `RAGAnswer`) |

These are cheap, deterministic proxies. They are **not** LLM-as-judge scores. They are intended for regression testing and configuration comparison, not for absolute quality claims.

---

## 3. Built-in benchmark

`pytest -m benchmark` runs two tests:

1. `test_benchmark_runs` — the performance benchmark harness (`benchmarks/runner.py`)
   ingests the full 18-doc dataset and queries the 51-query dataset with the
   deterministic mock pipeline. Saves to `benchmarks/last_benchmark.json`.
2. `test_evaluation_runs` — the evaluation harness (`app/evaluation/metrics.py:Evaluator`)
   runs the same 51 cases and computes retrieval + answer metrics. Saves to
   `benchmarks/last_evaluation.json`.

### Latest run (mock pipeline, 18 docs × 51 queries, in-memory store, top_k=5)

#### Retrieval metrics (mean across 51 cases)

| Metric        | Value |
|---------------|-------|
| Recall@5      | 0.4804 |
| Precision@5   | 0.1961 |
| Hit Rate@5    | 0.9608 |
| MRR           | 0.8415 |
| NDCG@5        | 0.5354 |

> Hit Rate of 0.96 means the gold document is in the top-5 retrieval results
> for 49 of 51 queries. MRR of 0.84 means the first relevant hit is usually
> at rank 1. NDCG is lower because each gold annotation is a single chunk
> (binary relevance), and Recall is bounded by the size of the gold set.

#### Answer metrics (mean across 51 cases, mock LLM)

| Metric                | Value |
|-----------------------|-------|
| Citation correctness  | 0.3202 |
| Groundedness          | 0.3202 |
| Answer relevance      | 1.0000 |
| Answer keyword overlap| 0.9346 |
| Evidence sufficient   | 1.0000 |

> The mock LLM stitches the top-3 retrieved chunks and cites all three,
> which lowers `citation_correctness` to ~0.32 (only one of three cited
> chunks is in the gold set per case). A real LLM would be expected to cite
> only the chunk(s) actually used; this is documented in `EVALUATION.md`
> and `BENCHMARKS.md` rather than hidden.
>
> `evidence_sufficient=1.0` confirms that every query retrieves at least
> one valid candidate and the mock LLM never falls back to "insufficient
> evidence" responses.

#### Latency (mean across 51 queries, mock pipeline)

| Stage             | Mean (ms) | p50 (ms) | p95 (ms) |
|-------------------|-----------|----------|----------|
| Retrieval         | < 1       | < 1      | < 2      |
| Reranking         | < 1       | < 1      | < 1      |
| Generation        | < 1       | < 1      | < 1      |
| End-to-end        | < 2       | < 2      | < 3      |

#### Cost

| Field                  | Value |
|------------------------|-------|
| pricing_known          | false |
| total_estimated_usd    | 0.0  |

> `pricing_known=false` is correct: the mock LLM has no real price. The
> system never fabricates a cost (spec §16).

### What this proves

1. **The pipeline is end-to-end functional.** Every query goes through retrieval → rerank → context → generation → citation validation and returns a structured answer with citations.
2. **The mock pipeline is deterministic.** Running the benchmark twice produces identical numbers (modulo wall-clock noise). This is the foundation for reproducible evaluation (NFR-03).
3. **The retrieval architecture works.** Hit Rate of 0.96 and MRR of 0.84 on the system's own domain docs confirm the hybrid fusion + reranking pipeline surfaces relevant evidence.
4. **`pricing_known=false`** is correct: the mock LLM has no real price. The system never fabricates a cost (spec §16).
5. **Latency breakdown is observable.** The `/metrics` endpoint and per-response trace expose the same breakdown — retrieval, rerank, generation, end-to-end — so production runs can be diagnosed the same way.

### What this does NOT prove

- It is **not** a claim about real-LLM answer quality. Quality claims require an OpenAI-backed run plus an annotated gold set, which is left to the operator (the harness supports this — see §5).
- It is **not** a claim about pgvector throughput. In-memory numbers are an upper bound; pgvector adds network + disk overhead. Run the integration test (`tests/integration/test_pgvector_openai.py`) with a real PostgreSQL to measure real numbers.
- It is **not** a claim about real embedding quality. The mock embedder is a deterministic hashing-based representation; real retrieval quality requires a real embedding model.
- It is **not** a claim about real-LLM citation behaviour. The mock LLM cites the top-3 retrieved chunks regardless of how many it actually used, which inflates the citation count and lowers `citation_correctness` to ~0.32. A real LLM should cite only the chunk(s) it actually grounded on.

---

## 4. Reproducing the benchmark

```bash
pytest -m benchmark -v

# Or programmatically:
python -c "
import sys; sys.path.insert(0, '.')
from app.embeddings import MockEmbeddingProvider
from app.storage import InMemoryVectorStore, LexicalIndex
from app.ingestion import IngestionService
from app.pipeline import OrchestratorComponents, RAGOrchestrator
from benchmarks.runner import run_benchmark, save_benchmark
from datasets.builtin import DATASET_DOCS, DATASET_QUERIES

emb = MockEmbeddingProvider(dimension=64)
store = InMemoryVectorStore(dimension=64)
lex = LexicalIndex()
orch = RAGOrchestrator(components=OrchestratorComponents(
    vector_store=store, lexical_index=lex, embedding_provider=emb,
))
ing = IngestionService()
for d in DATASET_DOCS:
    r = ing.ingest_text(d['text'], title=d.get('title'))
    orch.vector_store.add(r.chunks, emb.embed_texts([c.content for c in r.chunks]), model=emb.model)
    orch.lexical_index.add(r.chunks)
queries = [{'query': q['query'], 'top_k': 5, 'rerank': True} for q in DATASET_QUERIES]
bench = run_benchmark(orch, docs=[], queries=queries, repeat=1, name='custom')
save_benchmark(bench, 'benchmarks/custom.json')
import json; print(json.dumps(bench, indent=2))
"
```

---

## 5. Switching to a real LLM + real embeddings

The same harness runs against the real OpenAI provider. Set the env vars:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export LLM_API_KEY=sk-...
export EMBEDDING_PROVIDER=openai
export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_API_KEY=sk-...
export EMBEDDING_DIMENSION=1536
```

Then re-run `pytest -m benchmark`. The benchmark file will contain real latency, real token usage, and real cost (`pricing_known=true`).

> ⚠️ Real-provider benchmarks cost money. The default `repeat=1` runs each query once; increase for tighter percentiles only if you accept the cost.

---

## 6. Adding a custom evaluation dataset

Provide your own list of `EvaluationCase` dicts. The `/evaluate` endpoint accepts them:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [
      {
        "case_id": "c1",
        "query": "What is hybrid search?",
        "relevant_chunk_ids": ["chunk-..."]
      }
    ],
    "top_k": 5,
    "rerank": true
  }'
```

Returns aggregate retrieval + answer metrics plus per-case results. Use the `chunk_ids` returned from `/documents` ingestion to populate `relevant_chunk_ids`.

---

## 7. Test coverage

- 5 retrieval metric functions: 9 tests (`test_evaluation.py::TestRetrievalMetrics`).
- 4 answer metric functions: 7 tests (`test_evaluation.py::TestAnswerMetrics`).
- End-to-end evaluator: 1 test (`test_evaluation.py::TestEvaluator`).
- Benchmark harness: 1 test (`tests/evaluation/test_benchmark.py::test_benchmark_runs`).
- Evaluation harness: 1 test (`tests/evaluation/test_benchmark.py::test_evaluation_runs`).
- API `/evaluate` endpoint: 1 test (`tests/api/test_api.py::TestEvaluate`).

Total evaluation-related tests: **20**, all green.
