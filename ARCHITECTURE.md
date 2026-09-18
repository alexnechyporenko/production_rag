# Architecture

This document describes the production RAG system's architecture: components, boundaries, data flow, and the rationale behind the major decisions.

---

## 1. Target architecture

```
                    ┌─────────────────────┐
                    │      FastAPI         │
                    │      API Layer       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    RAG Pipeline      │
                    │    Orchestrator     │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   Query Processing      Retrieval Layer      Context Builder
          │                    │                    │
          │          ┌─────────┴─────────┐          │
          │          ▼                   ▼          │
          │      Dense Search       Lexical Search  │
          │          │                   │          │
          │          └─────────┬─────────┘          │
          │                    ▼                    │
          │              Hybrid Ranking             │
          │                    │                    │
          │                    ▼                    │
          │                Reranker                  │
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │     LLM Provider    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Answer + Citations  │
                    └─────────────────────┘
```

The **document pipeline** (offline, before query time):

```
Documents → Loader → Parser → Normalizer → Chunker → Embedding Provider → Vector Store + Lexical Index
```

---

## 2. Module responsibilities

| Module | Responsibility | Key file(s) |
|--------|----------------|-------------|
| `app.config` | Environment-driven settings | `settings.py` |
| `app.domain` | Typed models + error taxonomy | `models.py`, `errors.py` |
| `app.ingestion` | Load + normalize documents | `loader.py`, `service.py` |
| `app.chunking` | Deterministic chunking | `chunker.py` |
| `app.embeddings` | Provider-neutral embeddings | `provider.py` |
| `app.storage` | Vector store + BM25 lexical index | `vector_store.py`, `pgvector_store.py` |
| `app.retrieval` | Hybrid fusion | `hybrid.py` |
| `app.reranking` | Second-stage scoring | `reranker.py` |
| `app.context` | Bounded, deterministic context | `builder.py` |
| `app.generation` | LLM call + citation validation | `provider.py`, `prompt.py`, `service.py` |
| `app.evaluation` | Retrieval + answer metrics | `metrics.py` |
| `app.caching` | LRU + stable keys | `registry.py` |
| `app.observability` | structlog + metrics + tracing | `logging.py`, `metrics.py`, `tracing.py` |
| `app.pipeline` | End-to-end orchestration | `pipeline.py` |
| `app.api` | FastAPI routes | `routes.py`, `models.py` |
| `app.main` | `create_app()` | `main.py` |

Each module exposes a `Protocol` or a single facade so callers depend only on the contract, not the implementation. This satisfies **P-01 (Modularity)** and **P-02 (Provider Independence)**.

---

## 3. Request lifecycle (query path)

1. **Request enters FastAPI.** Middleware assigns a `request_id`, propagates it via contextvars.
2. **`/query` handler** builds a `RetrievalQuery` (validated by Pydantic) and calls `RAGOrchestrator.query_async`.
3. **Orchestrator validates the query** (length, content) and starts a `RequestTrace`.
4. **Retrieval.** `HybridRetriever.retrieve` runs:
   - **Dense retrieval** via `VectorStore.search` (cosine similarity).
   - **Lexical retrieval** via `LexicalIndex.search` (BM25Okapi).
   - **Fusion.** Default: linear with `alpha = HYBRID_ALPHA`. Alternative: RRF.
   - Result is a `RetrievalResult` with up to `top_k` candidates ranked by fused score.
5. **Reranking** (optional). If `RERANKER_PROVIDER != "none"`, the reranker rescores the top candidates using `(query, chunk)` pairs. The reranker MUST preserve `chunk_id` and `document_id` (validated in tests).
6. **Context assembly.** `ContextBuilder.build` produces a bounded `RAGContext`:
   - Deduplicates by `chunk_id`.
   - Enforces `max_chunks`, `max_chars`, `max_tokens`.
   - Preserves source metadata + citation ids.
   - Output is deterministic for the same input.
7. **Prompt construction.** `build_prompt` emits a `(system, user)` pair. The system prompt enforces grounded behaviour (no hallucination, cite or admit insufficient evidence).
8. **LLM call.** `LLMProvider.complete_async` is called. The mock provider extracts structured chunk records from the prompt and stitches a grounded answer; the OpenAI provider calls `/v1/chat/completions` with retries.
9. **Citation validation.** `validate_citations` checks that every `[cite:<chunk_id>]` marker in the answer references a chunk actually present in the context. Invalid citations are stripped; the count is reported as a warning.
10. **Cost computation.** `estimate_cost` uses the configured pricing table. If the model is unknown, `pricing_known=false` and total cost is `0.0` — never fabricated.
11. **Response.** `RAGAnswer` is returned with the text, citations, confidence, token usage, cost estimate, latency, retrieval result, rerank result, and warnings.
12. **Metrics.** Counters, histograms, and the request trace are updated. The `/metrics` endpoint exposes them.

---

## 4. Ingestion lifecycle (offline path)

1. **`IngestionService.ingest_*`** takes raw input (`text`, `bytes`, or `file`).
2. **`DocumentLoader`** dispatches on content-type (`text/plain`, `text/markdown`, `application/json`, `text/html`, `application/pdf`) and produces a normalized `Document` (plain UTF-8 text).
3. **Chunker** (`fixed` / `recursive` / `sentence`) splits the document into `DocumentChunk` records with **stable, deterministic ids** (`deterministic_chunk_id` = `sha1(doc_id, sequence, content[:128])` truncated). Re-ingesting the same document yields the same chunk ids → cache stability, idempotent re-indexing, reproducible evaluation.
4. **Embedding provider** embeds all chunk contents in a single batch.
5. **Vector store** upserts `(chunk, vector, model)` triples. The in-memory store is the default; pgvector is used when `DATABASE_URL` is PostgreSQL.
6. **Lexical index** tokenises and indexes the chunk contents for BM25.

**Failure isolation (NFR-06).** In `ingest_documents`, each item is wrapped in its own try/except. A failing item is reported as `error` in the result list; the rest of the batch proceeds.

---

## 5. Hybrid retrieval in detail

Given a query `q`:

1. **Dense scores** `d_i = cosine(query_vec, chunk_i_vec)` for `i` in top candidates.
2. **Lexical scores** `l_i = bm25_score(q, chunk_i)`.
3. **Min-max normalisation** brings each side to `[0, 1]`:
   ```
   d_norm_i = (d_i - min(d)) / (max(d) - min(d))     if max != min else 1.0 if d_i > 0 else 0.0
   l_norm_i = ... (same for l)
   ```
4. **Linear fusion:**
   ```
   hybrid_score_i = alpha * d_norm_i + (1 - alpha) * l_norm_i
   ```
5. **Sort** by `hybrid_score` descending; tie-break by `chunk_id` for determinism.
6. **Truncate** to `top_k`.

**Reciprocal Rank Fusion (alternative):**
```
rrf_score_i = sum_over_retrievers 1 / (k + rank_i)        # k=60 by default
```
Use `fusion="rrf"` to switch — useful when dense and lexical scores are not directly comparable.

`alpha` is configurable via `HYBRID_ALPHA` (default `0.5`). Setting `alpha=1.0` is dense-only; `alpha=0.0` is lexical-only.

---

## 6. Reranking

The reranker receives `(query, candidates)` and returns a re-scored, re-ordered list. It is **optional** — `RERANKER_PROVIDER=none` disables it and the pipeline still works.

- `MockReranker`: `score = overlap(query_tokens, chunk_tokens) + 0.3 * (1 - rank/N)`. Deterministic, free, useful for offline benchmarks.
- `CrossEncoderReranker`: thin client for an OpenAI-style rerank endpoint. Lazy-loaded.

Reranker MUST preserve `chunk_id` and `document_id`. Validated in `tests/unit/test_retrieval_rerank.py::TestMockReranker::test_preserves_chunk_ids`.

---

## 7. Context assembly

`ContextBuilder.build` is pure (no I/O, no randomness). Same input → same `RAGContext` every time.

Budget enforcement order:
1. Drop duplicates (by `chunk_id`).
2. Walk candidates in rank order.
3. Stop when either `max_chunks`, `max_chars`, or `max_tokens` would be exceeded.
4. Mark `truncated=True` if any candidate was dropped due to budget.

Token estimate: `chars / 4` (≈4 chars per token for English). This is a rough but reproducible proxy — it's used for budget enforcement and cost tracking, not billing.

---

## 8. Generation & citation validation

### Prompt contract

The system prompt is `SYSTEM_PROMPT` in `app/generation/prompt.py`. It explicitly forbids hallucination and requires `[cite:<chunk_id>]` markers for every factual claim.

The user prompt embeds evidence as structured records:

```
[CHUNK chunk-abc123 :: document_id=doc-456]
...chunk content...
[/CHUNK]
```

The mock LLM parses these records and stitches an answer. The OpenAI LLM sees the same prompt and is instructed to follow the same convention.

### Citation validation

`validate_citations(answer_text, context)` returns `(citations, rejected_ids, evidence_sufficient)`:
- `citations`: validated `Citation` objects, deduplicated, in order of first appearance.
- `rejected_ids`: ids found in the answer but not in the context (i.e. hallucinated or stale).
- `evidence_sufficient`: `True` iff at least one valid citation exists.

`remove_invalid_citations` rewrites rejected markers as `[citation removed]` so the final response is honest about gaps.

---

## 9. Cost tracking

Pricing is configured as flat per-1K-token prices in `Settings`. The lookup table maps model name → input/output prices.

- **Known model** (e.g. `gpt-4o-mini`): returns `pricing_known=True` and computes `total_cost = (in_tokens/1000) * in_price + (out_tokens/1000) * out_price`.
- **Unknown model** (e.g. `mock-llm-v1`): returns `pricing_known=False` and `total_cost=0.0`. **Never fabricated.**

---

## 10. Caching

`CacheRegistry` holds named `LRUCache` instances. Cache keys are SHA-256 hashes over all materially-affecting parameters:

- **Embeddings**: `(model, text)` — same text + same model → same vector.
- **Retrieval**: `(query, top_k, alpha, methods, filters)` — same query + same config → same candidates.
- **Reranking**: `(model, query, candidate_ids)` — same query + same candidates → same ranking.

`CACHE_ENABLED=false` disables the cache entirely (the default for tests). The cache is in-process only — no Redis dependency.

---

## 11. Observability

- **Logs**: `structlog` with JSON output (when `LOG_JSON=true`). Redacts keys like `api_key`, `token`, `secret` when `REDACT_LOGS=true`.
- **Metrics**: in-process `MetricsRegistry` with counters / gauges / histograms. Exposed at `/metrics`.
- **Tracing**: `RequestTrace` accumulates per-request data; `request_id` propagates via contextvars and the `X-Request-ID` response header.

---

## 12. Storage backends

| Backend | When to use | Setup |
|---------|-------------|-------|
| `InMemoryVectorStore` | Tests, dev, small datasets, anywhere PostgreSQL is not available | Default. `DATABASE_URL=sqlite+aiosqlite:///:memory:` |
| `PgVectorStore` | Production. PostgreSQL 16+ with the `vector` extension | `DATABASE_URL=postgresql+asyncpg://user:pwd@host:5432/db` and the `pgvector/pgvector:pg16` Docker image |

The `pgvector` store uses an IVFFlat index on the embedding column for cosine similarity. The schema is created on first use; no manual migration step required.

---

## 13. Failure model

All recoverable failures raise a subclass of `RAGError` with a stable `ErrorCode`. FastAPI translates these into HTTP responses with the standard `ErrorResponse` shape. The taxonomy:

| Code | Meaning |
|------|---------|
| `CONFIGURATION_ERROR` | Missing/invalid configuration |
| `DOCUMENT_PARSE_ERROR` | Unsupported or malformed content |
| `DOCUMENT_TOO_LARGE` | Document exceeded `MAX_DOCUMENT_BYTES` |
| `EMBEDDING_ERROR` | Embedding provider call failed |
| `VECTOR_STORE_ERROR` | Vector store call failed |
| `LEXICAL_SEARCH_ERROR` | BM25 search failed |
| `RETRIEVAL_ERROR` | Top-level retrieval failure |
| `RERANKING_ERROR` | Reranker call failed |
| `CONTEXT_LIMIT_ERROR` | Context exceeded configured limits |
| `LLM_PROVIDER_ERROR` | LLM HTTP/transport error |
| `LLM_TIMEOUT` | LLM timed out or was rate-limited |
| `LLM_OUTPUT_ERROR` | LLM returned malformed output |
| `INVALID_CITATION` | Citation validation failed (fatal) |
| `EVALUATION_ERROR` | Evaluation harness failure |
| `VALIDATION_ERROR` | Request validation failed |
| `NOT_FOUND` | Resource not found |
| `RATE_LIMITED` | Client should slow down |

---

## 14. Engineering principles (mapping to spec)

| Spec principle | How it's satisfied |
|----------------|--------------------|
| P-01 Modularity | Each major responsibility is a module with a single facade. |
| P-02 Provider independence | `EmbeddingProvider` and `LLMProvider` are `Protocol`s; mock + OpenAI impls are swappable via env. |
| P-03 Deterministic infrastructure | Deterministic chunk IDs, deterministic mock providers, deterministic context builder. |
| P-04 Structured interfaces | Pydantic models at every service boundary. |
| P-05 Testability | Mock providers + in-memory store mean unit tests need no external services. |
| P-06 Explicit failure | `RAGError` taxonomy with stable codes; no silent degradation. |
| P-07 Observable execution | structlog + MetricsRegistry + RequestTrace + `/metrics`. |
| P-08 Cost awareness | Context budget enforced; cost tracking reports `pricing_known=false` for unknown models. |
| P-09 Evidence-based claims | Benchmark harness in `benchmarks/`; results in `EVALUATION.md`. |
| P-10 Incremental implementation | Phased todo list in this implementation. |
