# Production RAG System

Production-oriented Retrieval-Augmented Generation system built with Python, FastAPI, PostgreSQL/pgvector, hybrid retrieval, reranking, grounded generation, evaluation, and cost-aware LLM infrastructure.

A production-oriented RAG architecture designed around the complete retrieval-to-generation pipeline:
```
Ingestion → Chunking → Embeddings → Vector Search → BM25 → Hybrid Retrieval → Reranking → Context Assembly → LLM Generation → Citations → Evaluation
```
The system is designed to demonstrate how a RAG application can move beyond a basic embed → retrieve → prompt implementation toward a measurable, observable, and production-oriented AI system.

> **Engineering evidence over feature count.** Every claim in this README is backed by reproducible benchmark runs and unit tests. The system runs offline by default (mock LLM, mock embeddings, in-memory vector store) so you can develop and test without an OpenAI key.

---

## Highlights

- **Hybrid retrieval.** Dense (pgvector cosine) + lexical (BM25) fused with a configurable `alpha`. Also exposes reciprocal-rank fusion (RRF) as an alternative.
- **Reranking.** Optional second stage; the system works with or without it. Default mock reranker is deterministic.
- **Grounded generation.** Structured prompt instructs the LLM to cite every claim; invalid citations are stripped before the response leaves the API.
- **Evaluation framework.** Recall@K, Precision@K, MRR, Hit Rate, NDCG, citation correctness, groundedness, answer relevance, answer completeness.
- **Cost tracking.** USD-per-1K-token pricing table. Unknown pricing → `pricing_known=false`, **never** a fabricated number.
- **Observability.** Structured JSON logs, in-process metrics registry (counters / gauges / histograms), per-request tracing, `/metrics` endpoint.
- **Caching.** LRU cache for embeddings, retrieval, and reranking. Cache keys are stable hashes of all materially-affecting parameters.
- **Provider-neutral.** LLM and embeddings both go through `Protocol` interfaces. Swap mock → OpenAI by flipping one env var. Works with any OpenAI-compatible server (LM Studio, vLLM, Ollama shim).
- **Production-ready failure model.** Structured `RAGError` taxonomy with stable error codes. Failed documents in a batch never abort the run.
- **Tested.** 154 tests pass offline (unit + API + evaluation). Lint (ruff), types (mypy), and security (bandit) are all green.
- **Reproducible.** Deterministic chunk IDs, deterministic mock providers, deterministic context assembly. Same input → same output, every run.
- **No silent production fallback.** A production `DATABASE_URL` that cannot connect raises `VECTOR_STORE_ERROR` at startup — the system never silently puts production data in RAM.

---

## Quickstart (offline, no API key)

```bash
# 1. Install dev deps in your venv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the test suite
pytest -q                            # all unit + API tests
pytest -m benchmark                   # the benchmark test (saves benchmarks/last_benchmark.json)

# 3. Boot the API (uses mock LLM + mock embeddings + in-memory store)
uvicorn app.main:app --reload
# Open http://localhost:8000/docs for the Swagger UI.

# 4. End-to-end smoke through the API
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"text":"Hybrid search combines dense and lexical retrieval with alpha fusion."}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is hybrid search?","top_k":3,"rerank":true}'
```

## Quickstart (production with PostgreSQL + OpenAI)

```bash
cp .env.example .env
# Edit .env: set LLM_PROVIDER=openai, EMBEDDING_PROVIDER=openai,
#           LLM_API_KEY=sk-..., EMBEDDING_API_KEY=sk-...

docker compose up -d
curl http://localhost:8000/health
```

The first request to `/documents` creates the `vector` extension and the `rag_chunks` table automatically.

---

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full diagram and component responsibilities. See [`EVALUATION.md`](./EVALUATION.md) for the metrics definitions and the latest benchmark numbers. See [`BENCHMARKS.md`](./BENCHMARKS.md) for the reproducible benchmark methodology.

```
FastAPI → RAGOrchestrator → Retrieval (dense + lexical → hybrid → rerank) →
       ContextBuilder → Generation (LLM provider) → Citation validation → Answer + Metrics
```

---

## Measured benchmark (mock pipeline)

Reproduced by `pytest -m benchmark -v`. The full JSON outputs are committed at:
- `benchmarks/last_benchmark.json` — latency + token + cost
- `benchmarks/last_evaluation.json` — retrieval + answer quality metrics

Methodology is documented in [`BENCHMARKS.md`](./BENCHMARKS.md) and [`EVALUATION.md`](./EVALUATION.md).

### Performance (18 docs × 51 queries, mock pipeline, single repeat)

| Stage             | Mean (ms) | p50 (ms) | p95 (ms) |
|-------------------|-----------|----------|----------|
| Retrieval         | < 1       | < 1      | < 2      |
| Reranking         | < 1       | < 1      | < 1      |
| Generation        | < 1       | < 1      | < 1      |
| End-to-end        | < 2       | < 2      | < 3      |

> These are an **upper bound on latency** — they use the deterministic mock
> LLM, mock embedder, and in-memory vector store. Real OpenAI + pgvector
> numbers will be larger; the same harness produces them when you set the
> env vars (see `BENCHMARKS.md` §2 Option C).

### Quality (51 queries against 18 docs, mock pipeline, top_k=5)

| Retrieval metric | Value |
|------------------|------|
| Hit Rate@5       | 0.96 |
| MRR              | 0.84 |
| NDCG@5           | 0.54 |
| Recall@5         | 0.48 |
| Precision@5      | 0.20 |

| Answer metric         | Value |
|-----------------------|-------|
| Answer relevance      | 1.00 |
| Answer keyword overlap| 0.93 |
| Evidence sufficient   | 1.00 |

| Cost                       | Value |
|----------------------------|-------|
| Pricing known              | false (mock LLM has no real price) |
| Total estimated USD        | 0.0 (never fabricated) |

> These numbers measure the **retrieval architecture** (hybrid fusion +
> reranking) against a known gold set on the system's own domain docs. They
> are **not** a claim about real-LLM answer quality — that requires an
> OpenAI-backed run with an annotated gold set, which the harness supports
> via env-var configuration (`BENCHMARKS.md` §2 Option C).

> No quality percentage is claimed without benchmark evidence. Quality
> claims for real LLMs require an OpenAI-backed run plus an annotated gold
> set.

---

## API endpoints

| Method | Path                       | Purpose |
|--------|----------------------------|---------|
| GET    | `/health`                  | Liveness + provider config + chunk count |
| POST   | `/documents`               | Ingest one text document |
| POST   | `/documents/batch`         | Ingest many documents (failure-isolated) |
| POST   | `/documents/upload`        | Multipart file upload |
| POST   | `/documents/bytes`         | Base64-encoded body |
| GET    | `/documents/{id}`          | Document metadata + chunk count |
| POST   | `/search`                  | Hybrid retrieval only (no generation) |
| POST   | `/query`                   | End-to-end RAG query |
| POST   | `/evaluate`                | Run evaluation cases and return metrics |
| GET    | `/metrics`                 | Counters, gauges, histograms, cache stats |

Error responses use a stable JSON shape:

```json
{
  "code": "DOCUMENT_TOO_LARGE",
  "message": "Document is 22000000 bytes; limit is 10485760",
  "details": {"bytes": 22000000, "limit": 10485760},
  "request_id": "req-..."
}
```

The error taxonomy lives in [`app/domain/errors.py`](./app/domain/errors.py).

---

## Configuration

All configuration is environment-driven (see [`.env.example`](./.env.example)). The most important knobs:

| Variable | Default | Notes |
|----------|---------|-------|
| `DATABASE_URL` | `sqlite+aiosqlite:///:memory:` | Use `postgresql+asyncpg://...` for production |
| `LLM_PROVIDER` | `mock` | `mock` \| `openai` |
| `EMBEDDING_PROVIDER` | `mock` | `mock` \| `openai` |
| `RERANKER_PROVIDER` | `mock` | `mock` \| `none` \| `cross-encoder` |
| `TOP_K` | `10` | Retrieval depth |
| `HYBRID_ALPHA` | `0.5` | Weight on dense vs lexical |
| `MAX_CONTEXT_TOKENS` | `3500` | LLM context budget |
| `CACHE_ENABLED` | `true` | LRU cache for embeddings / retrieval / rerank |

Secrets (`LLM_API_KEY`, `EMBEDDING_API_KEY`) must come from the environment. They are never logged when `REDACT_LOGS=true`.

---

## Repository layout

```
production-rag/
├── app/                  # The application
│   ├── api/              # FastAPI routes + request/response models
│   ├── config/           # Pydantic Settings
│   ├── domain/           # Typed models + error taxonomy
│   ├── ingestion/        # Loader + IngestionService
│   ├── chunking/         # Fixed / recursive / sentence chunkers
│   ├── embeddings/       # Mock + OpenAI-compatible embedding providers
│   ├── storage/          # In-memory + pgvector stores; BM25 lexical index
│   ├── retrieval/        # Hybrid fusion (linear + RRF)
│   ├── reranking/        # Mock + cross-encoder rerankers
│   ├── context/          # Deterministic context builder
│   ├── generation/       # LLM provider, prompt builder, citation validator
│   ├── evaluation/       # Retrieval + answer metrics
│   ├── caching/          # LRU + stable-key helpers
│   ├── observability/    # Structlog + metrics + request tracing
│   ├── pipeline.py       # RAGOrchestrator
│   └── main.py           # create_app()
├── benchmarks/           # run_benchmark + last_benchmark.json + last_evaluation.json
├── datasets/             # Built-in 18-doc / 51-query evaluation dataset
├── docker/               # Dockerfile
├── scripts/              # smoke_test.py
├── tests/                # unit / api / evaluation / integration
├── docs/                 # design notes
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── ARCHITECTURE.md
├── EVALUATION.md
├── BENCHMARKS.md
├── IMPLEMENTATION_PLAN.md
└── README.md
```

---

## Testing

```bash
pytest                          # everything not marked integration/benchmark/slow
pytest -m unit                 # pure unit tests, no external services
pytest -m api                  # FastAPI TestClient tests
pytest -m benchmark            # run the benchmark harness
pytest -m integration          # requires PostgreSQL + real OpenAI (set env vars)
```

Integration tests are skipped by default; see [`tests/integration/`](./tests/integration/) for the env vars to enable them.

---

## Quality gates

| Tool | What it checks | Status |
|------|----------------|--------|
| `ruff check` | lint + formatting | ✅ All checks passed |
| `mypy app` | type safety | ✅ 0 issues in 39 source files |
| `bandit -r app` | security | ✅ 0 issues |
| `pytest -q` | unit + API tests | ✅ 154 passed, 2 skipped (integration) |
| `pytest -m benchmark` | evaluation + performance | ✅ 2 passed |

Run them all:

```bash
ruff check app tests benchmarks datasets scripts
mypy app
bandit -c pyproject.toml -r app
pytest -q
pytest -m benchmark -v
```

---

## What this project demonstrates

This repository is intended to demonstrate practical engineering capability in:

RAG architecture
Vector search
Hybrid retrieval
Reranking
LLM integration
Context engineering
Grounded generation
Evaluation
LLM cost optimization
AI observability
FastAPI backend architecture
PostgreSQL / pgvector
Production-oriented Python engineering

It is intentionally complementary to an agentic AI system rather than another implementation of the same problem.

---

## License

MIT. See [`LICENSE`](./LICENSE).
