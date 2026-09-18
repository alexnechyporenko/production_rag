# Implementation Plan — Production RAG System

This document is the **current engineering roadmap** for the Production RAG
system. It mirrors the 20-phase execution model used during the build:

> Phase 00 → Phase 19, executed sequentially. Each phase has explicit
> acceptance criteria; a phase is complete only when **Implementation +
> Tests + Static analysis + Acceptance criteria + DoD** all PASS.

For the historical build log, see `docs/IMPLEMENTATION_HISTORY.md`
(the build happened in fewer passes than 20 distinct phases — several
phases were landed together — but every acceptance criterion is met).

---

## Phase 00 — Repository Bootstrap

**Goal**: working repo, config, dependency manifest, dev tooling.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-00.1 Repository structure exists (`app/`, `tests/`, `benchmarks/`, `datasets/`, `scripts/`, `docs/`, `docker/`) ✅
- AC-00.2 Python package imports successfully ✅
- AC-00.3 pytest executes ✅
- AC-00.4 Ruff executes ✅
- AC-00.5 mypy executes against configured source ✅
- AC-00.6 No secrets are committed ✅

---

## Phase 01 — Domain Models and Configuration

**Goal**: typed domain contracts + centralized configuration.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-01.1 Models validate required fields ✅
- AC-01.2 Invalid data is rejected ✅
- AC-01.3 Configuration loads from environment ✅
- AC-01.4 No provider credentials are hard-coded ✅
- AC-01.5 Unit tests cover models ✅

---

## Phase 02 — Document Ingestion

**Goal**: deterministic loader + parser + normalizer.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-02.1 Supported documents load (text/plain, text/markdown, application/json, text/html, application/pdf) ✅
- AC-02.2 Unsupported formats fail explicitly ✅
- AC-02.3 Metadata is preserved ✅
- AC-02.4 Empty documents are rejected ✅
- AC-02.5 Malformed documents produce structured errors ✅
- AC-02.6 Unit tests cover normal and failure cases ✅

---

## Phase 03 — Chunking

**Goal**: stable, deterministic chunks with metadata propagation.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-03.1 Chunking is deterministic ✅
- AC-03.2 Chunk IDs are stable ✅ (`deterministic_chunk_id = sha1(doc_id, seq, content[:128])`)
- AC-03.3 Document metadata propagates ✅
- AC-03.4 Chunk size constraints are enforced ✅
- AC-03.5 Empty chunks are not emitted ✅
- AC-03.6 Tests cover boundary conditions ✅

---

## Phase 04 — Embedding Abstraction

**Goal**: provider-neutral embeddings.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-04.1 Provider interface is explicit ✅ (`EmbeddingProvider` Protocol)
- AC-04.2 Real provider configuration is externalized ✅
- AC-04.3 Fake provider works offline ✅ (`MockEmbeddingProvider`)
- AC-04.4 Embedding dimensions are validated ✅
- AC-04.5 Provider errors are mapped to structured failures ✅ (`EmbeddingError`)
- AC-04.6 Tests do not require network access ✅

---

## Phase 05 — PostgreSQL + pgvector Storage

**Goal**: persist chunks, embeddings, and metadata with vector search.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-05.1 Database schema can be created from scratch ✅ (auto-init on first use)
- AC-05.2 Documents can be inserted ✅
- AC-05.3 Chunks can be inserted ✅
- AC-05.4 Embeddings can be stored ✅
- AC-05.5 Metadata can be retrieved ✅
- AC-05.6 Vector dimension mismatch is detected ✅
- AC-05.7 Integration tests exist ✅ (`tests/integration/test_pgvector_openai.py`)

---

## Phase 06 — Ingestion Pipeline

**Goal**: Loader → Normalizer → Chunker → Embedder → Storage.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-06.1 A document can travel through the complete ingestion pipeline ✅
- AC-06.2 Partial failures are explicit ✅ (failure isolation in `ingest_documents`)
- AC-06.3 Re-running ingestion does not unintentionally duplicate data ✅ (upsert semantics in both `InMemoryVectorStore` and `LexicalIndex`)
- AC-06.4 Document/chunk IDs remain stable ✅
- AC-06.5 Integration tests cover the pipeline ✅

---

## Phase 07 — Dense Retrieval

**Goal**: semantic vector retrieval.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-07.1 Query embedding is generated ✅
- AC-07.2 Top-k retrieval works ✅
- AC-07.3 Similarity scores are returned ✅
- AC-07.4 Metadata filtering works ✅
- AC-07.5 Results preserve chunk IDs ✅
- AC-07.6 Retrieval latency is measurable ✅

---

## Phase 08 — Lexical Retrieval

**Goal**: BM25-style retrieval independent from dense.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-08.1 Lexical queries return relevant candidates ✅
- AC-08.2 Ranking scores are returned ✅
- AC-08.3 Top-k is configurable ✅
- AC-08.4 Tests cover exact and partial term matches ✅
- AC-08.5 No vector search is required for lexical retrieval ✅

---

## Phase 09 — Hybrid Retrieval

**Goal**: deterministic dense + lexical fusion.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-09.1 Both retrieval strategies execute ✅
- AC-09.2 Results are merged deterministically ✅
- AC-09.3 Duplicate chunks are removed ✅
- AC-09.4 Fusion parameters are configurable ✅ (`alpha`, fusion strategy)
- AC-09.5 Ranking behavior is tested ✅
- AC-09.6 Dense-only mode remains available ✅
- AC-09.7 Lexical-only mode remains available ✅

---

## Phase 10 — Reranking

**Goal**: optional second-stage reranking with source-traceability.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-10.1 Reranking accepts query + candidates ✅
- AC-10.2 Candidate count is bounded ✅ (`top_k`)
- AC-10.3 Scores are returned ✅
- AC-10.4 Source identifiers survive reranking ✅ (tested)
- AC-10.5 Reranking can be disabled ✅ (`RERANKER_PROVIDER=none`)
- AC-10.6 Reranking failures are explicit ✅ (`RerankingError`)
- AC-10.7 Tests cover ordering ✅

---

## Phase 11 — Context Assembly and Budgeting

**Goal**: bounded, deduplicated, prioritized LLM context.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-11.1 Duplicate chunks are removed ✅
- AC-11.2 Context size is bounded ✅
- AC-11.3 Highest-ranked evidence is prioritized ✅
- AC-11.4 Source identifiers are preserved ✅
- AC-11.5 Oversized context is handled deterministically ✅
- AC-11.6 Context metrics are emitted ✅

---

## Phase 12 — LLM Generation

**Goal**: provider-neutral grounded generation.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-12.1 Provider abstraction exists ✅ (`LLMProvider` Protocol)
- AC-12.2 Fake provider supports offline tests ✅ (`MockLLMProvider`)
- AC-12.3 Real provider can be configured ✅ (`OpenAILLMProvider`)
- AC-12.4 Prompt contains retrieved context ✅
- AC-12.5 LLM output is validated ✅
- AC-12.6 Timeout is handled ✅ (`LLMTimeoutError`)
- AC-12.7 Provider failure is structured ✅ (`LLMProviderError`)
- AC-12.8 Token usage is captured when available ✅ (`TokenUsage`)

---

## Phase 13 — Citations and Grounding Validation

**Goal**: answers cannot silently reference nonexistent sources.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-13.1 Valid citation IDs are accepted ✅
- AC-13.2 Invalid citation IDs are detected ✅
- AC-13.3 Citation metadata maps to actual chunks ✅
- AC-13.4 Unsupported claims trigger explicit low-evidence state ✅
- AC-13.5 Citation validation is deterministic ✅
- AC-13.6 Tests cover invalid citations ✅

---

## Phase 14 — Token, Cost and Performance Tracking

**Goal**: every RAG request produces a measurable execution record.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-14.1 Metrics are attached to each query ✅
- AC-14.2 Token usage is recorded ✅
- AC-14.3 Pricing is configuration-driven ✅
- AC-14.4 Unknown pricing does not fabricate costs ✅ (`pricing_known=false`)
- AC-14.5 Latency is measured ✅
- AC-14.6 Metrics are testable ✅

---

## Phase 15 — Caching

**Goal**: deterministic, parameter-keyed caching with measurable hit/miss.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-15.1 Cache can be enabled/disabled ✅
- AC-15.2 Cache keys are deterministic ✅ (`stable_key` = SHA-256)
- AC-15.3 Model/config changes invalidate relevant keys ✅
- AC-15.4 Cache hits are measurable ✅
- AC-15.5 Cache misses are measurable ✅
- AC-15.6 Stale data behavior is defined ✅ (LRU eviction)
- AC-15.7 Tests cover hits and misses ✅

---

## Phase 16 — FastAPI API Layer

**Goal**: all spec endpoints exposed with structured errors.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-16.1 API starts successfully ✅
- AC-16.2 Request validation works ✅
- AC-16.3 Errors use structured responses ✅ (`ErrorResponse` with stable `code`)
- AC-16.4 `/query` executes the RAG pipeline ✅
- AC-16.5 `/search` exposes retrieval results ✅
- AC-16.6 `/health` reports service status ✅
- AC-16.7 API tests exist ✅

---

## Phase 17 — Evaluation Framework

**Goal**: reproducible RAG evaluation against an annotated dataset.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-17.1 Evaluation dataset exists ✅ (18 docs, 51 queries)
- AC-17.2 Dataset is versioned ✅ (committed to repo)
- AC-17.3 Retrieval metrics execute ✅ (Recall@K, Precision@K, MRR, Hit Rate, NDCG)
- AC-17.4 Answer metrics execute ✅ (groundedness, citation correctness, relevance, completeness, keyword overlap)
- AC-17.5 Results are persisted ✅ (`benchmarks/last_evaluation.json`)
- AC-17.6 Baseline results are reproducible ✅ (deterministic mock pipeline)
- AC-17.7 No unsupported quality claims are generated ✅

---

## Phase 18 — Docker, Security and Observability

**Goal**: reproducible local deployment; production-grade hardening.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-18.1 Docker image builds ✅ (multi-stage Dockerfile)
- AC-18.2 Docker Compose starts required services ✅ (Postgres 16 + pgvector + rag)
- AC-18.3 Database initializes ✅
- AC-18.4 API becomes healthy ✅ (HEALTHCHECK in Dockerfile)
- AC-18.5 Secrets are environment-driven ✅
- AC-18.6 Sensitive credentials are not logged ✅ (structlog redaction)
- AC-18.7 Bandit passes ✅ (0 issues)
- AC-18.8 Health checks work ✅

---

## Phase 19 — Final Benchmark, Audit and GitHub Release

**Goal**: portfolio-grade engineering artifact with reproducible evidence.

**Status**: ✅ PASS

**Acceptance Criteria**:
- AC-19.1 Full test suite passes ✅ (154 tests; 2 integration skipped without keys)
- AC-19.2 Ruff passes ✅
- AC-19.3 Security scan passes ✅
- AC-19.4 Mypy status is documented ✅ (0 issues in 39 source files)
- AC-19.5 Docker deployment works ✅
- AC-19.6 Evaluation benchmark runs reproducibly ✅
- AC-19.7 Performance benchmark runs reproducibly ✅
- AC-19.8 README contains architecture diagram ✅
- AC-19.9 README contains concrete benchmark evidence ✅
- AC-19.10 README does not contain unsupported claims ✅
- AC-19.11 `.env.example` contains all required configuration ✅
- AC-19.12 Repository is ready for public GitHub publication ✅

---

## Global Quality Gates (must hold across all phases)

| Gate | Tool | Status |
|------|------|--------|
| Tests | `pytest` | ✅ 154 passed, 2 skipped (integration) |
| Static analysis | `ruff` | ✅ clean |
| Typing | `mypy` | ✅ 0 issues in 39 source files |
| Security | `bandit` | ✅ 0 issues |
| Documentation | README + ARCHITECTURE + EVALUATION + BENCHMARKS | ✅ |
| Scope | Implementation matches this plan | ✅ |

---

## Reproducing the latest results

```bash
# Run the full test suite (offline, mock providers)
pytest -q

# Run the benchmark + evaluation (also offline)
pytest -m benchmark -v

# Inspect the results
cat benchmarks/last_benchmark.json
cat benchmarks/last_evaluation.json

# Run quality gates
ruff check app tests benchmarks datasets scripts
mypy app
bandit -c pyproject.toml -r app
```
