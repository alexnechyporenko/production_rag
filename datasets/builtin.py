"""Built-in evaluation dataset for the Production RAG system.

This is a small-but-realistic evaluation set with:

  * 18 short technical documents covering the system's own domain
    (RAG, retrieval, embeddings, pgvector, FastAPI, evaluation, cost tracking,
    caching, observability, security, Docker).
  * 50 evaluation queries with annotated `relevant_document_ids` (the gold set
    for retrieval metrics) and `expected_answer_keywords` (the gold set for
    answer metrics).

The dataset is intentionally **deterministic**: the same docs always produce
the same `deterministic_chunk_id`s, so re-running the benchmark or evaluation
yields the same numbers. This is what lets us claim reproducibility (NFR-03).

The dataset is intentionally **small enough to run in CI** in a few seconds
on the mock pipeline, while being **large enough to make Recall@5 / MRR /
groundedness numbers meaningful**. For a real research-grade benchmark,
replace `DATASET_DOCS` / `DATASET_CASES` with a domain-specific dataset of
your own (see EVALUATION.md §6).
"""

from __future__ import annotations

# ============================================================================
# Documents — 18 short technical texts covering the RAG system's own domain
# ============================================================================

DATASET_DOCS: list[dict[str, str]] = [
    {
        "title": "PostgreSQL overview",
        "text": (
            "PostgreSQL is a powerful open-source relational database. "
            "It supports advanced features such as JSONB, full-text search, "
            "and the pgvector extension for vector storage. pgvector enables "
            "efficient cosine similarity search over high-dimensional vectors."
        ),
    },
    {
        "title": "pgvector extension",
        "text": (
            "The pgvector extension adds vector types to PostgreSQL. It supports "
            "L2 distance, inner product, and cosine distance. The recommended "
            "index for cosine similarity is IVFFlat. For very large corpora, "
            "HNSW is also supported in pgvector 0.5+."
        ),
    },
    {
        "title": "FastAPI overview",
        "text": (
            "FastAPI is a modern Python web framework for building APIs. "
            "It uses Pydantic for validation, Starlette for ASGI handling, "
            "and supports async endpoints. FastAPI auto-generates OpenAPI "
            "documentation and provides strong type safety with mypy."
        ),
    },
    {
        "title": "Pydantic models",
        "text": (
            "Pydantic v2 provides data validation and settings management via "
            "Python type hints. BaseModel is used for typed domain models, while "
            "BaseSettings (pydantic-settings) is used for environment-driven "
            "configuration. Field validators run automatically on assignment."
        ),
    },
    {
        "title": "Hybrid search",
        "text": (
            "Hybrid search combines dense vector retrieval with lexical "
            "retrieval. Dense retrieval captures semantic similarity, while "
            "lexical retrieval such as BM25 captures exact term matches. "
            "A weighted alpha fusion strategy combines the two scores."
        ),
    },
    {
        "title": "BM25 lexical retrieval",
        "text": (
            "BM25 is a probabilistic ranking function for lexical retrieval. "
            "It scores documents based on term frequency, inverse document "
            "frequency, and document length normalization. The rank_bm25 "
            "Python package provides a pure-Python BM25Okapi implementation."
        ),
    },
    {
        "title": "Dense vector retrieval",
        "text": (
            "Dense vector retrieval computes cosine similarity between a query "
            "embedding and document embeddings stored in a vector database. "
            "It captures semantic similarity even when query and document share "
            "no exact terms. pgvector's IVFFlat index accelerates this search."
        ),
    },
    {
        "title": "Reranking",
        "text": (
            "Reranking is an optional second stage of retrieval. Given a set "
            "of candidate chunks, a cross-encoder model scores each query and "
            "chunk pair. Reranking can improve precision but adds latency. "
            "The system must function without a reranker."
        ),
    },
    {
        "title": "RAG overview",
        "text": (
            "Retrieval-Augmented Generation grounds a language model's answer "
            "in retrieved evidence. The retrieved chunks become the only "
            "source of truth for the answer, preventing hallucination. "
            "Citations allow the reader to verify each claim."
        ),
    },
    {
        "title": "Citation validation",
        "text": (
            "Citation validation checks that every [cite:chunk_id] marker in "
            "an LLM answer references an actual chunk in the retrieved context. "
            "Invalid citations are stripped before the response is returned. "
            "This prevents the system from silently endorsing hallucinations."
        ),
    },
    {
        "title": "Context assembly",
        "text": (
            "Context assembly deduplicates retrieved chunks, enforces a maximum "
            "chunk count, and enforces a token or character budget. It preserves "
            "source metadata and citation identifiers. The output must be "
            "deterministic for the same input."
        ),
    },
    {
        "title": "Token and cost tracking",
        "text": (
            "Token tracking records input, output, and total tokens per LLM call. "
            "Cost is estimated from a configuration-driven pricing table. "
            "Unknown pricing returns pricing_known=false and zero cost, never "
            "a fabricated number."
        ),
    },
    {
        "title": "Caching",
        "text": (
            "Caching reduces repeated computation. The system supports caches for "
            "embeddings, retrieval, and reranking. Cache keys are deterministic "
            "hashes over all parameters that materially affect the result, "
            "including model name, query text, and configuration values."
        ),
    },
    {
        "title": "Observability",
        "text": (
            "Observability combines structured logging, in-process metrics, and "
            "per-request tracing. Structlog emits JSON logs with redaction of "
            "sensitive fields. The metrics registry exposes counters, gauges, "
            "and histograms via the /metrics endpoint."
        ),
    },
    {
        "title": "Security",
        "text": (
            "Security best practices include: never hard-code secrets, validate "
            "input, limit document and query sizes, prevent path traversal, "
            "avoid arbitrary code execution, sanitize logs, and provide safe "
            "failure behavior. Bandit scans for common Python vulnerabilities."
        ),
    },
    {
        "title": "Docker deployment",
        "text": (
            "Docker deployment uses a multi-stage build: a builder stage installs "
            "dependencies, a runtime stage copies only the installed packages and "
            "the application source. Docker Compose brings up PostgreSQL with "
            "pgvector and the FastAPI service with health checks."
        ),
    },
    {
        "title": "Evaluation metrics",
        "text": (
            "Retrieval metrics include Recall@K, Precision@K, MRR, Hit Rate, "
            "and NDCG. Answer metrics include groundedness, citation correctness, "
            "answer relevance, and answer completeness. All metrics are computed "
            "from the gold annotation in the evaluation dataset."
        ),
    },
    {
        "title": "Chunking",
        "text": (
            "Chunking divides a document into smaller pieces with stable "
            "identifiers. The system supports fixed-size, sentence-based, and "
            "recursive chunking strategies. Overlap is configurable to preserve "
            "context across chunk boundaries."
        ),
    },
]


# ============================================================================
# Evaluation cases — 50 queries with gold annotations
# ============================================================================

# Helper: build cases from a compact (query, [relevant_titles], [keywords]) tuple.
_CASES_RAW: list[tuple[str, list[str], list[str]]] = [
    # --- Retrieval / Hybrid search ----------------------------------------
    ("What is hybrid search?", ["Hybrid search"], ["hybrid", "dense", "lexical"]),
    ("How does hybrid retrieval combine dense and lexical search?", ["Hybrid search"], ["alpha", "fusion"]),
    ("What does the alpha parameter control in hybrid retrieval?", ["Hybrid search"], ["alpha", "weight", "fusion"]),
    ("What does dense retrieval capture that lexical retrieval does not?", ["Dense vector retrieval"], ["semantic", "similarity"]),
    ("What does BM25 capture that dense retrieval does not?", ["BM25 lexical retrieval"], ["exact", "term", "matches"]),

    # --- BM25 specifics ---------------------------------------------------
    ("What scoring function does BM25 use?", ["BM25 lexical retrieval"], ["term", "frequency", "idf"]),
    ("Which Python package provides BM25Okapi?", ["BM25 lexical retrieval"], ["rank_bm25"]),

    # --- Dense retrieval specifics ----------------------------------------
    ("What distance metric does dense vector retrieval use?", ["Dense vector retrieval"], ["cosine", "similarity"]),
    ("What index does pgvector recommend for cosine similarity?", ["pgvector extension"], ["ivfflat", "index"]),

    # --- pgvector + PostgreSQL --------------------------------------------
    ("What is pgvector?", ["pgvector extension"], ["extension", "vector", "postgresql"]),
    ("What vector distances does pgvector support?", ["pgvector extension"], ["l2", "inner", "cosine"]),
    ("What is PostgreSQL?", ["PostgreSQL overview"], ["relational", "database"]),
    ("What advanced features does PostgreSQL support?", ["PostgreSQL overview"], ["jsonb", "full-text", "pgvector"]),

    # --- FastAPI / Pydantic -----------------------------------------------
    ("What is FastAPI?", ["FastAPI overview"], ["web", "framework", "python"]),
    ("What does FastAPI use for validation?", ["FastAPI overview"], ["pydantic"]),
    ("What ASGI handler does FastAPI use?", ["FastAPI overview"], ["starlette"]),
    ("What is Pydantic v2 used for?", ["Pydantic models"], ["validation", "settings"]),
    ("What does BaseSettings do?", ["Pydantic models"], ["environment", "configuration"]),

    # --- Reranking --------------------------------------------------------
    ("Is reranking required?", ["Reranking"], ["optional", "second", "stage"]),
    ("What does a cross-encoder reranker do?", ["Reranking"], ["cross-encoder", "scores", "pair"]),
    ("What is the trade-off of reranking?", ["Reranking"], ["precision", "latency"]),

    # --- RAG / grounding --------------------------------------------------
    ("What is RAG?", ["RAG overview"], ["retrieval", "augmented", "generation"]),
    ("How does RAG prevent hallucination?", ["RAG overview"], ["retrieved", "evidence", "source"]),
    ("Why are citations useful in RAG?", ["RAG overview"], ["verify", "claim"]),

    # --- Citations --------------------------------------------------------
    ("How are invalid citations handled?", ["Citation validation"], ["stripped", "removed"]),
    ("What does citation validation check?", ["Citation validation"], ["chunk_id", "context"]),
    ("What happens to hallucinated citations?", ["Citation validation"], ["stripped", "prevent"]),

    # --- Context ----------------------------------------------------------
    ("What does context assembly do?", ["Context assembly"], ["deduplicates", "budget"]),
    ("Why must context assembly be deterministic?", ["Context assembly"], ["deterministic"]),
    ("What does context assembly enforce?", ["Context assembly"], ["max", "chunk", "token"]),

    # --- Cost / Token tracking -------------------------------------------
    ("How is cost estimated?", ["Token and cost tracking"], ["pricing", "table"]),
    ("What happens when pricing is unknown?", ["Token and cost tracking"], ["pricing_known", "false", "fabricated"]),
    ("What does token tracking record?", ["Token and cost tracking"], ["input", "output", "total"]),

    # --- Caching ----------------------------------------------------------
    ("What does caching reduce?", ["Caching"], ["repeated", "computation"]),
    ("What makes a cache key valid?", ["Caching"], ["deterministic", "parameters"]),
    ("What can be cached?", ["Caching"], ["embeddings", "retrieval", "reranking"]),

    # --- Observability ----------------------------------------------------
    ("What does observability combine?", ["Observability"], ["logging", "metrics", "tracing"]),
    ("How are sensitive logs handled?", ["Observability"], ["redaction", "structured"]),
    ("What metrics types are exposed?", ["Observability"], ["counters", "gauges", "histograms"]),

    # --- Security ---------------------------------------------------------
    ("What security practices does the system follow?", ["Security"], ["validate", "limit", "sanitize"]),
    ("What tool scans for Python vulnerabilities?", ["Security"], ["bandit"]),

    # --- Docker -----------------------------------------------------------
    ("What stages does the Dockerfile use?", ["Docker deployment"], ["multi-stage", "builder", "runtime"]),
    ("What does Docker Compose bring up?", ["Docker deployment"], ["postgres", "pgvector", "fastapi"]),

    # --- Evaluation -------------------------------------------------------
    ("What retrieval metrics are supported?", ["Evaluation metrics"], ["recall", "precision", "mrr", "ndcg"]),
    ("What answer metrics are supported?", ["Evaluation metrics"], ["groundedness", "citation", "relevance"]),
    ("What is the gold annotation used for?", ["Evaluation metrics"], ["retrieval", "answer", "metrics"]),

    # --- Chunking ---------------------------------------------------------
    ("What chunking strategies are supported?", ["Chunking"], ["fixed", "sentence", "recursive"]),
    ("Why is overlap configurable?", ["Chunking"], ["context", "boundaries"]),
    ("What makes chunk IDs stable?", ["Chunking"], ["deterministic", "stable"]),

    # --- Cross-cutting / edge cases --------------------------------------
    ("Can the system work without a reranker?", ["Reranking"], ["optional", "without"]),
    ("What is the role of citations in a RAG answer?", ["Citation validation", "RAG overview"], ["verify", "claim"]),
]


def _build_cases() -> list[dict[str, object]]:
    """Resolve the relevant_titles into relevant_document_ids using the
    deterministic_chunk_id of the first chunk of each named document.
    """
    from app.domain import deterministic_chunk_id

    # Build a title -> doc_text map for resolving titles to chunk ids.
    title_to_text: dict[str, str] = {d["title"]: d["text"] for d in DATASET_DOCS}
    cases: list[dict[str, object]] = []
    for i, (query, relevant_titles, keywords) in enumerate(_CASES_RAW):
        relevant_doc_ids: list[str] = []
        # We don't have doc ids at module load time — they're generated by
        # the IngestionService. So we annotate with titles; the evaluator
        # resolves them after ingestion.
        for title in relevant_titles:
            if title not in title_to_text:
                raise RuntimeError(f"DATASET_CASES references unknown title: {title!r}")
            # For deterministic pre-computation, also include a synthetic
            # doc_id-keyed chunk id (used by tests that don't run ingestion).
            text = title_to_text[title]
            # Use a stable doc_id derived from the title so tests can predict it.
            doc_id = f"doc-{title.lower().replace(' ', '-')}"
            relevant_doc_ids.append(deterministic_chunk_id(doc_id, 0, text))
        cases.append({
            "case_id": f"case-{i:03d}",
            "query": query,
            "relevant_document_titles": relevant_titles,  # resolved at eval time
            "relevant_chunk_ids": relevant_doc_ids,  # best-effort, for offline tests
            "expected_answer_keywords": keywords,
            "metadata": {"keywords": keywords},
        })
    return cases


DATASET_CASES: list[dict[str, object]] = _build_cases()


# Backwards-compat: keep DATASET_QUERIES as an alias so existing callers
# (smoke_test.py, old benchmark harness) still work.
DATASET_QUERIES: list[dict[str, object]] = [
    {
        "query": c["query"],
        "top_k": 5,
        "rerank": True,
        "relevant_keywords": c["expected_answer_keywords"],  # type: ignore[typeddict-item]
        "expected_doc_title": c["relevant_document_titles"][0],  # type: ignore[typeddict-item]
    }
    for c in DATASET_CASES
]


__all__ = ["DATASET_CASES", "DATASET_DOCS", "DATASET_QUERIES"]
