"""Typed domain models for the RAG system.

These Pydantic models are the contract between every component
(ingestion → storage → retrieval → context → generation → evaluation).
Nothing crosses a service boundary as a bare dict; everything is a typed model.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    """Generate a deterministic-friendly prefixed UUID4 string."""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# ============================================================================
# Documents & chunks
# ============================================================================


class DocumentSource(StrEnum):
    """Where a document came from (used for routing in the loader)."""

    text = "text"
    file = "file"
    url = "url"
    api = "api"


class DocumentMetadata(BaseModel):
    """Free-form but typed metadata bag for a document."""

    model_config = ConfigDict(extra="allow")

    source: DocumentSource = DocumentSource.api
    content_type: str = Field(default="text/plain", description="MIME type of source")
    language: str | None = None
    author: str | None = None
    title: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Normalized document. Content is plain UTF-8 text after parsing."""

    id: str = Field(default_factory=lambda: _new_id("doc"))
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    @field_validator("content")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Document content must not be empty")
        return v


class DocumentChunk(BaseModel):
    """A chunk of a document with a stable, deterministic identifier."""

    id: str
    document_id: str
    content: str
    sequence: int = Field(ge=0, description="0-indexed position within the document")
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content must not be empty")
        return v


# ============================================================================
# Embeddings
# ============================================================================


class Embedding(BaseModel):
    """Vector representation of a chunk or query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: list[float]
    model: str
    dimension: int

    @field_validator("dimension")
    @classmethod
    def _check_dim(cls, v: int, info) -> int:
        vec = info.data.get("vector")
        if vec is not None and v != len(vec):
            raise ValueError(f"dimension {v} != len(vector) {len(vec)}")
        return v


# ============================================================================
# Retrieval
# ============================================================================


class RetrievalQuery(BaseModel):
    """A fully-formed retrieval request after validation & processing."""

    query_id: str = Field(default_factory=lambda: _new_id("q"))
    text: str
    top_k: int = Field(default=10, ge=1, le=200)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    methods: tuple[str, ...] = Field(default=("dense", "lexical"))
    rerank: bool = True
    user_id: str | None = None


class RetrievedChunk(BaseModel):
    """A single retrieved candidate. Carries enough metadata to explain itself."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    retrieval_method: str
    rank: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None  # optional, not normally returned


class RetrievalResult(BaseModel):
    """The outcome of one retrieval operation."""

    query_id: str
    query: str
    candidates: list[RetrievedChunk]
    latency_ms: float
    methods_used: tuple[str, ...]
    counts: dict[str, int] = Field(default_factory=dict)


class RerankResult(BaseModel):
    """The outcome of a reranking operation over a candidate set."""

    query_id: str
    ranked: list[RetrievedChunk]
    latency_ms: float
    model: str | None = None
    scores: list[float] = Field(default_factory=list)


# ============================================================================
# Context & generation
# ============================================================================


class ContextItem(BaseModel):
    """One piece of evidence included in the LLM context."""

    chunk_id: str
    document_id: str
    content: str
    rank: int = Field(ge=0)
    score: float
    retrieval_method: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGContext(BaseModel):
    """Assembled, bounded context ready for the prompt."""

    items: list[ContextItem]
    total_chars: int
    estimated_tokens: int
    deduped_count: int
    truncated: bool = False


class Citation(BaseModel):
    """A reference from the answer back to a retrieved chunk."""

    chunk_id: str
    document_id: str
    char_start: int | None = None
    char_end: int | None = None
    snippet: str | None = None


class TokenUsage(BaseModel):
    """Token accounting for one LLM call."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    model: str

    @classmethod
    def from_pair(cls, *, input_tokens: int, output_tokens: int, model: str) -> TokenUsage:
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model,
        )


class CostEstimate(BaseModel):
    """Estimated USD cost for one LLM call."""

    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    pricing_known: bool = Field(
        default=True,
        description="False when pricing for the model is not configured.",
    )

    @classmethod
    def unknown(cls, currency: str = "USD") -> CostEstimate:
        return cls(
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            currency=currency,
            pricing_known=False,
        )


class RAGAnswer(BaseModel):
    """Final structured answer contract."""

    query_id: str
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    used_chunks: list[str] = Field(default_factory=list)
    token_usage: TokenUsage
    cost_estimate: CostEstimate
    latency_ms: float
    context: RAGContext | None = None
    retrieval: RetrievalResult | None = None
    rerank: RerankResult | None = None
    request_id: str | None = None
    evidence_sufficient: bool = True
    model: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ============================================================================
# Evaluation
# ============================================================================


class EvaluationCase(BaseModel):
    """One evaluation example. `relevant_chunk_ids` is the gold set."""

    case_id: str
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating one case."""

    case_id: str
    retrieval_metrics: dict[str, float]
    answer_metrics: dict[str, float]
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float
    error: str | None = None


class BenchmarkResult(BaseModel):
    """Aggregate benchmark metrics across many cases."""

    name: str
    n_cases: int
    retrieval_metrics_mean: dict[str, float]
    answer_metrics_mean: dict[str, float]
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    token_usage_total: TokenUsage | None = None
    estimated_cost_total: CostEstimate | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    config: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Helpers
# ============================================================================


def deterministic_chunk_id(document_id: str, sequence: int, content: str) -> str:
    """Stable chunk id derived from (document, sequence, content hash).

    Re-ingesting the same document yields the same chunk ids, which is
    important for cache stability, evaluation reproducibility and idempotent
    re-indexing.
    """
    h = hashlib.sha1(
        f"{document_id}::{sequence}::{content[:128]}".encode(), usedforsecurity=False
    ).hexdigest()
    return f"chunk-{h[:24]}"


def now_ms() -> float:
    """Monotonic-ish millisecond timestamp for latency reporting."""
    return time.perf_counter() * 1000.0


__all__ = [
    "BenchmarkResult",
    "Citation",
    "ContextItem",
    "CostEstimate",
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentSource",
    "Embedding",
    "EvaluationCase",
    "EvaluationResult",
    "RAGAnswer",
    "RAGContext",
    "RAGQuery",
    "RerankResult",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedChunk",
    "TokenUsage",
    "deterministic_chunk_id",
    "now_ms",
]


# Backwards/forward alias: callers sometimes refer to a "RAGQuery".
RAGQuery = RetrievalQuery
