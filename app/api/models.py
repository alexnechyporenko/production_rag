"""API request/response models.

Pydantic models that travel over the wire. These are intentionally distinct
from the domain models so the API can evolve independently of the internal
contract (versioning, deprecation, response shaping).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Health
# ============================================================================


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    providers: dict[str, str] = Field(default_factory=dict)
    chunk_count: int = 0


# ============================================================================
# Documents
# ============================================================================


class IngestTextRequest(BaseModel):
    text: str
    content_type: str = "text/plain"
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestBytesRequest(BaseModel):
    """Base64-encoded document bytes."""

    data: str = Field(..., description="base64-encoded document bytes")
    content_type: str = "application/octet-stream"
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestFileUpload(BaseModel):
    """Metadata for multipart upload; the file itself comes via UploadFile."""

    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestBatchRequest(BaseModel):
    items: list[dict[str, Any]] = Field(..., min_length=1, max_length=200)


class IngestResponse(BaseModel):
    document_id: str
    chunk_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class IngestBatchResponse(BaseModel):
    results: list[IngestResponse]


class DocumentInfo(BaseModel):
    document_id: str
    chunk_count: int
    title: str | None = None
    content_type: str | None = None


# ============================================================================
# Search / Query
# ============================================================================


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1, le=200)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    methods: list[str] = Field(default_factory=lambda: ["dense", "lexical"])

    @field_validator("methods")
    @classmethod
    def _check_methods(cls, v: list[str]) -> list[str]:
        allowed = {"dense", "lexical"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"methods must be subset of {allowed}; got bad={bad}")
        if not v:
            raise ValueError("methods must not be empty")
        return v


class SearchResponse(BaseModel):
    query_id: str
    query: str
    candidates: list[dict[str, Any]]
    latency_ms: float
    methods_used: list[str]
    counts: dict[str, int]


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1, le=200)
    rerank: bool = True
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    snippet: str | None = None


class QueryResponse(BaseModel):
    request_id: str | None
    query_id: str
    answer: str
    citations: list[CitationResponse]
    confidence: float
    used_chunks: list[str]
    evidence_sufficient: bool
    token_usage: dict[str, Any]
    cost_estimate: dict[str, Any]
    latency_ms: float
    retrieval: dict[str, Any] | None = None
    rerank: dict[str, Any] | None = None
    model: str | None
    warnings: list[str] = Field(default_factory=list)
    trace: dict[str, Any] | None = None


# ============================================================================
# Evaluation
# ============================================================================


class EvaluationCaseRequest(BaseModel):
    case_id: str
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None


class EvaluateRequest(BaseModel):
    cases: list[EvaluationCaseRequest]
    top_k: int = 10
    rerank: bool = True
    alpha: float | None = None


class EvaluateResponse(BaseModel):
    n_cases: int
    retrieval_metrics_mean: dict[str, float]
    answer_metrics_mean: dict[str, float]
    latency_ms_mean: float
    cases: list[dict[str, Any]]


# ============================================================================
# Metrics
# ============================================================================


class MetricsResponse(BaseModel):
    counters: dict[str, float]
    gauges: dict[str, float]
    histograms: dict[str, dict[str, float]]
    caches: dict[str, dict[str, int]]


# ============================================================================
# Error
# ============================================================================


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None
