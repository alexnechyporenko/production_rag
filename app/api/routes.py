"""FastAPI routes for the Production RAG API.

Every endpoint returns JSON. Domain errors are mapped to HTTP responses with
a stable `ErrorResponse` body via the `app.api.errors` exception handlers.
"""

from __future__ import annotations

import base64
import contextlib
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.models import (
    DocumentInfo,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    IngestBatchRequest,
    IngestBatchResponse,
    IngestBytesRequest,
    IngestResponse,
    IngestTextRequest,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.config import get_settings
from app.domain import NotFoundError, RetrievalQuery
from app.observability import get_logger, get_registry

_logger = get_logger("app.api")


def get_orchestrator() -> Any:
    """Return the application-scoped orchestrator singleton."""
    from app.main import get_app_orchestrator

    return get_app_orchestrator()


router = APIRouter()


# ============================================================================
# /health
# ============================================================================


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    orch = get_orchestrator()
    s = get_settings()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        providers={
            "llm": s.llm_provider,
            "embedding": s.embedding_provider,
            "reranker": s.reranker_provider,
        },
        chunk_count=orch.vector_store.count(),
    )


# ============================================================================
# /documents
# ============================================================================


@router.post("/documents", response_model=IngestResponse, status_code=201)
def ingest_text(req: IngestTextRequest) -> IngestResponse:
    orch = get_orchestrator()
    items = [{"kind": "text", **req.model_dump(exclude_none=True)}]
    res = orch.ingest_documents(items)[0]
    if res["error"]:
        raise HTTPException(status_code=422, detail=res["error"])
    return IngestResponse(**res)


@router.post("/documents/batch", response_model=IngestBatchResponse, status_code=201)
def ingest_batch(req: IngestBatchRequest) -> IngestBatchResponse:
    orch = get_orchestrator()
    results = orch.ingest_documents(req.items)
    return IngestBatchResponse(results=[IngestResponse(**r) for r in results])


@router.post("/documents/upload", response_model=IngestResponse, status_code=201)
async def ingest_upload(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> IngestResponse:
    """Multipart upload. Content-type is inferred from the file extension."""
    import os


    orch = get_orchestrator()
    s = get_settings()
    raw = await file.read()
    if len(raw) > s.max_document_bytes:
        raise HTTPException(status_code=413, detail="document_too_large")
    # Save to a temp file under our own sandbox so loader path-safety is happy.
    tmp = f"/tmp/rag-upload-{uuid.uuid4().hex}{os.path.splitext(file.filename or '')[1]}"
    with open(tmp, "wb") as f:
        f.write(raw)
    items = [{"kind": "file", "path": tmp, "title": title}]
    try:
        res = orch.ingest_documents(items)[0]
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)
    if res["error"]:
        raise HTTPException(status_code=422, detail=res["error"])
    return IngestResponse(**res)


@router.post("/documents/bytes", response_model=IngestResponse, status_code=201)
def ingest_bytes(req: IngestBytesRequest) -> IngestResponse:
    """Base64-encoded body upload — useful when multipart is unavailable."""
    orch = get_orchestrator()
    try:
        raw = base64.b64decode(req.data, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid base64: {exc}") from exc
    items = [
        {
            "kind": "bytes",
            "data": raw,
            "content_type": req.content_type,
            "title": req.title,
            **req.metadata,
        }
    ]
    res = orch.ingest_documents(items)[0]
    if res["error"]:
        raise HTTPException(status_code=422, detail=res["error"])
    return IngestResponse(**res)


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str) -> DocumentInfo:
    orch = get_orchestrator()
    docs = orch.vector_store.list_documents()
    for d in docs:
        if d["document_id"] == document_id:
            return DocumentInfo(**d)
    raise NotFoundError(f"Document not found: {document_id}")


# ============================================================================
# /search
# ============================================================================


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    orch = get_orchestrator()
    s = get_settings()
    q = RetrievalQuery(
        text=req.query,
        top_k=req.top_k or s.top_k,
        alpha=req.alpha,
        min_similarity=req.min_similarity,
        filters=req.filters,
        methods=tuple(req.methods),
        rerank=False,
    )
    result = orch.search(q)
    return SearchResponse(
        query_id=result.query_id,
        query=result.query,
        candidates=[c.model_dump() for c in result.candidates],
        latency_ms=result.latency_ms,
        methods_used=list(result.methods_used),
        counts=result.counts,
    )


# ============================================================================
# /query
# ============================================================================


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    from app.observability import current_request_id

    orch = get_orchestrator()
    s = get_settings()
    q = RetrievalQuery(
        text=req.query,
        top_k=req.top_k or s.top_k,
        alpha=req.alpha,
        filters=req.filters,
        rerank=req.rerank,
    )
    ans = orch.query(q)
    return QueryResponse(
        request_id=current_request_id(),
        query_id=ans.query_id,
        answer=ans.answer,
        citations=[c.model_dump() for c in ans.citations],
        confidence=ans.confidence,
        used_chunks=ans.used_chunks,
        evidence_sufficient=ans.evidence_sufficient,
        token_usage=ans.token_usage.model_dump(),
        cost_estimate=ans.cost_estimate.model_dump(),
        latency_ms=ans.latency_ms,
        retrieval=ans.retrieval.model_dump() if ans.retrieval else None,
        rerank=ans.rerank.model_dump() if ans.rerank else None,
        model=ans.model,
        warnings=ans.warnings,
    )


# ============================================================================
# /evaluate
# ============================================================================


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    from app.evaluation import Evaluator

    orch = get_orchestrator()
    ev = Evaluator(orchestrator=orch, top_k=req.top_k, alpha=req.alpha, rerank=req.rerank)
    result = ev.evaluate([c.model_dump() for c in req.cases])
    return EvaluateResponse(
        n_cases=result["n_cases"],
        retrieval_metrics_mean=result["retrieval_metrics_mean"],
        answer_metrics_mean=result["answer_metrics_mean"],
        latency_ms_mean=result["latency_ms_mean"],
        cases=result["cases"],
    )


# ============================================================================
# /metrics
# ============================================================================


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    from typing import cast

    snap = get_registry().snapshot()
    from app.caching import get_cache_registry

    caches = get_cache_registry().snapshot()
    counters = cast(dict[str, float], snap.get("counters", {}))
    gauges = cast(dict[str, float], snap.get("gauges", {}))
    histograms = cast(dict[str, dict[str, float]], snap.get("histograms", {}))
    return MetricsResponse(
        counters=counters,
        gauges=gauges,
        histograms=histograms,
        caches=caches,
    )


__all__ = ["router"]
