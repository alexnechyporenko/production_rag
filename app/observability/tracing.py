"""Request tracing: request_id propagation + per-request metrics context."""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from app.domain.models import now_ms

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    rid = f"req-{uuid.uuid4().hex[:24]}"
    _request_id_var.set(rid)
    return rid


def current_request_id() -> str:
    rid = _request_id_var.get()
    if not rid:
        rid = new_request_id()
    return rid


@dataclass
class RequestTrace:
    """Accumulator for one RAG request, used by the orchestrator and exported
    via the `/metrics` endpoint and per-response diagnostic fields.
    """

    request_id: str = field(default_factory=new_request_id)
    query: str = ""
    started_at_ms: float = field(default_factory=now_ms)

    retrieval_count: int = 0
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    context_size_chars: int = 0
    context_chunks: int = 0
    llm_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    total_latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        self.total_latency_ms = now_ms() - self.started_at_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "retrieval_count": self.retrieval_count,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 3),
            "reranking_latency_ms": round(self.reranking_latency_ms, 3),
            "context_size": self.context_size_chars,
            "context_chunks": self.context_chunks,
            "llm_latency_ms": round(self.llm_latency_ms, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": round(self.estimated_cost, 8),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "warnings": list(self.warnings),
        }


__all__ = ["RequestTrace", "current_request_id", "new_request_id"]
