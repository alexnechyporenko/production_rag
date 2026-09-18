"""Reranker: optional second-stage refinement of retrieved candidates.

A reranker receives (query, candidates) and returns a re-scored, re-ordered
list. It is OPTIONAL — the rest of the system MUST work without one (spec §11).

Two implementations:

  - MockReranker: deterministic, offline. Boosts candidates whose content
    overlaps the query terms and penalises candidates that already ranked
    poorly. Useful for unit tests and for running offline benchmarks.
  - CrossEncoderReranker: thin client for an OpenAI-compatible reranker
    endpoint. Lazy-loaded; safe to omit.

Implementations MUST preserve source identifiers (chunk_id, document_id).
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from app.domain import RerankResult, RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Reranker protocol."""

    @property
    def name(self) -> str: ...

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> RerankResult: ...


# ============================================================================
# Mock reranker
# ============================================================================


class MockReranker:
    """Deterministic offline reranker.

    Score = overlap(query_tokens, candidate_tokens) + 0.3 * (1 - rank_norm)
    where rank_norm = rank / N.

    Same query + candidates always yields the same ordering (tie-broken by chunk_id).
    """

    def __init__(self, name: str = "mock-reranker-v1") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> RerankResult:
        from app.observability import now_ms as _now_ms
        from app.storage.vector_store import tokenize

        start = _now_ms()
        if not candidates:
            return RerankResult(query_id="", ranked=[], latency_ms=_now_ms() - start, model=self._name)
        q_tokens = set(tokenize(query))
        n = len(candidates)
        scored: list[tuple[float, RetrievedChunk]] = []
        for c in candidates:
            c_tokens = set(tokenize(c.content))
            overlap = len(q_tokens & c_tokens)
            rank_norm = c.rank / max(n, 1)
            score = overlap + 0.3 * (1 - rank_norm)
            scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        k = top_k if top_k is not None else len(scored)
        out: list[RetrievedChunk] = []
        scores: list[float] = []
        for rank, (score, c) in enumerate(scored[:k]):
            out.append(
                c.model_copy(
                    update={
                        "score": score,
                        "rank": rank,
                        "retrieval_method": "reranked",
                    }
                )
            )
            scores.append(score)
        latency = _now_ms() - start
        # query_id is propagated from candidates' retrieval context if available.
        return RerankResult(
            query_id="",
            ranked=out,
            latency_ms=latency,
            model=self._name,
            scores=scores,
        )


# ============================================================================
# Cross-encoder reranker (OpenAI-compatible)
# ============================================================================


class CrossEncoderReranker:
    """Reranker that calls an external cross-encoder endpoint.

    The expected contract is an OpenAI-style POST returning scores in the
    same order as the inputs. Implementation is intentionally minimal.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        model: str = "cross-encoder-v1",
        timeout: float = 30.0,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key or ""
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> RerankResult:
        return asyncio.run(self.rerank_async(query, candidates, top_k=top_k))

    async def rerank_async(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> RerankResult:
        import httpx

        from app.observability import now_ms as _now_ms

        start = _now_ms()
        if not candidates:
            return RerankResult(query_id="", ranked=[], latency_ms=_now_ms() - start, model=self._model)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "query": query,
            "documents": [c.content for c in candidates],
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                scores = resp.json().get("scores", [])
        except httpx.HTTPError as exc:  # pragma: no cover
            raise RerankingError(f"Cross-encoder call failed: {exc}") from exc
        if len(scores) != len(candidates):
            raise RerankingError(
                f"Cross-encoder returned {len(scores)} scores for {len(candidates)} candidates"
            )
        scored = sorted(zip(scores, candidates, strict=True), key=lambda x: (-x[0], x[1].chunk_id))
        k = top_k if top_k is not None else len(scored)
        out = [
            c.model_copy(update={"score": float(s), "rank": i, "retrieval_method": "reranked"})
            for i, (s, c) in enumerate(scored[:k])
        ]
        return RerankResult(
            query_id="",
            ranked=out,
            latency_ms=_now_ms() - start,
            model=self._model,
            scores=[float(s) for s, _ in scored[:k]],
        )


# Lazy: avoid import error if not used.
from app.domain import RerankingError  # noqa: E402

# ============================================================================
# Factory
# ============================================================================


def get_reranker(provider: str | None = None, **kwargs: Any) -> Reranker | None:
    """Return a reranker instance, or None when reranking is disabled.

    `provider == "none"` returns None. `provider == "mock"` returns MockReranker.
    Any other provider name attempts to load a cross-encoder-style reranker.
    """
    from app.config import get_settings

    s = get_settings()
    p = provider or s.reranker_provider
    if p == "none":
        return None
    if p == "mock":
        return MockReranker(name=s.reranker_model or "mock-reranker-v1")
    if p == "cross-encoder":
        return CrossEncoderReranker(
            endpoint=kwargs.get("endpoint", "http://localhost:8001/rerank"),
            api_key=kwargs.get("api_key", ""),
            model=s.reranker_model or "cross-encoder-v1",
        )
    raise RerankingError(f"Unknown reranker provider: {p}")


__all__ = ["CrossEncoderReranker", "MockReranker", "Reranker", "get_reranker"]
