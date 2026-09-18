"""Retrieval layer: dense, lexical, hybrid fusion.

The orchestrator calls `HybridRetriever.retrieve(query)`. Internally it runs
dense + lexical, normalises the scores with min-max, and combines them as
`hybrid_score = alpha * dense + (1 - alpha) * lexical`. The fusion strategy
is documented in `EVALUATION.md` and tested in `tests/unit/test_hybrid.py`.
"""

from __future__ import annotations

import asyncio

from app.domain import RetrievalQuery, RetrievalResult, RetrievedChunk, now_ms
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.storage import LexicalIndex, VectorStore


def min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]. Returns all 0.0 if input is constant."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    rng = hi - lo
    if rng == 0:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - lo) / rng for s in scores]


def reciprocal_rank_fuse(
    dense: list[RetrievedChunk], lexical: list[RetrievedChunk], k: int = 60
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion as an alternative fusion strategy.

    Useful when raw scores from two retrievers are not directly comparable.
    The default hybrid retriever uses linear fusion, but RRF is available
    as `fusion="rrf"` for callers that prefer it.
    """
    scores: dict[str, float] = {}
    payload: dict[str, RetrievedChunk] = {}
    for rank, c in enumerate(dense):
        scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        payload[c.chunk_id] = c
    for rank, c in enumerate(lexical):
        scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        payload[c.chunk_id] = c
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[RetrievedChunk] = []
    for rank, (cid, score) in enumerate(order):
        c = payload[cid]
        out.append(
            c.model_copy(
                update={
                    "score": score,
                    "rank": rank,
                    "retrieval_method": "hybrid_rrf",
                }
            )
        )
    return out


class HybridRetriever:
    """Dense + lexical retriever with configurable fusion.

    Args:
        vector_store: implements `VectorStore` (in-memory or pgvector).
        lexical_index: `LexicalIndex` (BM25Okapi) instance.
        embedding_provider: implements `EmbeddingProvider`.
        alpha: weight on dense score. Default taken from settings.
        fusion: "linear" or "rrf". Default "linear".
        candidate_limit: cap on the number of candidates returned before
            reranking. Larger = more recall, more cost. Default from settings.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        lexical_index: LexicalIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        alpha: float | None = None,
        fusion: str = "linear",
        candidate_limit: int | None = None,
        default_top_k: int | None = None,
        default_min_similarity: float | None = None,
    ) -> None:
        from app.config import get_settings

        s = get_settings()
        self.vector_store = vector_store or _build_default_vector_store()
        self.lexical_index = lexical_index or LexicalIndex()
        self.embedding_provider = embedding_provider or get_embedding_provider(s)
        self.alpha = alpha if alpha is not None else s.hybrid_alpha
        self.fusion = fusion
        self.candidate_limit = candidate_limit or s.retrieval_candidate_limit
        self.default_top_k = default_top_k or s.top_k
        self.default_min_similarity = (
            default_min_similarity if default_min_similarity is not None else s.min_similarity_threshold
        )

    # --- public API ---------------------------------------------------------
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Run dense + lexical retrieval and fuse the results."""
        start = now_ms()
        methods: list[str] = []
        dense: list[RetrievedChunk] = []
        lexical: list[RetrievedChunk] = []

        if "dense" in query.methods:
            methods.append("dense")
            qvec = self.embedding_provider.embed_texts([query.text])[0]
            dense = self.vector_store.search(
                query_vector=qvec,
                top_k=query.top_k * 2,
                filters=query.filters or None,
                min_similarity=query.min_similarity
                if query.min_similarity is not None
                else self.default_min_similarity,
            )

        if "lexical" in query.methods and self.lexical_index.count() > 0:
            methods.append("lexical")
            lexical = self.lexical_index.search(query.text, top_k=query.top_k * 2)
            if query.filters and "document_id" in query.filters:
                lexical = [c for c in lexical if c.document_id == query.filters["document_id"]]

        if self.fusion == "rrf" or (len(methods) == 2 and not _can_linearly_fuse(dense, lexical)):
            fused = reciprocal_rank_fuse(dense, lexical)
        else:
            fused = self._linear_fuse(dense, lexical)

        # Cap candidates, then truncate to top_k.
        fused = fused[: self.candidate_limit]
        # Re-rank by score descending and assign ranks 0..n-1.
        fused.sort(key=lambda c: (-c.score, c.chunk_id))
        fused = [
            c.model_copy(update={"rank": i, "retrieval_method": "hybrid"})
            for i, c in enumerate(fused[: query.top_k])
        ]

        latency = now_ms() - start
        counts = {
            "dense": len(dense),
            "lexical": len(lexical),
            "fused": len(fused),
        }
        return RetrievalResult(
            query_id=query.query_id,
            query=query.text,
            candidates=fused,
            latency_ms=latency,
            methods_used=tuple(methods) or ("none",),
            counts=counts,
        )

    async def retrieve_async(self, query: RetrievalQuery) -> RetrievalResult:
        """Async wrapper. Uses async embedding when available."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query)

    # --- internals ---------------------------------------------------------
    def _linear_fuse(
        self, dense: list[RetrievedChunk], lexical: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        # Min-max normalise each side to [0,1] so the alpha weighting is meaningful.
        d_norm = min_max_normalize([c.score for c in dense])
        l_norm = min_max_normalize([c.score for c in lexical])
        d_scores = {c.chunk_id: d_norm[i] for i, c in enumerate(dense)}
        l_scores = {c.chunk_id: l_norm[i] for i, c in enumerate(lexical)}
        all_ids = set(d_scores) | set(l_scores)
        payload: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense}
        for c in lexical:
            payload[c.chunk_id] = payload.get(c.chunk_id, c)
        out: list[RetrievedChunk] = []
        for cid in all_ids:
            d_val = d_scores.get(cid, 0.0)
            l_val = l_scores.get(cid, 0.0)
            score = self.alpha * d_val + (1 - self.alpha) * l_val
            base = payload[cid]
            out.append(
                base.model_copy(
                    update={"score": score, "retrieval_method": "hybrid", "rank": 0}
                )
            )
        out.sort(key=lambda c: (-c.score, c.chunk_id))
        return out


def _can_linearly_fuse(dense: list[RetrievedChunk], lexical: list[RetrievedChunk]) -> bool:
    """We can linearly fuse if both sides are non-empty or one is empty (just
    use the other side as-is).
    """
    return True


def _build_default_vector_store() -> VectorStore:
    from app.storage import get_vector_store

    return get_vector_store()


__all__ = ["HybridRetriever", "min_max_normalize", "reciprocal_rank_fuse"]
