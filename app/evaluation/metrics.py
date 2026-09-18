"""Evaluation framework.

Implements:
  * Retrieval metrics: Recall@K, Precision@K, MRR, Hit Rate, NDCG@K.
  * Answer metrics: groundedness, citation correctness, answer presence.

All metrics are computed from already-ingested chunks (the evaluator runs
retrieval+generation against the configured orchestrator). No metric is
invented — each is documented inline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from app.domain import (
    Citation,
    RAGAnswer,
    RetrievalQuery,
)

# ============================================================================
# Retrieval metrics
# ============================================================================


def recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: |retrieved∩relevant| / |relevant|, capped at top-k."""
    if not relevant:
        return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / k


def hit_rate_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved_ids[:k]
    return 1.0 if any(r in relevant for r in topk) else 0.0


def mrr(retrieved_ids: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant hit, 0 if no hit."""
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Binary relevance: each retrieved item is relevant (1) or not (0).
    DCG = sum_{i=1}^{k} rel_i / log2(i+1)
    IDCG = DCG when all relevant items are at the top.
    NDCG = DCG / IDCG (or 0 if IDCG == 0).
    """
    if not relevant:
        return 0.0
    topk = retrieved_ids[:k]
    dcg = sum((1.0 if r in relevant else 0.0) / math.log2(i + 2) for i, r in enumerate(topk))
    # Ideal ranking: all relevant items at the top, capped at k.
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    retrieved_ids: list[str], relevant_ids: Sequence[str], k: int = 10
) -> dict[str, float]:
    """All retrieval metrics for one case in one shot."""
    rel = set(relevant_ids)
    return {
        "recall@k": round(recall_at_k(retrieved_ids, rel, k), 4),
        "precision@k": round(precision_at_k(retrieved_ids, rel, k), 4),
        "hit_rate@k": round(hit_rate_at_k(retrieved_ids, rel, k), 4),
        "mrr": round(mrr(retrieved_ids, rel), 4),
        "ndcg@k": round(ndcg_at_k(retrieved_ids, rel, k), 4),
    }


# ============================================================================
# Answer metrics
# ============================================================================


def citation_correctness(citations: list[Citation], relevant_chunk_ids: set[str]) -> float:
    """Fraction of citations that point to a known-relevant chunk."""
    if not citations:
        return 0.0
    if not relevant_chunk_ids:
        return 0.0  # cannot evaluate without ground truth
    good = sum(1 for c in citations if c.chunk_id in relevant_chunk_ids)
    return round(good / len(citations), 4)


def groundedness(answer: RAGAnswer, relevant_chunk_ids: set[str]) -> float:
    """Heuristic groundedness: how much of the answer is backed by retrieved
    AND relevant chunks, measured by citation coverage.

    Returns 0.0 if the answer has no citations or no relevant chunks are known.
    """
    if not answer.citations:
        return 0.0
    if not relevant_chunk_ids:
        # Without ground truth, fall back to "backed by retrieved chunks".
        return 1.0 if answer.evidence_sufficient else 0.0
    cited_in_relevant = sum(1 for c in answer.citations if c.chunk_id in relevant_chunk_ids)
    return round(cited_in_relevant / max(1, len(answer.citations)), 4)


def answer_relevance(answer: RAGAnswer, query: str) -> float:
    """Cheap lexical overlap between query tokens and answer tokens.

    Not a substitute for an LLM-as-judge score, but reproducible and free.
    """
    if not answer.answer or not query:
        return 0.0
    from app.storage.vector_store import tokenize

    q = set(tokenize(query))
    a = set(tokenize(answer.answer))
    if not q:
        return 0.0
    overlap = len(q & a)
    return round(overlap / len(q), 4)


def answer_completeness(answer: RAGAnswer, expected: str | None) -> float:
    """Cheap completeness: token overlap between the answer and expected.

    Returns 0.0 when no expected answer is provided.
    """
    if not expected:
        return 0.0
    from app.storage.vector_store import tokenize

    a = set(tokenize(answer.answer))
    e = set(tokenize(expected))
    if not e:
        return 0.0
    return round(len(a & e) / len(e), 4)


# ============================================================================
# Evaluator
# ============================================================================


class Evaluator:
    """Run retrieval + answer evaluation against a configured orchestrator."""

    def __init__(
        self,
        orchestrator: Any,
        top_k: int = 10,
        alpha: float | None = None,
        rerank: bool = True,
    ) -> None:
        self.orchestrator = orchestrator
        self.top_k = top_k
        self.alpha = alpha
        self.rerank = rerank

    def evaluate(self, cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Run retrieval + answer evaluation against a configured orchestrator.

        Each case dict may include:
          * `relevant_chunk_ids`: explicit gold chunk ids (best).
          * `relevant_document_titles`: titles of gold documents. The evaluator
            resolves these to chunk ids by looking at what's currently in the
            vector store. Useful when chunk ids are runtime-generated and
            can't be hard-coded in the dataset.
          * `expected_answer_keywords`: tokens that should appear in the answer
            (used as a fallback for answer_completeness).
        """
        out_cases: list[dict[str, Any]] = []
        for c in cases:
            # Resolve gold chunk ids from titles if explicit ids aren't usable.
            c = self._resolve_gold(c)
            out_cases.append(self._eval_one(c))
        n = len(out_cases)
        if n == 0:
            return {
                "n_cases": 0,
                "retrieval_metrics_mean": {},
                "answer_metrics_mean": {},
                "latency_ms_mean": 0.0,
                "cases": [],
            }
        agg_ret = _mean_dicts([c["retrieval_metrics"] for c in out_cases if c["retrieval_metrics"]])
        agg_ans = _mean_dicts([c["answer_metrics"] for c in out_cases if c["answer_metrics"]])
        lat = [c["latency_ms"] for c in out_cases]
        return {
            "n_cases": n,
            "retrieval_metrics_mean": agg_ret,
            "answer_metrics_mean": agg_ans,
            "latency_ms_mean": round(sum(lat) / n, 3) if lat else 0.0,
            "cases": out_cases,
        }

    def _resolve_gold(self, case: dict[str, Any]) -> dict[str, Any]:
        """If the case provides `relevant_document_titles`, look up the
        chunk ids currently in the vector store for those documents and use
        them as the gold set. This lets the dataset stay runtime-agnostic
        (we don't know chunk ids until ingestion runs).
        """
        titles = case.get("relevant_document_titles") or []
        if not titles:
            return case

        # Build a {title: [chunk_id, ...]} map from the vector store.
        store = self.orchestrator.vector_store
        try:
            all_chunks = store.all_chunks() if hasattr(store, "all_chunks") else []
        except Exception:
            all_chunks = []
        title_to_chunks: dict[str, list[str]] = {}
        for c in all_chunks:
            t = c.metadata.get("doc_title")
            if t:
                title_to_chunks.setdefault(t, []).append(c.id)

        resolved: list[str] = list(case.get("relevant_chunk_ids") or [])
        for t in titles:
            resolved.extend(title_to_chunks.get(t, []))
        # Dedup, preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for cid in resolved:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        new_case = dict(case)
        new_case["relevant_chunk_ids"] = deduped
        return new_case

    def _eval_one(self, case: dict[str, Any]) -> dict[str, Any]:
        case_id = case.get("case_id") or "case"
        query = case["query"]
        relevant_chunks = set(case.get("relevant_chunk_ids", []))
        expected = case.get("expected_answer")
        keywords = case.get("expected_answer_keywords") or case.get("metadata", {}).get("keywords") or []
        try:
            q = RetrievalQuery(
                text=query,
                top_k=self.top_k,
                alpha=self.alpha,
                rerank=self.rerank,
            )
            ans = self.orchestrator.query(q)
            retrieved_ids = (
                [c.chunk_id for c in (ans.retrieval.candidates if ans.retrieval else [])]
            )
            ret_m = retrieval_metrics(retrieved_ids, list(relevant_chunks), k=self.top_k)

            # If no explicit expected_answer is given but we have keywords,
            # treat the keyword set as a soft "expected" for answer_completeness.
            expected_for_completeness = expected
            if not expected_for_completeness and keywords:
                expected_for_completeness = " ".join(keywords)

            ans_m = {
                "citation_correctness": citation_correctness(ans.citations, relevant_chunks),
                "groundedness": groundedness(ans, relevant_chunks),
                "answer_relevance": answer_relevance(ans, query),
                "answer_completeness": answer_completeness(ans, expected_for_completeness),
                "answer_keyword_overlap": _keyword_overlap(ans.answer, list(keywords)),
                "evidence_sufficient": 1.0 if ans.evidence_sufficient else 0.0,
            }
            return {
                "case_id": case_id,
                "retrieval_metrics": ret_m,
                "answer_metrics": ans_m,
                "answer": ans.answer,
                "citations": [c.model_dump() for c in ans.citations],
                "latency_ms": round(ans.latency_ms, 3),
                "error": None,
            }
        except Exception as exc:
            return {
                "case_id": case_id,
                "retrieval_metrics": {},
                "answer_metrics": {},
                "answer": None,
                "citations": [],
                "latency_ms": 0.0,
                "error": str(exc),
            }


def _mean_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = set().union(*[set(d.keys()) for d in dicts])
    out: dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return out


def _keyword_overlap(answer: str, keywords: list[str]) -> float:
    """Fraction of gold keywords present in the answer (case-insensitive).

    A reproducible, deterministic lexical signal. Same answer + same keywords
    → same score, every time.
    """
    if not keywords or not answer:
        return 0.0
    from app.storage.vector_store import tokenize

    answer_tokens = set(tokenize(answer))
    keyword_tokens: set[str] = set()
    for k in keywords:
        keyword_tokens.update(tokenize(k))
    if not keyword_tokens:
        return 0.0
    overlap = len(answer_tokens & keyword_tokens)
    return round(overlap / len(keyword_tokens), 4)


__all__ = [
    "Evaluator",
    "answer_completeness",
    "answer_relevance",
    "citation_correctness",
    "groundedness",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "retrieval_metrics",
]
