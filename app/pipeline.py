"""RAG pipeline orchestrator.

Wires the components together into the canonical pipeline (spec §8):

    Query Validation → Query Processing → Dense Retrieval → Lexical Retrieval
    → Hybrid Fusion → Reranking → Context Filtering → Context Budgeting
    → Prompt Construction → LLM Generation → Citation Validation → Answer → Metrics

The orchestrator is sync-by-default but exposes `query_async` for use from
FastAPI's async endpoints. It is intentionally a thin coordinator — business
logic lives in the component modules so each remains independently testable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.caching import get_cache_registry
from app.config import Settings, get_settings
from app.context import ContextBuilder
from app.domain import (
    CostEstimate,
    RAGAnswer,
    RerankResult,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
    ValidationError,
    now_ms,
)
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.generation import GenerationService
from app.observability import RequestTrace, get_logger, get_registry
from app.reranking import Reranker, get_reranker
from app.retrieval import HybridRetriever
from app.storage import InMemoryVectorStore, LexicalIndex, VectorStore, get_vector_store

_logger = get_logger("app.rag.orchestrator")


@dataclass
class OrchestratorComponents:
    """Pluggable components for the orchestrator.

    Defaults are wired up from settings, so a vanilla `RAGOrchestrator()` is
    ready to serve queries offline (mock LLM, mock embedder, in-memory store).
    """

    vector_store: VectorStore = field(default_factory=lambda: get_vector_store())
    lexical_index: LexicalIndex = field(default_factory=LexicalIndex)
    embedding_provider: EmbeddingProvider = field(default_factory=get_embedding_provider)
    retriever: HybridRetriever | None = None
    reranker: Reranker | None = None
    context_builder: ContextBuilder | None = None
    generation: GenerationService | None = None


class RAGOrchestrator:
    """Coordinates the RAG pipeline."""

    def __init__(
        self,
        components: OrchestratorComponents | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        c = components or OrchestratorComponents()

        # Wire defaults if missing.
        self.vector_store = c.vector_store
        self.lexical_index = c.lexical_index
        self.embedding_provider = c.embedding_provider
        self.retriever = c.retriever or HybridRetriever(
            vector_store=self.vector_store,
            lexical_index=self.lexical_index,
            embedding_provider=self.embedding_provider,
        )
        self.reranker = c.reranker if c.reranker is not None else get_reranker()
        self.context_builder = c.context_builder or ContextBuilder(
            max_chunks=self.settings.max_context_chunks,
            max_chars=self.settings.max_chunk_chars * self.settings.max_context_chunks,
            max_tokens=self.settings.max_context_tokens,
        )
        self.generation = c.generation or GenerationService(settings=self.settings)
        self._cache = get_cache_registry()

    # --- Ingestion --------------------------------------------------------
    def ingest_documents(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ingest + embed + index a batch of documents.

        Each item dict must match the `IngestionService.ingest_*` signatures,
        with a `kind` key in {"text", "bytes", "file"}.

        Returns a list of {document_id, chunk_ids, warnings, error} dicts.
        Failure isolation: a single bad document does not abort the batch.
        """
        from app.ingestion import IngestionService

        results: list[dict[str, Any]] = []
        svc = IngestionService()
        for item in items:
            kind = item.get("kind", "text")
            try:
                if kind == "text":
                    r = svc.ingest_text(
                        item["text"],
                        content_type=item.get("content_type", "text/plain"),
                        title=item.get("title"),
                        **{k: v for k, v in item.items() if k not in {"kind", "text", "content_type", "title"}},
                    )
                elif kind == "bytes":
                    r = svc.ingest_bytes(
                        item["data"],
                        content_type=item.get("content_type", "application/octet-stream"),
                        title=item.get("title"),
                        **{k: v for k, v in item.items() if k not in {"kind", "data", "content_type", "title"}},
                    )
                elif kind == "file":
                    r = svc.ingest_file(item["path"], **{k: v for k, v in item.items() if k not in {"kind", "path"}})
                else:
                    raise ValidationError(f"Unknown ingest kind: {kind!r}")
            except Exception as exc:
                results.append(
                    {"document_id": None, "chunk_ids": [], "warnings": [], "error": str(exc)}
                )
                continue

            # Embed + index.
            try:
                chunks = r.chunks
                vectors = self.embedding_provider.embed_texts([c.content for c in chunks])
                self.vector_store.add(chunks, vectors, model=self.embedding_provider.model)
                self.lexical_index.add(chunks)
                results.append(
                    {
                        "document_id": r.document_id,
                        "chunk_ids": r.chunk_ids,
                        "warnings": r.warnings,
                        "error": None,
                    }
                )
                get_registry().inc("ingest.documents")
                get_registry().inc("ingest.chunks", len(chunks))
            except Exception as exc:
                results.append(
                    {
                        "document_id": r.document_id,
                        "chunk_ids": [],
                        "warnings": r.warnings,
                        "error": str(exc),
                    }
                )
        return results

    # --- Retrieval --------------------------------------------------------
    def search(self, query: RetrievalQuery) -> RetrievalResult:
        """Hybrid retrieval only (no generation)."""
        self._validate_query(query)
        if not isinstance(self.vector_store, InMemoryVectorStore):
            # Don't cache for backends that may have changed out-of-band.
            return self.retriever.retrieve(query)
        return self._cache.cached_call(
            "retrieval",
            ("search", query.text, query.top_k, query.alpha, query.methods, query.filters),
            lambda: self.retriever.retrieve(query),
        )

    def _rerank(
        self, query: RetrievalQuery, candidates: list[RetrievedChunk]
    ) -> tuple[RerankResult | None, list[RetrievedChunk]]:
        if not query.rerank or self.reranker is None or not candidates:
            return None, candidates
        result = self.reranker.rerank(query.text, candidates, top_k=self.settings.rerank_top_k)
        # Propagate query_id for tracing.
        result.query_id = query.query_id
        return result, result.ranked

    # --- Query ------------------------------------------------------------
    def query(self, query: RetrievalQuery) -> RAGAnswer:
        """End-to-end query: retrieve → rerank → build context → generate."""
        return asyncio.run(self.query_async(query))

    async def query_async(self, query: RetrievalQuery) -> RAGAnswer:
        self._validate_query(query)
        trace = RequestTrace(query=query.text)

        # 1. Retrieval
        retrieval = self.search(query)
        trace.retrieval_count = len(retrieval.candidates)
        trace.retrieval_latency_ms = retrieval.latency_ms
        get_registry().observe("rag.retrieval_latency_ms", retrieval.latency_ms)

        if not retrieval.candidates:
            # No evidence — return an empty answer instead of hallucinating.
            trace.finish()
            return self._no_evidence_answer(query, trace)

        # 2. Reranking
        rerank_result, ranked = self._rerank(query, retrieval.candidates)
        if rerank_result is not None:
            trace.reranking_latency_ms = rerank_result.latency_ms
            get_registry().observe("rag.rerank_latency_ms", rerank_result.latency_ms)

        # 3. Context assembly
        context = self.context_builder.build(ranked)
        trace.context_size_chars = context.total_chars
        trace.context_chunks = len(context.items)

        # 4. Generation
        gen_start = now_ms()
        answer = await self.generation.generate_async(
            question=query.text, context=context, query_id=query.query_id
        )
        trace.llm_latency_ms = now_ms() - gen_start
        trace.input_tokens = answer.token_usage.input_tokens
        trace.output_tokens = answer.token_usage.output_tokens
        if answer.cost_estimate.pricing_known:
            trace.estimated_cost = answer.cost_estimate.total_cost

        # 5. Stitch retrieval + rerank into the answer for transparency
        answer.retrieval = retrieval
        answer.rerank = rerank_result
        answer.request_id = trace.request_id
        answer.warnings.extend(trace.warnings)

        trace.finish()
        get_registry().observe("rag.total_latency_ms", trace.total_latency_ms)
        get_registry().inc("rag.queries")
        if not answer.evidence_sufficient:
            get_registry().inc("rag.insufficient_evidence")
        return answer

    # --- Helpers ----------------------------------------------------------
    def _validate_query(self, query: RetrievalQuery) -> None:
        if not query.text or not query.text.strip():
            raise ValidationError("Query text must not be empty")
        if len(query.text) > self.settings.max_query_chars:
            raise ValidationError(
                f"Query too long: {len(query.text)} > {self.settings.max_query_chars}",
                details={"len": len(query.text), "limit": self.settings.max_query_chars},
            )

    def _no_evidence_answer(self, query: RetrievalQuery, trace: RequestTrace) -> RAGAnswer:
        from app.domain import TokenUsage

        trace.finish()
        return RAGAnswer(
            query_id=query.query_id,
            answer=(
                "The provided evidence is insufficient to answer this question."
            ),
            citations=[],
            confidence=0.0,
            used_chunks=[],
            token_usage=TokenUsage.from_pair(input_tokens=0, output_tokens=0, model="mock-llm-v1"),
            cost_estimate=CostEstimate.unknown(),
            latency_ms=trace.total_latency_ms,
            context=None,
            retrieval=None,
            rerank=None,
            request_id=trace.request_id,
            evidence_sufficient=False,
            model=None,
            warnings=["no_evidence: retrieval returned 0 candidates"],
        )


__all__ = ["OrchestratorComponents", "RAGOrchestrator"]
