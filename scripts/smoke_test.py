"""Manual end-to-end smoke test of the RAG pipeline.

Runs offline using mock providers. Useful as a development check.
"""

from __future__ import annotations

import asyncio
import sys

# Bootstrap path
sys.path.insert(0, ".")

from app.chunking import get_chunker
from app.context import ContextBuilder
from app.domain import RetrievalQuery
from app.embeddings import MockEmbeddingProvider
from app.generation import GenerationService
from app.ingestion import IngestionService
from app.reranking import MockReranker
from app.retrieval import HybridRetriever
from app.storage import InMemoryVectorStore, LexicalIndex

SAMPLE_DOCS = [
    (
        "PostgreSQL is a powerful open-source relational database. "
        "It supports advanced features such as JSONB, full-text search, and the pgvector extension "
        "for vector storage. pgvector enables efficient similarity search using cosine distance."
    ),
    (
        "FastAPI is a modern Python web framework for building APIs. "
        "It uses Pydantic for validation, Starlette for ASGI handling, and supports async endpoints. "
        "FastAPI auto-generates OpenAPI documentation and provides strong type safety."
    ),
    (
        "Hybrid search combines dense vector retrieval with lexical retrieval. "
        "Dense retrieval captures semantic similarity, while lexical retrieval (e.g. BM25) captures "
        "exact term matches. A weighted alpha fusion is a common combination strategy."
    ),
    (
        "Reranking is an optional second stage of retrieval. Given a set of candidate chunks, "
        "a cross-encoder model scores each (query, chunk) pair. Reranking can improve precision "
        "but adds latency. The system must function without a reranker."
    ),
]


async def main() -> None:
    emb = MockEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    lexical = LexicalIndex()

    # Ingest
    ingest = IngestionService(chunker=get_chunker("recursive", chunk_size=600, overlap=100))
    all_chunks = []
    for text in SAMPLE_DOCS:
        r = ingest.ingest_text(text)
        all_chunks.extend(r.chunks)
    print(f"Ingested {len(SAMPLE_DOCS)} documents → {len(all_chunks)} chunks")

    # Embed + index
    vectors = emb.embed_texts([c.content for c in all_chunks])
    store.add(all_chunks, vectors, model=emb.model)
    lexical.add(all_chunks)
    print(f"Vector store has {store.count()} chunks; lexical index has {lexical.count()}")

    # Retrieve
    retriever = HybridRetriever(
        vector_store=store,
        lexical_index=lexical,
        embedding_provider=emb,
        alpha=0.6,
    )
    query = RetrievalQuery(text="How does hybrid retrieval combine dense and lexical search?")
    rr = retriever.retrieve(query)
    print(f"\nRetrieval ({rr.latency_ms:.1f} ms) — counts={rr.counts}")
    for c in rr.candidates[:5]:
        print(f"  rank={c.rank} score={c.score:.3f} method={c.retrieval_method} id={c.chunk_id[:16]}…")
        print(f"    {c.content[:90]}…")

    # Rerank
    reranker = MockReranker()
    rer = reranker.rerank(query.text, rr.candidates, top_k=5)
    print(f"\nRerank ({rer.latency_ms:.1f} ms)")
    for c in rer.ranked[:3]:
        print(f"  rank={c.rank} score={c.score:.3f} id={c.chunk_id[:16]}…")

    # Context
    builder = ContextBuilder(max_chunks=5, max_chars=4000, max_tokens=1000)
    ctx = builder.build(rer.ranked)
    print(f"\nContext: {len(ctx.items)} chunks, {ctx.total_chars} chars, truncated={ctx.truncated}")

    # Generate
    gen = GenerationService()
    answer = await gen.generate_async(
        question=query.text,
        context=ctx,
        query_id=query.query_id,
    )
    print(f"\nAnswer ({answer.latency_ms:.1f} ms, {answer.token_usage.total_tokens} tokens)")
    print(answer.answer)
    print(f"\nCitations: {len(answer.citations)}; sufficient={answer.evidence_sufficient}")
    print(f"Cost: {answer.cost_estimate}")


if __name__ == "__main__":
    asyncio.run(main())
