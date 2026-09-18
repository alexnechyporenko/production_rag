"""Storage layer: vector store + lexical index.

Two backends are provided:

  - `InMemoryVectorStore`: pure-Python, dependency-free. Default for tests and
    for environments without PostgreSQL. Persistence is process-local.
  - `PgVectorStore`: PostgreSQL + pgvector. Used in production. Falls back
    gracefully when pgvector is not installed in the database.

The `LexicalIndex` is a thin wrapper around `rank_bm25.BM25Okapi`. Both backends
share the same `VectorStore` protocol, so the retrieval layer is fully
backend-agnostic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.domain import DocumentChunk, Embedding, RetrievedChunk

# ============================================================================
# VectorStore protocol
# ============================================================================


@runtime_checkable
class VectorStore(Protocol):
    """Backend-agnostic vector store.

    Implementations MUST be safe to use from a single asyncio event loop
    (synchronous implementations are wrapped in `run_in_executor` by callers
    when needed).
    """

    @property
    def dimension(self) -> int: ...

    def add(self, chunks: Sequence[DocumentChunk], vectors: Sequence[list[float]], model: str) -> int:
        """Upsert chunks + vectors. Returns the number of rows actually stored."""

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks to `query_vector`."""

    def get(self, chunk_id: str) -> DocumentChunk | None: ...

    def list_documents(self) -> list[dict[str, Any]]: ...

    def count(self) -> int: ...

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns rows deleted."""

    def reset(self) -> None:
        """Wipe the store (used by tests)."""


# ============================================================================
# Lexical index (BM25)
# ============================================================================


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class LexicalIndex:
    """BM25Okapi wrapper with deterministic behaviour.

    The index stores the raw chunk content and tokenisation so reranking and
    retrieval scoring can be reproduced exactly.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        # Imported lazily so tests that don't touch BM25 stay fast.
        from rank_bm25 import BM25Okapi

        self._k1 = k1
        self._b = b
        self._bm25: Any | None = None
        # Upsert semantics: keyed by chunk_id so re-ingestion cannot create
        # duplicate lexical entries for the same chunk.
        self._by_id: dict[str, DocumentChunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._dirty = True
        self._BM25Okapi = BM25Okapi

    def add(self, chunks: Iterable[DocumentChunk]) -> int:
        """Add or update chunks. Returns the number of unique chunks stored.

        Re-ingesting a chunk with the same `chunk_id` updates the content
        in place — no duplicate entries are created (spec §06 AC-06.3).
        """
        for c in chunks:
            self._by_id[c.id] = c
            self._tokens[c.id] = tokenize(c.content)
        self._dirty = True
        return len(self._by_id)

    def remove(self, chunk_ids: Iterable[str]) -> int:
        """Remove chunks by id. Returns the number actually removed."""
        n = 0
        for cid in chunk_ids:
            if cid in self._by_id:
                del self._by_id[cid]
                del self._tokens[cid]
                n += 1
        if n:
            self._dirty = True
        return n

    def _ensure_built(self) -> None:
        if self._dirty or self._bm25 is None:
            tokens = list(self._tokens.values())
            if not tokens:
                self._bm25 = None
            else:
                self._bm25 = self._BM25Okapi(tokens, k1=self._k1, b=self._b)
            self._dirty = False

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self._ensure_built()
        if self._bm25 is None or not query.strip():
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # Get top_k indices by score, descending. Stable tie-breaking by index.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        out: list[RetrievedChunk] = []
        # Iterate over chunks in the same order passed to BM25Okapi above.
        chunk_ids = list(self._by_id.keys())
        # Determine a "no match" cutoff: if no query token appears anywhere in
        # the corpus, every score will be very negative or zero — return empty.
        # Otherwise include all candidates (BM25 scores can legitimately be
        # negative due to the IDF term with small corpora).
        max_score = float(scores.max()) if len(scores) else 0.0
        for rank, idx in enumerate(order[:top_k]):
            score = float(scores[idx])
            # Skip candidates with no token overlap (score at or below floor).
            if max_score <= 0.0:
                continue
            cid = chunk_ids[idx]
            chunk = self._by_id[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    retrieval_method="lexical",
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )
        return out

    def count(self) -> int:
        return len(self._by_id)

    def reset(self) -> None:
        self._by_id.clear()
        self._tokens.clear()
        self._bm25 = None
        self._dirty = True


# ============================================================================
# In-memory vector store
# ============================================================================


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class InMemoryVectorStore:
    """Pure-Python vector store with cosine similarity.

    Default backend. Deterministic, no external dependencies. Thread-safe
    via a single RLock.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension
        self._vectors: dict[str, np.ndarray] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_to_chunks: dict[str, list[str]] = {}
        self._model: str = "unknown"
        import threading

        self._lock = threading.RLock()

    @property
    def dimension(self) -> int:
        return self._dim

    def add(self, chunks: Sequence[DocumentChunk], vectors: Sequence[list[float]], model: str) -> int:
        """Upsert chunks + vectors. Returns the number of rows actually stored.

        Re-ingesting a chunk with the same `chunk_id` updates its vector and
        content in place; the per-document index is rebuilt so duplicates
        cannot accumulate (spec §06 AC-06.3).
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        added = 0
        with self._lock:
            for c, v in zip(chunks, vectors, strict=True):
                if len(v) != self._dim:
                    raise ValueError(
                        f"vector dimension mismatch: got {len(v)} expected {self._dim}"
                    )
                is_new = c.id not in self._chunks
                self._vectors[c.id] = np.asarray(v, dtype=np.float32)
                self._chunks[c.id] = c
                if is_new:
                    self._doc_to_chunks.setdefault(c.document_id, []).append(c.id)
                added += 1
            self._model = model
        return added

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        with self._lock:
            if not self._vectors:
                return []
            q = np.asarray(query_vector, dtype=np.float32)
            if len(q) != self._dim:
                raise ValueError(
                    f"query vector dimension mismatch: got {len(q)} expected {self._dim}"
                )
            # Compute all similarities, then sort.
            scored: list[tuple[float, str]] = []
            for cid, vec in self._vectors.items():
                if filters and not _matches(self._chunks[cid], filters):
                    continue
                sim = _cosine(q, vec)
                if sim >= min_similarity:
                    scored.append((sim, cid))
            scored.sort(key=lambda x: (-x[0], x[1]))
            out: list[RetrievedChunk] = []
            for rank, (score, cid) in enumerate(scored[:top_k]):
                c = self._chunks[cid]
                out.append(
                    RetrievedChunk(
                        chunk_id=c.id,
                        document_id=c.document_id,
                        content=c.content,
                        score=score,
                        retrieval_method="dense",
                        rank=rank,
                        metadata=c.metadata,
                    )
                )
            return out

    def get(self, chunk_id: str) -> DocumentChunk | None:
        with self._lock:
            return self._chunks.get(chunk_id)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for doc_id, chunk_ids in self._doc_to_chunks.items():
                if not chunk_ids:
                    continue
                first = self._chunks[chunk_ids[0]]
                out.append(
                    {
                        "document_id": doc_id,
                        "chunk_count": len(chunk_ids),
                        "title": first.metadata.get("doc_title"),
                        "content_type": first.metadata.get("doc_content_type"),
                    }
                )
            return out

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            chunk_ids = self._doc_to_chunks.pop(document_id, [])
            for cid in chunk_ids:
                self._vectors.pop(cid, None)
                self._chunks.pop(cid, None)
            return len(chunk_ids)

    def reset(self) -> None:
        with self._lock:
            self._vectors.clear()
            self._chunks.clear()
            self._doc_to_chunks.clear()

    def all_chunks(self) -> list[DocumentChunk]:
        with self._lock:
            return list(self._chunks.values())


def _matches(chunk: DocumentChunk, filters: dict[str, Any]) -> bool:
    """A simple AND-filter over chunk.metadata and document_id."""
    for k, v in filters.items():
        if k == "document_id":
            if chunk.document_id != v:
                return False
        elif k in chunk.metadata:
            if chunk.metadata[k] != v:
                return False
        else:
            return False
    return True


# ============================================================================
# Factory
# ============================================================================


def get_vector_store(dimension: int | None = None, settings: Any = None) -> VectorStore:
    """Return the configured vector store.

    Backend selection rules (no silent production fallback):

      * `FORCE_IN_MEMORY_STORE=true`  → InMemoryVectorStore (explicit offline mode)
      * `DATABASE_URL=sqlite+aiosqlite://...` → InMemoryVectorStore (dev default)
      * `DATABASE_URL=postgresql+asyncpg://...` → PgVectorStore; if it cannot
        be constructed, raise `VectorStoreError` (NEVER silently fall back,
        because that would put production data in RAM and lose it on restart).
    """
    from app.config import get_settings
    from app.domain import VectorStoreError

    s = settings or get_settings()
    dim = dimension or s.embedding_dimension

    # Explicit offline mode: tests, dev, anything that asks for in-memory.
    if s.force_in_memory_store or s.database_url.startswith("sqlite"):
        return InMemoryVectorStore(dim)

    # Production mode: build the pgvector store. Any failure must surface —
    # silently falling back to RAM in production is a data-loss bug.
    if "postgresql" not in s.database_url and "postgres" not in s.database_url:
        raise VectorStoreError(
            f"DATABASE_URL is neither sqlite nor postgresql: {s.database_url!r}. "
            "Set DATABASE_URL=postgresql+asyncpg://... for production, or "
            "FORCE_IN_MEMORY_STORE=true for offline mode.",
            details={"database_url": s.database_url},
        )

    try:
        from app.storage.pgvector_store import PgVectorStore  # type: ignore

        return PgVectorStore(dsn=s.database_url, dimension=dim)
    except ImportError as exc:
        raise VectorStoreError(
            "PgVectorStore requires `asyncpg` and `pgvector`. "
            "Install them (`pip install asyncpg pgvector`) or set "
            "FORCE_IN_MEMORY_STORE=true for offline mode.",
            details={"database_url": s.database_url},
            cause=exc,
        ) from exc
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to construct PgVectorStore: {exc}",
            details={"database_url": s.database_url},
            cause=exc,
        ) from exc


__all__ = [
    "InMemoryVectorStore",
    "LexicalIndex",
    "VectorStore",
    "get_vector_store",
    "tokenize",
]


# Re-export Embedding so callers can construct it cheaply.
_ = Embedding
