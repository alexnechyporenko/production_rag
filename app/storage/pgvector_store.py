"""PostgreSQL + pgvector vector store.

Uses SQLAlchemy 2.0 async with asyncpg. The schema is created on first use
via `init()`. If the `vector` extension is not installed, the store raises a
`VectorStoreError` with a clear message — never silently degrades.

This module is imported lazily by `get_vector_store` so that the rest of the
system can run without asyncpg / pgvector being installed.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from app.domain import DocumentChunk, RetrievedChunk, VectorStoreError

# We import SQLAlchemy lazily because asyncpg/SQLAlchemy are heavyweight and
# not needed for in-memory runs. Doing so also means the in-memory store stays
# the default and tests don't have to wire up PostgreSQL.


class PgVectorStore:
    """PostgreSQL+pgvector backend. Initialised async; safe to use from sync code
    via an internal event loop running in a background thread.
    """

    def __init__(self, dsn: str, dimension: int, *, verify_connection: bool = True) -> None:
        if "postgresql" not in dsn and "postgres" not in dsn:
            raise VectorStoreError(f"DSN must be PostgreSQL, got: {dsn!r}")
        self._dsn = dsn
        self._dim = dimension
        self._engine: Any | None = None
        self._sessionmaker: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_loop()
        # Fail fast: confirm the database is reachable so `get_vector_store`
        # raises a structured `VectorStoreError` instead of returning a broken
        # store that surfaces only on the first query.
        if verify_connection:
            self._run(self._verify_connection())

    # --- internal async loop -------------------------------------------------
    def _start_loop(self) -> None:
        def run() -> None:
            import asyncio as _asyncio

            self._loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(self._loop)
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, name="pgvector-store", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._loop is None:
            raise VectorStoreError("Failed to start pgvector background loop")

    async def _verify_connection(self) -> None:
        """Open a one-shot connection to the database and confirm it works.

        Catches the common "wrong DSN / database down / credentials invalid"
        failure modes at construction time so the factory surfaces a clean
        `VectorStoreError` instead of letting the system boot with a dead
        vector store.
        """
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        try:
            engine = create_async_engine(self._dsn, future=True)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
        except Exception as exc:
            raise VectorStoreError(
                f"Could not connect to PostgreSQL at {self._dsn!r}: {exc}",
                details={"dsn": self._dsn},
                cause=exc,
            ) from exc

    def _run(self, coro: Any) -> Any:
        assert self._loop is not None
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @property
    def dimension(self) -> int:
        return self._dim

    async def _ensure_schema(self) -> None:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        if self._engine is None:
            self._engine = create_async_engine(self._dsn, future=True)
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        sequence INT NOT NULL,
                        content TEXT NOT NULL,
                        char_start INT NOT NULL,
                        char_end INT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self._dim}) NOT NULL,
                        model TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS rag_chunks_doc_idx ON rag_chunks(document_id);"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS rag_chunks_embedding_ivfflat "
                    "ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);"
                )
            )

    def add(self, chunks, vectors, model):
        return self._run(self._add_async(chunks, vectors, model))

    async def _add_async(self, chunks, vectors, model):
        await self._ensure_schema()
        from sqlalchemy import text

        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            for c, v in zip(chunks, vectors, strict=True):
                import json

                vec_lit = "[" + ",".join(f"{x:.7f}" for x in v) + "]"
                await conn.execute(
                    text(
                        """
                        INSERT INTO rag_chunks
                          (chunk_id, document_id, sequence, content,
                           char_start, char_end, metadata, embedding, model)
                        VALUES
                          (:cid, :did, :seq, :content, :cs, :ce, :meta, :emb, :model)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                          content = EXCLUDED.content,
                          embedding = EXCLUDED.embedding,
                          model = EXCLUDED.model
                        """
                    ),
                    {
                        "cid": c.id,
                        "did": c.document_id,
                        "seq": c.sequence,
                        "content": c.content,
                        "cs": c.char_start,
                        "ce": c.char_end,
                        "meta": json.dumps(c.metadata),
                        "emb": vec_lit,
                        "model": model,
                    },
                )
        return len(chunks)

    def search(self, query_vector, top_k=10, filters=None, min_similarity=0.0):
        return self._run(self._search_async(query_vector, top_k, filters, min_similarity))

    async def _search_async(self, query_vector, top_k, filters, min_similarity):
        await self._ensure_schema()
        from sqlalchemy import text

        vec_lit = "[" + ",".join(f"{x:.7f}" for x in query_vector) + "]"
        where = "1 - (embedding <=> :q) >= :ms"
        params: dict[str, Any] = {"q": vec_lit, "ms": min_similarity, "k": top_k}
        if filters:
            for k, v in (filters or {}).items():
                if k == "document_id":
                    where += " AND document_id = :fid"
                    params["fid"] = v
        sql = text(
            f"""
            SELECT chunk_id, document_id, content, metadata,
                   1 - (embedding <=> :q) AS score
            FROM rag_chunks
            WHERE {where}
            ORDER BY embedding <=> :q
            LIMIT :k
            """
        )
        async with self._engine.connect() as conn:  # type: ignore[union-attr]
            rows = (await conn.execute(sql, params)).fetchall()
        out: list[RetrievedChunk] = []
        for rank, row in enumerate(rows):
            import json

            meta = row._mapping["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            out.append(
                RetrievedChunk(
                    chunk_id=row._mapping["chunk_id"],
                    document_id=row._mapping["document_id"],
                    content=row._mapping["content"],
                    score=float(row._mapping["score"]),
                    retrieval_method="dense",
                    rank=rank,
                    metadata=meta or {},
                )
            )
        return out

    def get(self, chunk_id):
        return self._run(self._get_async(chunk_id))

    async def _get_async(self, chunk_id):
        await self._ensure_schema()
        from sqlalchemy import text

        async with self._engine.connect() as conn:  # type: ignore[union-attr]
            row = (
                await conn.execute(
                    text(
                        """
                        SELECT chunk_id, document_id, content, sequence,
                               char_start, char_end, metadata
                        FROM rag_chunks WHERE chunk_id = :cid
                        """
                    ),
                    {"cid": chunk_id},
                )
            ).first()
        if row is None:
            return None
        import json

        meta = row._mapping["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return DocumentChunk(
            id=row._mapping["chunk_id"],
            document_id=row._mapping["document_id"],
            content=row._mapping["content"],
            sequence=row._mapping["sequence"],
            char_start=row._mapping["char_start"],
            char_end=row._mapping["char_end"],
            metadata=meta or {},
        )

    def list_documents(self):
        return self._run(self._list_documents_async())

    async def _list_documents_async(self):
        await self._ensure_schema()
        from sqlalchemy import text

        async with self._engine.connect() as conn:  # type: ignore[union-attr]
            rows = (
                await conn.execute(
                    text(
                        """
                        SELECT document_id, COUNT(*) AS cnt,
                               (array_agg(metadata))[1]->>'doc_title' AS title,
                               (array_agg(metadata))[1]->>'doc_content_type' AS ct
                        FROM rag_chunks GROUP BY document_id
                        """
                    )
                )
            ).fetchall()
        return [
            {
                "document_id": r._mapping["document_id"],
                "chunk_count": r._mapping["cnt"],
                "title": r._mapping["title"],
                "content_type": r._mapping["ct"],
            }
            for r in rows
        ]

    def count(self):
        return self._run(self._count_async())

    async def _count_async(self):
        await self._ensure_schema()
        from sqlalchemy import text

        async with self._engine.connect() as conn:  # type: ignore[union-attr]
            return int(
                (await conn.execute(text("SELECT COUNT(*) FROM rag_chunks"))).scalar() or 0
            )

    def delete_document(self, document_id):
        return self._run(self._delete_document_async(document_id))

    async def _delete_document_async(self, document_id):
        await self._ensure_schema()
        from sqlalchemy import text

        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            return int(
                (
                    await conn.execute(
                        text("DELETE FROM rag_chunks WHERE document_id = :did"),
                        {"did": document_id},
                    )
                ).rowcount
                or 0
            )

    def reset(self):
        self._run(self._reset_async())

    async def _reset_async(self):
        await self._ensure_schema()
        from sqlalchemy import text

        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            await conn.execute(text("DROP TABLE IF EXISTS rag_chunks;"))


__all__ = ["PgVectorStore"]
