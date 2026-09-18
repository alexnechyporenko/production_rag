"""Ingestion service: orchestrates loader + chunker + downstream storage.

`IngestionService.ingest_text` / `ingest_bytes` / `ingest_file` returns a list of
the chunks that were created. Document-level failure is isolated: a single bad
document never aborts a batch run (see `ingest_many`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.domain import Document, DocumentChunk, DocumentSource
from app.ingestion.loader import DocumentLoader


class IngestionResult:
    """Result of an ingestion operation. Cheap dataclass-style holder."""

    def __init__(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
        warnings: list[str] | None = None,
    ) -> None:
        self.document = document
        self.chunks = list(chunks)
        self.warnings = warnings or []

    @property
    def document_id(self) -> str:
        return self.document.id

    @property
    def chunk_ids(self) -> list[str]:
        return [c.id for c in self.chunks]

    def __repr__(self) -> str:
        return (
            f"IngestionResult(document_id={self.document_id!r}, "
            f"chunks={len(self.chunks)}, warnings={len(self.warnings)})"
        )


class IngestionService:
    """High-level facade: load → chunk → return chunks ready for embedding/storage.

    The service itself does NOT touch the vector store or the lexical index —
    those are the responsibility of the storage layer. This keeps ingestion
    independently testable (P-05) and lets the API layer decide where to persist.
    """

    def __init__(self, loader: DocumentLoader | None = None, chunker: Any | None = None) -> None:
        from app.chunking import get_chunker

        self.loader = loader or DocumentLoader()
        self.chunker = chunker or get_chunker("recursive", chunk_size=1200, overlap=200)

    # --- single-document ---------------------------------------------------
    def ingest_text(
        self,
        text: str,
        *,
        content_type: str = "text/plain",
        title: str | None = None,
        source: DocumentSource = DocumentSource.api,
        **metadata: Any,
    ) -> IngestionResult:
        doc = self.loader.load_text(text, content_type=content_type, title=title, source=source, **metadata)
        return IngestionResult(doc, self.chunker.chunk(doc))

    def ingest_bytes(
        self,
        data: bytes,
        *,
        content_type: str = "text/plain",
        title: str | None = None,
        source: DocumentSource = DocumentSource.api,
        **metadata: Any,
    ) -> IngestionResult:
        doc = self.loader.load_bytes(data, content_type=content_type, title=title, source=source, **metadata)
        return IngestionResult(doc, self.chunker.chunk(doc))

    def ingest_file(self, path: str, **metadata: Any) -> IngestionResult:
        doc = self.loader.load_file(path, **metadata)
        return IngestionResult(doc, self.chunker.chunk(doc))

    # --- batch -------------------------------------------------------------
    def ingest_many(self, items: Iterable[dict[str, Any]]) -> list[IngestionResult]:
        """Ingest a batch. A failing item is captured as a warning, not an exception.

        Each item is a dict matching one of the ingest_* signatures. The dict
        must include a `kind` key with value `text` / `bytes` / `file`.
        """
        out: list[IngestionResult] = []
        for i, item in enumerate(items):
            kind = item.get("kind", "text")
            try:
                if kind == "text":
                    r = self.ingest_text(**{k: v for k, v in item.items() if k != "kind"})
                elif kind == "bytes":
                    r = self.ingest_bytes(**{k: v for k, v in item.items() if k != "kind"})
                elif kind == "file":
                    r = self.ingest_file(**{k: v for k, v in item.items() if k != "kind"})
                else:
                    raise ValueError(f"unknown kind: {kind!r}")
                out.append(r)
            except Exception as exc:
                # Failure isolation: record and continue.
                out.append(
                    IngestionResult(
                        document=Document(content="__failed__", id=f"failed-{i}"),
                        chunks=[],
                        warnings=[f"{type(exc).__name__}: {exc}"],
                    )
                )
        return out


__all__ = ["IngestionResult", "IngestionService"]
