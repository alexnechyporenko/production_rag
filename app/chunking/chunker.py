"""Document chunking.

Chunkers are deterministic: the same input always produces the same chunk ids.
Three strategies are provided:

  - FixedCharChunker:   chunk by character window with optional overlap
  - SentenceChunker:    chunk by sentence boundaries (regex-based)
  - RecursiveChunker:   split by paragraph → sentence → word, recursively

All chunkers implement the `Chunker` protocol and return `list[DocumentChunk]`
with `deterministic_chunk_id` ids.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain import Document, DocumentChunk, deterministic_chunk_id


@runtime_checkable
class Chunker(Protocol):
    """Chunker protocol. Implementations MUST be deterministic."""

    def chunk(self, document: Document) -> list[DocumentChunk]: ...


# --- helpers ----------------------------------------------------------------


def _build_chunk(document: Document, sequence: int, content: str, start: int, end: int) -> DocumentChunk:
    return DocumentChunk(
        id=deterministic_chunk_id(document.id, sequence, content),
        document_id=document.id,
        content=content,
        sequence=sequence,
        char_start=start,
        char_end=end,
        metadata={
            "doc_title": document.metadata.title,
            "doc_source": document.metadata.source.value,
            "doc_content_type": document.metadata.content_type,
        },
    )


# --- fixed char chunker -----------------------------------------------------


class FixedCharChunker:
    """Chunk by a fixed character window with optional overlap."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.content
        n = len(text)
        if n == 0:
            return []
        step = max(1, self.chunk_size - self.overlap)
        out: list[DocumentChunk] = []
        seq = 0
        pos = 0
        while pos < n:
            end = min(pos + self.chunk_size, n)
            piece = text[pos:end]
            if piece.strip():
                out.append(_build_chunk(document, seq, piece, pos, end))
                seq += 1
            if end == n:
                break
            pos += step
        return out


# --- sentence chunker -------------------------------------------------------


_SENTENCE_SPLIT_RE = __import__("re").compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class SentenceChunker:
    """Group whole sentences until target size, then start a new chunk."""

    def __init__(self, target_chars: int = 1200, overlap_sentences: int = 1) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars must be > 0")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.target_chars = target_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.content
        if not text.strip():
            return []
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            sentences = [text]
        out: list[DocumentChunk] = []
        seq = 0
        i = 0
        char_cursor = 0  # running offset into document.content
        while i < len(sentences):
            buf: list[str] = []
            total = 0
            j = i
            while j < len(sentences) and (total == 0 or total + len(sentences[j]) <= self.target_chars):
                buf.append(sentences[j])
                total += len(sentences[j]) + 1
                j += 1
            content = " ".join(buf)
            start = char_cursor
            end = min(start + len(content), len(text))
            if content.strip():
                out.append(_build_chunk(document, seq, content, start, end))
                seq += 1
            char_cursor = end
            if j == i:  # one oversized sentence
                j = i + 1
            i = max(j - self.overlap_sentences, i + 1)
        return out


# --- recursive chunker ------------------------------------------------------


class RecursiveChunker:
    """Recursive paragraph → sentence → word chunker, similar to LlamaIndex."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.content
        if not text.strip():
            return []
        pieces = _recursive_split(text, self.chunk_size, separators=["\n\n", "\n", ". ", " "])
        out: list[DocumentChunk] = []
        seq = 0
        pos = 0
        for piece in pieces:
            if not piece.strip():
                continue
            start = text.find(piece, pos)
            if start < 0:
                start = pos
            end = min(start + len(piece), len(text))
            out.append(_build_chunk(document, seq, piece, start, end))
            seq += 1
            pos = end
        # Re-apply overlap by merging if too small (best-effort)
        if self.overlap > 0 and len(out) > 1:
            out = _apply_overlap(out, document, self.overlap)
        return out


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    sep = separators[0]
    rest = separators[1:]
    parts = text.split(sep) if sep != " " else list(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if len(p) > chunk_size and rest:
            sub = _recursive_split(p, chunk_size, rest)
            for s in sub:
                if len(buf) + len(s) + len(sep) <= chunk_size:
                    buf = (buf + sep + s) if buf else s
                else:
                    if buf:
                        out.append(buf)
                    buf = s
            continue
        candidate = (buf + sep + p) if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _apply_overlap(chunks: list[DocumentChunk], doc: Document, overlap: int) -> list[DocumentChunk]:
    if len(chunks) <= 1:
        return chunks
    new_chunks: list[DocumentChunk] = [chunks[0]]
    for i, c in enumerate(chunks[1:], start=1):
        prev_tail = chunks[i - 1].content[-overlap:]
        merged = (prev_tail + " " + c.content).strip()
        new_chunks.append(
            DocumentChunk(
                id=deterministic_chunk_id(doc.id, c.sequence, merged),
                document_id=doc.id,
                content=merged,
                sequence=c.sequence,
                char_start=c.char_start - overlap if c.char_start >= overlap else c.char_start,
                char_end=c.char_end,
                metadata=c.metadata,
            )
        )
    return new_chunks


# --- factory ----------------------------------------------------------------


def get_chunker(name: str = "recursive", **kwargs: int) -> Chunker:
    """Return a chunker by name."""
    if name == "fixed":
        return FixedCharChunker(**kwargs)
    if name == "sentence":
        return SentenceChunker(**kwargs)
    if name == "recursive":
        return RecursiveChunker(**kwargs)
    raise ValueError(f"Unknown chunker: {name}")


__all__ = [
    "Chunker",
    "FixedCharChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "get_chunker",
]
