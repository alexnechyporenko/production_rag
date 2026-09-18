"""Context builder: deterministic, budget-bounded LLM context assembly.

Responsibilities (spec §12):
  1. remove duplicate chunks
  2. enforce maximum chunk count
  3. enforce maximum context size (chars and estimated tokens)
  4. preserve source metadata + citation identifiers
  5. prioritise higher-ranked evidence
  6. produce deterministic output

The builder is pure: same input → same output, every time.
"""

from __future__ import annotations

from app.domain import ContextItem, RAGContext, RetrievedChunk

# Cheap & deterministic token estimate: ~4 chars per token for English.
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Rough but reproducible token estimate."""
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


class ContextBuilder:
    """Assemble a bounded RAGContext from ranked candidates."""

    def __init__(
        self,
        max_chunks: int = 12,
        max_chars: int = 14000,
        max_tokens: int = 3500,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0")
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_chunks = max_chunks
        self.max_chars = max_chars
        self.max_tokens = max_tokens

    def build(self, candidates: list[RetrievedChunk]) -> RAGContext:
        seen: set[str] = set()
        items: list[ContextItem] = []
        total_chars = 0
        total_tokens = 0
        truncated = False

        # Assume candidates are already sorted by score desc; preserve that order.
        for c in candidates:
            if c.chunk_id in seen:
                continue
            item_tokens = estimate_tokens(c.content)
            item_chars = len(c.content)
            if len(items) >= self.max_chunks:
                truncated = True
                break
            if total_chars + item_chars > self.max_chars:
                truncated = True
                break
            if total_tokens + item_tokens > self.max_tokens:
                truncated = True
                break
            items.append(
                ContextItem(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    rank=len(items),
                    score=c.score,
                    retrieval_method=c.retrieval_method,
                    metadata=c.metadata,
                )
            )
            seen.add(c.chunk_id)
            total_chars += item_chars
            total_tokens += item_tokens

        return RAGContext(
            items=items,
            total_chars=total_chars,
            estimated_tokens=total_tokens,
            deduped_count=len(items),
            truncated=truncated,
        )


__all__ = ["ContextBuilder", "ContextItem", "estimate_tokens"]
