"""Prompt construction and citation validation for grounded generation.

The prompt is intentionally minimal but explicit about the rules (spec §13):

  * answer from supplied evidence only;
  * distinguish evidence from uncertainty;
  * avoid unsupported facts;
  * reference source identifiers as `[cite:<chunk_id>]`;
  * say so when evidence is insufficient.

`build_prompt` produces `(system, user)` strings. `validate_citations`
returns only citations that reference an actual retrieved chunk.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.domain import Citation, InvalidCitationError, RAGContext

_CITATION_RE = re.compile(r"\[cite:([a-zA-Z0-9_\-]+)\]")


SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant. You MUST follow these rules:\n"
    "1. Answer ONLY using the evidence provided in the user message.\n"
    "2. Cite every factual statement with [cite:<chunk_id>] markers.\n"
    "3. If the evidence does not contain the answer, say:\n"
    "   'The provided evidence is insufficient to answer this question.'\n"
    "4. Do not invent, assume, or hallucinate facts not present in the evidence.\n"
    "5. Distinguish clearly between what is stated in the evidence and what is uncertain.\n"
    "6. Keep the answer concise and direct.\n"
)


def build_prompt(question: str, context: RAGContext, *, max_chars_per_chunk: int = 1200) -> tuple[str, str]:
    """Build (system, user) prompt strings from the question and context."""
    if not question.strip():
        raise InvalidCitationError("Question must not be empty")

    parts: list[str] = []
    parts.append(f"Question: {question.strip()}\n")
    parts.append("Evidence:\n")
    for item in context.items:
        content = item.content.strip()
        if len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk] + "..."
        parts.append(
            f"[CHUNK {item.chunk_id} :: document_id={item.document_id}]\n{content}\n[/CHUNK]\n"
        )
    parts.append(
        "\nAnswer the question using only the evidence above. "
        "Cite with [cite:<chunk_id>] markers. "
        "If the evidence is insufficient, say so explicitly."
    )
    user_prompt = "\n".join(parts)
    return SYSTEM_PROMPT, user_prompt


def extract_citation_ids(text: str) -> list[str]:
    """Return all chunk ids referenced by `[cite:...]` markers in `text`."""
    return _CITATION_RE.findall(text)


def validate_citations(
    answer_text: str, context: RAGContext
) -> tuple[list[Citation], list[str], bool]:
    """Validate that every `[cite:...]` marker in `answer_text` references
    an actual chunk in `context`.

    Returns:
        citations: list[Citation] — validated, in order of first appearance.
        rejected_ids: list[str] — ids found in text but not in context.
        evidence_sufficient: bool — True if at least one valid citation exists.
    """
    valid_ids = {item.chunk_id for item in context.items}
    seen: set[str] = set()
    citations: list[Citation] = []
    rejected: list[str] = []
    for cid in extract_citation_ids(answer_text):
        if cid not in valid_ids:
            rejected.append(cid)
            continue
        if cid in seen:
            continue
        seen.add(cid)
        item = next(i for i in context.items if i.chunk_id == cid)
        citations.append(
            Citation(
                chunk_id=cid,
                document_id=item.document_id,
                snippet=item.content[:280],
            )
        )
    return citations, rejected, bool(citations)


def remove_invalid_citations(text: str, rejected_ids: Iterable[str]) -> str:
    """Strip citation markers for ids that did not validate."""
    for cid in set(rejected_ids):
        text = text.replace(f"[cite:{cid}]", "[citation removed]")
    return text


__all__ = [
    "SYSTEM_PROMPT",
    "build_prompt",
    "extract_citation_ids",
    "remove_invalid_citations",
    "validate_citations",
]
