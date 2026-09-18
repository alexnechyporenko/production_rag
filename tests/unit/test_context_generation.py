"""Unit tests for context builder + generation + citation validation."""

from __future__ import annotations

import pytest
from app.context import ContextBuilder, estimate_tokens
from app.domain import (
    ContextItem,
    InvalidCitationError,
    RAGContext,
    RetrievedChunk,
)
from app.generation import (
    GenerationService,
    build_prompt,
    extract_citation_ids,
    remove_invalid_citations,
    validate_citations,
)
from app.generation.provider import MockLLMProvider


def _retrieved(i: int, content: str, score: float = 0.9, method: str = "hybrid") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{i}",
        document_id=f"doc-{i}",
        content=content,
        score=score,
        retrieval_method=method,
        rank=i,
        metadata={},
    )


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("a") == 1  # at least 1
        assert estimate_tokens("") == 0
        # 8 chars / 4 = 2 tokens
        assert estimate_tokens("12345678") == 2


class TestContextBuilder:
    def test_basic_assembly(self):
        b = ContextBuilder(max_chunks=3, max_chars=10000, max_tokens=10000)
        ctx = b.build([_retrieved(i, f"chunk {i}") for i in range(5)])
        assert len(ctx.items) == 3
        assert ctx.truncated is True
        # items keep their original ranks
        assert [i.rank for i in ctx.items] == [0, 1, 2]

    def test_deduplication(self):
        b = ContextBuilder(max_chunks=10, max_chars=10000, max_tokens=10000)
        c1 = _retrieved(0, "same")
        c2 = _retrieved(0, "same")  # same id, same content
        ctx = b.build([c1, c2])
        assert len(ctx.items) == 1
        assert ctx.deduped_count == 1

    def test_token_budget_truncates(self):
        # 1 token per 4 chars. Set max_tokens so only 1 chunk fits.
        b = ContextBuilder(max_chunks=10, max_chars=10000, max_tokens=2)
        chunks = [_retrieved(0, "very long content that exceeds the budget" * 5)]
        ctx = b.build(chunks)
        assert len(ctx.items) == 0  # first chunk alone is too big
        assert ctx.truncated is True

    def test_char_budget_truncates(self):
        b = ContextBuilder(max_chunks=10, max_chars=10, max_tokens=10000)
        ctx = b.build([_retrieved(0, "this is way too long for ten chars")])
        assert len(ctx.items) == 0
        assert ctx.truncated is True

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            ContextBuilder(max_chunks=0)
        with pytest.raises(ValueError):
            ContextBuilder(max_chars=0)
        with pytest.raises(ValueError):
            ContextBuilder(max_tokens=0)

    def test_deterministic_output(self):
        b = ContextBuilder(max_chunks=3, max_chars=10000, max_tokens=10000)
        cands = [_retrieved(i, f"chunk {i}") for i in range(10)]
        ctx1 = b.build(cands)
        ctx2 = b.build(cands)
        assert [i.chunk_id for i in ctx1.items] == [i.chunk_id for i in ctx2.items]


class TestPromptBuilder:
    def test_build_prompt_includes_question_and_chunks(self):
        b = ContextBuilder(max_chunks=2, max_chars=10000, max_tokens=10000)
        ctx = b.build([_retrieved(0, "evidence A"), _retrieved(1, "evidence B")])
        system, user = build_prompt("Q1", ctx)
        assert "Q1" in user
        assert "evidence A" in user
        assert "evidence B" in user
        assert "grounded" in system.lower()

    def test_empty_question_rejected(self):
        b = ContextBuilder()
        ctx = b.build([_retrieved(0, "x")])
        with pytest.raises(InvalidCitationError):
            build_prompt("   ", ctx)


class TestCitationValidation:
    def _ctx(self, ids: list[str]) -> RAGContext:
        items = [
            ContextItem(chunk_id=cid, document_id=f"doc-{cid}", content=cid, rank=i, score=1.0, retrieval_method="hybrid")
            for i, cid in enumerate(ids)
        ]
        return RAGContext(items=items, total_chars=10, estimated_tokens=2, deduped_count=len(items))

    def test_extract_ids(self):
        text = "A [cite:chunk-1] and [cite:chunk-2]."
        assert extract_citation_ids(text) == ["chunk-1", "chunk-2"]

    def test_validate_keeps_valid_citations(self):
        ctx = self._ctx(["chunk-1", "chunk-2"])
        text = "X [cite:chunk-1]. Y [cite:chunk-2]."
        citations, rejected, sufficient = validate_citations(text, ctx)
        assert len(citations) == 2
        assert rejected == []
        assert sufficient is True

    def test_validate_rejects_unknown_ids(self):
        ctx = self._ctx(["chunk-1"])
        text = "X [cite:chunk-1] and [cite:chunk-unknown]."
        citations, rejected, sufficient = validate_citations(text, ctx)
        assert len(citations) == 1
        assert "chunk-unknown" in rejected
        assert sufficient is True

    def test_validate_no_citations_insufficient(self):
        ctx = self._ctx(["chunk-1"])
        text = "Plain answer with no citations."
        citations, _rejected, sufficient = validate_citations(text, ctx)
        assert citations == []
        assert sufficient is False

    def test_validate_dedupes(self):
        ctx = self._ctx(["chunk-1"])
        text = "X [cite:chunk-1]. Y [cite:chunk-1] again."
        citations, _rejected, sufficient = validate_citations(text, ctx)
        assert len(citations) == 1
        assert sufficient is True

    def test_remove_invalid_citations(self):
        text = "X [cite:bad1] Y [cite:good] Z [cite:bad2]."
        out = remove_invalid_citations(text, ["bad1", "bad2"])
        assert "[cite:bad1]" not in out
        assert "[cite:bad2]" not in out
        assert "[cite:good]" in out


class TestGenerationService:
    @pytest.mark.asyncio
    async def test_generate_returns_grounded_answer(self):
        b = ContextBuilder(max_chunks=3, max_chars=10000, max_tokens=10000)
        ctx = b.build([_retrieved(0, "hybrid search combines dense and lexical retrieval")])
        svc = GenerationService(llm_provider=MockLLMProvider())
        ans = await svc.generate_async(question="What is hybrid search?", context=ctx, query_id="q1")
        assert ans.query_id == "q1"
        assert ans.evidence_sufficient is True
        assert len(ans.citations) >= 1
        assert ans.token_usage.total_tokens > 0

    def test_cost_unknown_for_mock(self):
        from app.domain import TokenUsage
        from app.generation.service import estimate_cost

        # Mock model is not in the price table.
        usage = TokenUsage.from_pair(input_tokens=100, output_tokens=50, model="mock-llm-v1")
        cost = estimate_cost(usage)
        assert cost.pricing_known is False
        assert cost.total_cost == 0.0

    def test_cost_known_for_gpt_4o_mini(self):
        from app.domain import TokenUsage
        from app.generation.service import estimate_cost

        usage = TokenUsage.from_pair(input_tokens=1000, output_tokens=500, model="gpt-4o-mini")
        cost = estimate_cost(usage)
        assert cost.pricing_known is True
        # Prices are per 1K tokens: $0.000150/1K in, $0.000600/1K out.
        # 1000 in * $0.00015/1K + 500 out * $0.0006/1K = $0.00015 + $0.00030 = $0.00045
        assert abs(cost.total_cost - 0.00045) < 1e-9
