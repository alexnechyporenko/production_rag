"""Grounded answer service.

End-to-end generation step: given a `RAGContext` and a question, call the LLM
provider, validate the citations, compute token usage and estimated cost.

The service is responsible for:
  * assembling the prompt deterministically;
  * calling the LLM provider (sync or async);
  * parsing and validating citations;
  * computing `CostEstimate` using configuration-driven pricing
    (unknown pricing → `pricing_known=False`, never fabricated).
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.domain import (
    CostEstimate,
    RAGAnswer,
    RAGContext,
    TokenUsage,
    now_ms,
)
from app.generation.prompt import build_prompt, remove_invalid_citations, validate_citations
from app.generation.provider import LLMProvider, LLMResponse, get_llm_provider


def estimate_cost(usage: TokenUsage, settings: Settings | None = None) -> CostEstimate:
    """Compute a USD cost estimate from token usage.

    Returns `CostEstimate.unknown()` when pricing for the model is not
    configured — never a fabricated number (spec §16).
    """
    s = settings or get_settings()
    in_price = s.llm_input_price_per_1k(usage.model)
    out_price = s.llm_output_price_per_1k(usage.model)
    if in_price is None or out_price is None:
        return CostEstimate.unknown()
    in_cost = (usage.input_tokens / 1000.0) * in_price
    out_cost = (usage.output_tokens / 1000.0) * out_price
    return CostEstimate(
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=in_cost + out_cost,
        pricing_known=True,
    )


class GenerationService:
    """Orchestrates one grounded generation call."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.llm_provider = llm_provider or get_llm_provider(settings)
        self.settings = settings or get_settings()

    def generate(
        self,
        question: str,
        context: RAGContext,
        *,
        query_id: str,
        temperature: float | None = None,
    ) -> RAGAnswer:
        start = now_ms()
        system, user = build_prompt(question, context)
        resp = self.llm_provider.complete(system, user, temperature=temperature)
        return self._finalize(question, context, query_id, resp, start, warnings=[])

    async def generate_async(
        self,
        question: str,
        context: RAGContext,
        *,
        query_id: str,
        temperature: float | None = None,
    ) -> RAGAnswer:
        start = now_ms()
        system, user = build_prompt(question, context)
        resp = await self.llm_provider.complete_async(system, user, temperature=temperature)
        return self._finalize(question, context, query_id, resp, start, warnings=[])

    # --- internal ---------------------------------------------------------
    def _finalize(
        self,
        question: str,
        context: RAGContext,
        query_id: str,
        resp: LLMResponse,
        start_ms: float,
        warnings: list[str],
    ) -> RAGAnswer:
        citations, rejected, evidence_sufficient = validate_citations(resp.text, context)
        text = resp.text
        if rejected:
            warnings.append(f"Rejected {len(rejected)} invalid citation(s): {rejected[:5]}")
            text = remove_invalid_citations(text, rejected)
        cost = estimate_cost(resp.token_usage, self.settings)
        return RAGAnswer(
            query_id=query_id,
            answer=text,
            citations=citations,
            confidence=_confidence(citations, context, evidence_sufficient),
            used_chunks=[c.chunk_id for c in citations],
            token_usage=resp.token_usage,
            cost_estimate=cost,
            latency_ms=now_ms() - start_ms,
            context=context,
            model=resp.model,
            evidence_sufficient=evidence_sufficient,
            warnings=warnings,
        )


def _confidence(
    citations: list, context: RAGContext, evidence_sufficient: bool
) -> float:
    """A heuristic confidence score in [0, 1].

    Higher when more of the top-ranked evidence is actually cited.
    """
    if not evidence_sufficient or not citations:
        return 0.0
    cited = {c.chunk_id for c in citations}
    top = context.items[: min(3, len(context.items))]
    if not top:
        return 0.0
    hits = sum(1 for i in top if i.chunk_id in cited)
    return round(hits / len(top), 3)


__all__ = ["GenerationService", "estimate_cost"]
