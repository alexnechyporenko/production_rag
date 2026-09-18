"""LLM provider abstraction.

Two implementations:
  - MockLLMProvider: deterministic offline generator. Returns a grounded answer
    by stitching together retrieved chunks. Used for tests, benchmarks, and
    for environments without an LLM API key.
  - OpenAILLMProvider: thin OpenAI-compatible client (works against OpenAI,
    LM Studio, vLLM, Ollama with OpenAI shim, etc.).

Both return an `LLMResponse` carrying the text plus token usage. Token usage
for the mock provider is estimated using the same heuristic as the context
builder (≈4 chars / token).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol, runtime_checkable

import httpx

from app.config import Settings, get_settings
from app.domain import (
    LLMOutputError,
    LLMProviderError,
    LLMTimeoutError,
    TokenUsage,
)


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> LLMResponse: ...

    async def complete_async(
        self, system: str, user: str, *, temperature: float | None = None
    ) -> LLMResponse: ...


# ============================================================================
# LLMResponse model
# ============================================================================


class LLMResponse:
    """Minimal response wrapper shared by all providers."""

    def __init__(
        self,
        text: str,
        *,
        token_usage: TokenUsage,
        model: str,
        finish_reason: str | None = None,
    ) -> None:
        self.text = text
        self.token_usage = token_usage
        self.model = model
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.token_usage.total_tokens}, "
            f"text_len={len(self.text)})"
        )


# ============================================================================
# Mock provider (deterministic, offline)
# ============================================================================


_CITATION_RE = re.compile(r"\[cite:([a-zA-Z0-9_\-]+)\]")


class MockLLMProvider:
    """Deterministic offline LLM.

    Behaviour:
      - Inspects the prompt for embedded chunk records of the form
        `[CHUNK <id> :: document_id=<doc>]<content>[/CHUNK]`.
      - Picks the top-N chunks by prompt position and stitches them together.
      - Emits citations as `[cite:<chunk_id>]` markers.
      - Returns estimated token usage derived from chars.

    This is a *real* implementation of the `LLMProvider` protocol: the rest
    of the system treats it as an opaque model, so swapping in the OpenAI
    provider is a config change, not a code change.
    """

    def __init__(self, model: str = "mock-llm-v1") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> LLMResponse:
        return asyncio.run(self.complete_async(system, user, temperature=temperature))

    async def complete_async(
        self, system: str, user: str, *, temperature: float | None = None
    ) -> LLMResponse:
        from app.context import estimate_tokens

        await asyncio.sleep(0)
        chunks = _extract_chunks_from_prompt(user)
        if not chunks:
            text = (
                "I could not find any evidence in the provided context. "
                "The answer cannot be determined from the available sources."
            )
        else:
            text = self._compose_answer(user, chunks)
        in_tok = estimate_tokens(system) + estimate_tokens(user)
        out_tok = estimate_tokens(text)
        return LLMResponse(
            text=text,
            token_usage=TokenUsage.from_pair(
                input_tokens=in_tok, output_tokens=out_tok, model=self._model
            ),
            model=self._model,
            finish_reason="stop",
        )

    # --- internals ---------------------------------------------------------
    def _compose_answer(self, user_prompt: str, chunks: list[tuple[str, str, str]]) -> str:
        """`chunks` is a list of (chunk_id, document_id, content)."""
        # Take the first 3 chunks (priority by prompt position).
        selected = chunks[:3]
        # Detect the user query by looking for a literal "Question:" line.
        question = _extract_question(user_prompt)
        lines: list[str] = []
        if question:
            lines.append(f"In response to: {question}")
        else:
            lines.append("Based on the retrieved evidence:")
        for cid, _did, content in selected:
            # Truncate each chunk to keep the answer compact.
            snippet = content.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            lines.append(f"- [cite:{cid}] {snippet}")
        if len(chunks) > len(selected):
            lines.append(
                f"(Additional supporting evidence available: {len(chunks) - len(selected)} chunks.)"
            )
        lines.append("")
        lines.append(
            "Note: this answer was generated by the mock LLM and uses only the provided context."
        )
        return "\n".join(lines)


_CHUNK_RECORD_RE = re.compile(
    r"\[CHUNK\s+(?P<id>[^\s:]+)\s+::\s+document_id=(?P<doc>[^\]]+)\](?P<content>.*?)\[/CHUNK\]",
    re.DOTALL,
)


def _extract_chunks_from_prompt(prompt: str) -> list[tuple[str, str, str]]:
    """Pull structured chunk records out of the prompt."""
    out: list[tuple[str, str, str]] = []
    for m in _CHUNK_RECORD_RE.finditer(prompt):
        cid = m.group("id").strip()
        did = m.group("doc").strip()
        content = m.group("content").strip()
        if cid and content:
            out.append((cid, did, content))
    return out


def _extract_question(prompt: str) -> str | None:
    for line in prompt.splitlines():
        s = line.strip()
        if s.lower().startswith("question:"):
            return s[len("question:"):].strip()
    return None


# ============================================================================
# OpenAI-compatible provider
# ============================================================================


class OpenAILLMProvider:
    """Calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        s = get_settings()
        self._api_key = api_key if api_key is not None else s.llm_api_key
        self._base_url = (base_url or s.llm_base_url).rstrip("/")
        self._model = model or s.llm_model
        self._timeout = timeout or s.llm_timeout_seconds
        self._max_retries = max_retries if max_retries is not None else s.llm_max_retries
        if not self._api_key:
            raise LLMProviderError(
                "LLM_API_KEY is not set; cannot use OpenAILLMProvider.",
                details={"base_url": self._base_url, "model": self._model},
            )

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> LLMResponse:
        return asyncio.run(self.complete_async(system, user, temperature=temperature))

    async def complete_async(
        self, system: str, user: str, *, temperature: float | None = None
    ) -> LLMResponse:
        s = get_settings()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else s.llm_temperature,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self._base_url}/chat/completions"
        last_exc: Exception | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 429:
                    raise LLMTimeoutError("LLM rate-limited", details={"status": 429})
                resp.raise_for_status()
                data = resp.json()
                text = _extract_text(data)
                usage = _extract_usage(data, self._model)
                finish = _extract_finish(data)
                return LLMResponse(
                    text=text,
                    token_usage=usage,
                    model=self._model,
                    finish_reason=finish,
                )
            except httpx.TimeoutException as exc:
                last_exc = LLMTimeoutError("LLM request timed out", cause=exc)
                continue
            except httpx.HTTPStatusError as exc:
                last_exc = LLMProviderError(
                    f"LLM API returned {exc.response.status_code}",
                    details={"body": exc.response.text[:512]},
                    cause=exc,
                )
                continue
            except (LLMTimeoutError, LLMProviderError):
                # Propagate these directly.
                raise
            except httpx.HTTPError as exc:
                last_exc = LLMProviderError("LLM HTTP error", cause=exc)
                continue
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise LLMOutputError("Malformed LLM output", cause=exc) from exc
        if last_exc:
            raise last_exc
        raise LLMProviderError("LLM call failed after retries")


def _extract_text(data: dict[str, Any]) -> str:
    text: str = data["choices"][0]["message"]["content"]
    return text


def _extract_usage(data: dict[str, Any], model: str) -> TokenUsage:
    usage = data.get("usage", {}) or {}
    return TokenUsage.from_pair(
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        model=model,
    )


def _extract_finish(data: dict[str, Any]) -> str | None:
    try:
        finish: str | None = data["choices"][0].get("finish_reason")
        return finish
    except (KeyError, IndexError, TypeError):
        return None


# ============================================================================
# Factory
# ============================================================================


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.llm_provider == "mock":
        return MockLLMProvider()
    if s.llm_provider == "openai":
        return OpenAILLMProvider()
    raise LLMProviderError(f"Unknown LLM provider: {s.llm_provider}")


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "get_llm_provider",
]
