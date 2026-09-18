"""Embedding provider abstraction.

A provider-neutral interface (`EmbeddingProvider`) plus two implementations:

  - MockEmbeddingProvider: deterministic hash-based vectors for offline use.
  - OpenAIEmbeddingProvider: calls an OpenAI-compatible /v1/embeddings endpoint.

Both implementations accept a list[str] and return a list[list[float]] of the
same length. Batching is handled by the caller (or by `EmbeddingService`).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
from typing import Protocol, runtime_checkable

import httpx

from app.config import Settings, get_settings
from app.domain import EmbeddingError


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-neutral embedding interface."""

    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]: ...


# ============================================================================
# Mock provider (deterministic, offline, dependency-free)
# ============================================================================


class MockEmbeddingProvider:
    """Deterministic hashing-based embeddings for tests and benchmarks.

    Same text always yields the same vector. Tokens that share substrings
    produce nearby vectors (via overlapping hash buckets) so the mock has
    non-trivial retrieval behaviour — useful for offline evaluation runs.
    """

    def __init__(self, model: str = "mock-embedding-v1", dimension: int | None = None) -> None:
        self._model = model
        self._dimension = dimension or get_settings().embedding_dimension

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        # Pure CPU work; yield to the loop so async stays cooperative.
        await asyncio.sleep(0)
        return self.embed_texts(texts)

    # --- internals ---------------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        dim = self._dimension
        vec = [0.0] * dim
        if not text:
            return _normalize(vec)
        # Tokenise on whitespace + character n-grams for substring overlap.
        tokens = text.lower().split()
        for i, tok in enumerate(tokens):
            # Bucket per token, weighted by position to give some structural signal.
            bucket = abs(int(hashlib.sha1(f"{i}::{tok}".encode(), usedforsecurity=False).hexdigest(), 16)) % dim
            vec[bucket] += 1.0 / math.sqrt(i + 1)
            # Plus 3-char n-gram buckets to encourage substring similarity.
            for j in range(0, max(0, len(tok) - 2)):
                ng = tok[j : j + 3]
                b = abs(int(hashlib.blake2b(ng.encode(), digest_size=4).hexdigest(), 16)) % dim
                vec[b] += 0.3
        return _normalize(vec)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


# ============================================================================
# OpenAI-compatible provider
# ============================================================================


class OpenAIEmbeddingProvider:
    """Calls an OpenAI-compatible `/embeddings` endpoint.

    This implementation deliberately uses httpx and not the `openai` SDK so
    that any compatible server (LM Studio, vLLM, Ollama with OpenAI shim, …)
    can be plugged in by changing only `EMBEDDING_BASE_URL`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
    ) -> None:
        s = get_settings()
        self._api_key = api_key if api_key is not None else s.embedding_api_key
        self._base_url = (base_url or s.embedding_base_url).rstrip("/")
        self._model = model or s.embedding_model
        self._dimension = dimension or s.embedding_dimension
        self._batch_size = batch_size or s.embedding_batch_size
        self._timeout = timeout or s.embedding_timeout_seconds
        if not self._api_key:
            raise EmbeddingError(
                "EMBEDDING_API_KEY is not set; cannot use OpenAIEmbeddingProvider.",
                details={"base_url": self._base_url, "model": self._model},
            )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.embed_texts_async(texts))

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                out.extend(await self._embed_batch(client, batch))
        return out

    async def _embed_batch(self, client: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self._model, "input": batch}
        try:
            resp = await client.post(f"{self._base_url}/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise EmbeddingError("Embedding request timed out", cause=exc) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Embedding API returned {exc.response.status_code}",
                details={"body": exc.response.text[:512]},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError("Embedding HTTP error", cause=exc) from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise EmbeddingError("Embedding response was not JSON", cause=exc) from exc

        # OpenAI: {"data": [{"embedding": [...], "index": 0}, ...]}
        try:
            items = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("Malformed embedding response", cause=exc) from exc


# ============================================================================
# Factory
# ============================================================================


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider.

    Defaults to the mock provider so the system runs offline out of the box.
    """
    s = settings or get_settings()
    if s.embedding_provider == "mock":
        return MockEmbeddingProvider(dimension=s.embedding_dimension)
    if s.embedding_provider == "openai":
        return OpenAIEmbeddingProvider()
    raise EmbeddingError(f"Unknown embedding provider: {s.embedding_provider}")


# Helper for serialization to a binary blob used by storage layers.
def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(data: bytes, dim: int) -> list[float]:
    if len(data) != dim * 4:
        raise ValueError(f"packed vector has wrong length: {len(data)} != {dim*4}")
    return list(struct.unpack(f"{dim}f", data))


__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
    "pack_vector",
    "unpack_vector",
]
