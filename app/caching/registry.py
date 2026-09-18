"""Cache layer.

A small, dependency-free in-process cache for deterministic operations
(embeddings, retrieval, reranking, context prep). Keys MUST include all
parameters that materially affect the result — the `CacheKey` helper encodes
this contract explicitly.

The cache is enabled by `CACHE_ENABLED=true` and is capped at
`CACHE_MAX_ENTRIES` (LRU eviction).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

from app.config import get_settings
from app.observability import get_logger

T = TypeVar("T")


_logger = get_logger("app.caching")


def stable_key(*parts: Any) -> str:
    """Stable, deterministic hash key from positional + named parts.

    Each part is JSON-serialised (with `sort_keys=True`). Non-serialisable
    objects fall back to `repr(part)` so the key is still deterministic.
    """
    serialised: list[str] = []
    for p in parts:
        try:
            serialised.append(json.dumps(p, sort_keys=True, default=str))
        except (TypeError, ValueError):
            serialised.append(repr(p))
    payload = "|".join(serialised).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LRUCache[T]:
    """Small in-process LRU. Thread-safe."""

    def __init__(self, max_entries: int = 2048) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._max = max_entries
        self._data: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        with self._lock:
            v = self._data.get(key)
            if v is None:
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return v

    def put(self, key: str, value: T) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._max:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._data)}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0


class CacheRegistry:
    """Holds all named caches in the system + a global on/off switch."""

    def __init__(self) -> None:
        s = get_settings()
        self.enabled = s.cache_enabled
        self._caches: dict[str, LRUCache[Any]] = {}
        self._max = s.cache_max_entries

    def cache_for(self, name: str) -> LRUCache[Any]:
        if name not in self._caches:
            self._caches[name] = LRUCache(max_entries=self._max)
        return self._caches[name]

    def cached_call(
        self,
        name: str,
        key_parts: tuple[Any, ...],
        factory: Callable[[], T],
    ) -> T:
        """Return the cached result for `key_parts`, or `factory()` and store it."""
        if not self.enabled:
            return factory()
        key = stable_key(name, *key_parts)
        cache = self.cache_for(name)
        hit = cache.get(key)
        if hit is not None:
            return hit  # type: ignore[no-any-return]
        val = factory()
        cache.put(key, val)
        return val

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {name: c.stats() for name, c in self._caches.items()}

    def clear(self) -> None:
        for c in self._caches.values():
            c.clear()


# Singleton
_registry: CacheRegistry | None = None
_registry_lock = threading.RLock()


def get_cache_registry() -> CacheRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = CacheRegistry()
        return _registry


def reset_cache_registry() -> None:
    """Test helper — wipe all caches and recreate the registry."""
    global _registry
    with _registry_lock:
        _registry = None


__all__ = [
    "CacheRegistry",
    "LRUCache",
    "get_cache_registry",
    "reset_cache_registry",
    "stable_key",
]
