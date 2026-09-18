"""In-process metrics registry.

Thread-safe, dependency-free counters and histograms. Designed for being
scraped by the `/metrics` endpoint or for being embedded directly in response
bodies. No Prometheus client is required to run the system.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable

from app.domain.models import now_ms


class MetricsRegistry:
    """A minimal in-process metrics registry.

    Supports:
      - counters (monotonic)
      - gauges (settable)
      - histograms (latency/distribution) with running mean & count
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "sum": 0.0, "min": float("inf"), "max": float("-inf")}
        )

    # --- counters ----------------------------------------------------------
    def inc(self, name: str, value: float = 1.0) -> None:
        if value < 0:
            raise ValueError("counters are monotonic; use inc(abs(v)) or gauge")
        with self._lock:
            self._counters[name] += value

    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    # --- gauges ------------------------------------------------------------
    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def gauge(self, name: str) -> float | None:
        with self._lock:
            return self._gauges.get(name)

    # --- histograms --------------------------------------------------------
    def observe(self, name: str, value: float) -> None:
        with self._lock:
            h = self._histograms[name]
            h["count"] += 1
            h["sum"] += value
            if value < h["min"]:
                h["min"] = value
            if value > h["max"]:
                h["max"] = value

    def histogram(self, name: str) -> dict[str, float]:
        with self._lock:
            h = dict(self._histograms.get(name, {}))
            if h.get("count"):
                h["mean"] = h["sum"] / h["count"]
            return h

    # --- export ------------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: dict(v) for k, v in self._histograms.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Singleton
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _registry


def reset_registry() -> None:
    """Test helper — wipe all metrics."""
    _registry.reset()


# --- Convenience decorators --------------------------------------------------


def time_it(name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorator that records wall-clock latency in ms under `name`."""

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            start = now_ms()
            try:
                return fn(*args, **kwargs)
            finally:
                _registry.observe(name, now_ms() - start)

        return wrapper

    return decorator


__all__ = ["MetricsRegistry", "get_registry", "reset_registry", "time_it"]
