"""Tests for config + observability + caching."""

from __future__ import annotations

import pytest
from app.caching import LRUCache, get_cache_registry, reset_cache_registry, stable_key
from app.config import Settings, get_settings, reset_settings_cache
from app.observability import (
    MetricsRegistry,
    configure_logging,
    get_logger,
    get_registry,
)


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.llm_provider == "mock"
        assert s.embedding_provider == "mock"
        assert s.top_k == 10
        assert s.hybrid_alpha == 0.5

    def test_get_settings_caches(self):
        reset_settings_cache()
        a = get_settings()
        b = get_settings()
        assert a is b
        reset_settings_cache()
        c = get_settings()
        assert c is not a

    def test_cors_origin_list_star(self):
        s = Settings(cors_origins="*")
        assert s.cors_origin_list == ["*"]

    def test_cors_origin_list_explicit(self):
        s = Settings(cors_origins="http://a, http://b")
        assert s.cors_origin_list == ["http://a", "http://b"]

    def test_llm_price_known(self):
        s = Settings()
        assert s.llm_input_price_per_1k("gpt-4o-mini") == s.price_llm_input_gpt_4o_mini
        assert s.llm_input_price_per_1k("unknown-model") is None


class TestMetricsRegistry:
    def test_counter(self):
        r = MetricsRegistry()
        r.inc("x")
        r.inc("x", 2)
        assert r.counter("x") == 3
        with pytest.raises(ValueError):
            r.inc("x", -1)

    def test_gauge(self):
        r = MetricsRegistry()
        r.set_gauge("g", 42.0)
        assert r.gauge("g") == 42.0
        assert r.gauge("missing") is None

    def test_histogram(self):
        r = MetricsRegistry()
        for v in (1, 2, 3):
            r.observe("h", v)
        h = r.histogram("h")
        assert h["count"] == 3
        assert h["sum"] == 6
        assert h["min"] == 1
        assert h["max"] == 3
        assert abs(h["mean"] - 2.0) < 1e-9

    def test_snapshot(self):
        r = get_registry()
        r.inc("snap_counter")
        snap = r.snapshot()
        assert "snap_counter" in snap["counters"]


class TestLogging:
    def test_configure_idempotent(self):
        configure_logging(Settings(log_level="INFO"))
        configure_logging(Settings(log_level="WARNING"))
        log = get_logger("test")
        log.info("hello", x=1)  # should not raise


class TestLRUCache:
    def test_put_get(self):
        c = LRUCache(max_entries=2)
        c.put("a", 1)
        c.put("b", 2)
        assert c.get("a") == 1
        assert c.get("b") == 2

    def test_lru_evicts_oldest(self):
        c = LRUCache(max_entries=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")  # touch a
        c.put("c", 3)  # should evict b, not a
        assert c.get("a") == 1
        assert c.get("b") is None
        assert c.get("c") == 3

    def test_stats(self):
        c = LRUCache(max_entries=2)
        c.put("a", 1)
        c.get("a")
        c.get("missing")
        stats = c.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_clear(self):
        c = LRUCache(max_entries=2)
        c.put("a", 1)
        c.clear()
        assert c.stats()["size"] == 0


class TestStableKey:
    def test_deterministic(self):
        assert stable_key("a", "b", 1) == stable_key("a", "b", 1)

    def test_different_inputs_differ(self):
        assert stable_key("a") != stable_key("b")

    def test_order_matters(self):
        assert stable_key("a", "b") != stable_key("b", "a")

    def test_dict_order_invariant(self):
        # Dictionaries are sorted before hashing → order-independent.
        assert stable_key({"a": 1, "b": 2}) == stable_key({"b": 2, "a": 1})


class TestCacheRegistry:
    def test_cached_call_first_miss(self):
        reset_cache_registry()
        # Force enable for this test.
        from app.config import get_settings

        # Use a fresh settings with cache enabled.
        get_settings().cache_enabled = True
        cr = get_cache_registry()
        cr.enabled = True
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return "value"

        v1 = cr.cached_call("test", ("k1",), factory)
        v2 = cr.cached_call("test", ("k1",), factory)
        assert v1 == v2 == "value"
        assert calls["n"] == 1

    def test_disabled_skips_cache(self):
        reset_cache_registry()
        cr = get_cache_registry()
        cr.enabled = False
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return "x"

        cr.cached_call("test", ("k",), factory)
        cr.cached_call("test", ("k",), factory)
        assert calls["n"] == 2

    def test_snapshot(self):
        reset_cache_registry()
        cr = get_cache_registry()
        cr.enabled = True
        cr.cached_call("a", ("k",), lambda: 1)
        snap = cr.snapshot()
        assert "a" in snap
