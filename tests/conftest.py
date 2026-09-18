"""Shared pytest fixtures for the Production RAG test suite.

These fixtures keep the test suite hermetic: every test starts with a fresh
in-memory store, a fresh lexical index, mock providers, a clean cache, and
clean metrics. No external services are touched unless a test is explicitly
marked `@pytest.mark.integration`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path even when pytest is invoked from a subdir.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force offline operation for tests. We deliberately *override* any ambient
# .env file the user may have in a parent directory — the test suite must be
# hermetic regardless of where pytest is invoked from.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["EMBEDDING_DIMENSION"] = "64"
os.environ["RERANKER_PROVIDER"] = "mock"
os.environ["CACHE_ENABLED"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["LOG_JSON"] = "true"
# Ensure the test process doesn't read an unrelated .env from a parent dir.
os.environ["FORCE_IN_MEMORY_STORE"] = "true"

import pytest  # noqa: E402
from app.caching import reset_cache_registry  # noqa: E402
from app.config import reset_settings_cache  # noqa: E402
from app.embeddings import MockEmbeddingProvider  # noqa: E402
from app.observability import reset_registry  # noqa: E402
from app.storage import InMemoryVectorStore, LexicalIndex  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all process-wide state between tests for hermeticity."""
    reset_settings_cache()
    reset_cache_registry()
    reset_registry()
    yield
    reset_settings_cache()
    reset_cache_registry()
    reset_registry()


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(dimension=64)


@pytest.fixture
def lexical_index() -> LexicalIndex:
    return LexicalIndex()


@pytest.fixture
def embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=64)
