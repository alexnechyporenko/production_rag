"""Integration tests — explicitly marked, opt-in.

These tests require external infrastructure (a real PostgreSQL with pgvector,
or a real OpenAI API key). They are skipped by default and only run when the
appropriate environment variables are present.

Run with: `pytest -m integration`
"""

from __future__ import annotations

import os
import uuid

import pytest

pg_url = os.environ.get("RAG_INTEGRATION_PG_URL")
openai_key = os.environ.get("RAG_INTEGRATION_OPENAI_KEY")


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not pg_url, reason="set RAG_INTEGRATION_PG_URL to run pgvector integration tests")
class TestPgVectorStore:
    def test_round_trip(self):
        from app.domain import DocumentChunk
        from app.storage.pgvector_store import PgVectorStore

        store = PgVectorStore(dsn=pg_url, dimension=8)
        store.reset()
        try:
            cid = f"chunk-it-{uuid.uuid4().hex[:8]}"
            doc_id = f"doc-it-{uuid.uuid4().hex[:8]}"
            chunk = DocumentChunk(
                id=cid, document_id=doc_id, content="hello world",
                sequence=0, char_start=0, char_end=11,
            )
            n = store.add([chunk], [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], model="test")
            assert n == 1
            got = store.get(cid)
            assert got is not None
            assert got.content == "hello world"
            results = store.search([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], top_k=5)
            assert any(c.chunk_id == cid for c in results)
        finally:
            store.reset()


@pytest.mark.skipif(
    not openai_key, reason="set RAG_INTEGRATION_OPENAI_KEY to run real LLM tests"
)
class TestRealLLM:
    def test_openai_completion(self):
        from app.generation.provider import OpenAILLMProvider

        prov = OpenAILLMProvider(api_key=openai_key, model="gpt-4o-mini")
        resp = prov.complete("You are a test assistant.", "Say 'ok'.")
        assert resp.text.strip()
        assert resp.token_usage.total_tokens > 0
