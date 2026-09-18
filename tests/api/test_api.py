"""API tests against the FastAPI app (no external services)."""

from __future__ import annotations

import base64

import pytest
from app.main import create_app, reset_app_orchestrator
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    reset_app_orchestrator()
    app = create_app()
    with TestClient(app) as c:
        yield c


SAMPLE_DOCS = [
    {"kind": "text", "text": "PostgreSQL is a database with pgvector for vector search.", "title": "PG"},
    {"kind": "text", "text": "FastAPI is a Python web framework using Pydantic.", "title": "FastAPI"},
    {"kind": "text", "text": "Hybrid search combines dense and lexical retrieval with alpha fusion.", "title": "Hybrid"},
]


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["providers"]["llm"] == "mock"
        assert body["chunk_count"] == 0


class TestIngestion:
    def test_ingest_single_text(self, client):
        r = client.post("/documents", json={"text": "Hello world", "title": "t"})
        assert r.status_code == 201
        body = r.json()
        assert body["document_id"].startswith("doc-")
        assert len(body["chunk_ids"]) >= 1

    def test_ingest_batch(self, client):
        r = client.post("/documents/batch", json={"items": SAMPLE_DOCS})
        assert r.status_code == 201
        body = r.json()
        assert len(body["results"]) == 3
        for res in body["results"]:
            assert res["error"] is None
            assert res["document_id"].startswith("doc-")

    def test_ingest_bytes_base64(self, client):
        data = base64.b64encode(b"Hello base64").decode()
        r = client.post("/documents/bytes", json={"data": data, "content_type": "text/plain"})
        assert r.status_code == 201
        assert r.json()["error"] is None

    def test_ingest_upload_multipart(self, client, tmp_path):
        p = tmp_path / "u.txt"
        p.write_text("upload content here")
        with open(p, "rb") as f:
            r = client.post("/documents/upload", files={"file": ("u.txt", f, "text/plain")}, data={"title": "u"})
        assert r.status_code == 201
        assert r.json()["error"] is None

    def test_ingest_too_large_rejected(self, client):
        # Use the configured limit (10 MiB by default); we'll exceed by patching max_bytes.
        # Easier: send a much larger body and expect 413.
        big = "x" * (11 * 1024 * 1024)
        r = client.post("/documents", json={"text": big})
        # Either 413 from upload limit or 422 from validation; both acceptable.
        assert r.status_code in {413, 422}


class TestSearch:
    def _seed(self, client):
        client.post("/documents/batch", json={"items": SAMPLE_DOCS})

    def test_search_returns_candidates(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "hybrid search dense lexical", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["query_id"].startswith("q-")
        assert body["latency_ms"] > 0
        assert len(body["candidates"]) > 0
        assert body["candidates"][0]["retrieval_method"] == "hybrid"
        top = body["candidates"][0]
        assert "Hybrid search" in top["content"]

    def test_search_methods_validation(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "x", "methods": ["bogus"]})
        assert r.status_code == 422

    def test_search_empty_methods_validation(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "x", "methods": []})
        assert r.status_code == 422


class TestQuery:
    def _seed(self, client):
        client.post("/documents/batch", json={"items": SAMPLE_DOCS})

    def test_query_returns_grounded_answer(self, client):
        self._seed(client)
        r = client.post("/query", json={"query": "What is hybrid search?", "top_k": 3, "rerank": True})
        assert r.status_code == 200
        ans = r.json()
        assert ans["request_id"]
        assert ans["evidence_sufficient"] is True
        assert len(ans["citations"]) >= 1
        assert ans["token_usage"]["total_tokens"] > 0
        # mock LLM → pricing unknown
        assert ans["cost_estimate"]["pricing_known"] is False
        assert ans["trace"] is None  # we don't expose trace in the response body yet

    def test_query_with_no_evidence(self, client):
        # No docs ingested
        r = client.post("/query", json={"query": "What is hybrid search?"})
        assert r.status_code == 200
        ans = r.json()
        assert ans["evidence_sufficient"] is False
        assert ans["citations"] == []
        assert "insufficient" in ans["answer"].lower()


class TestEvaluate:
    def _seed(self, client):
        client.post("/documents/batch", json={"items": SAMPLE_DOCS})
        # We don't know the chunk ids ahead of time, so let's grab them from search.
        s = client.post("/search", json={"query": "hybrid", "top_k": 5})
        return [c["chunk_id"] for c in s.json()["candidates"]]

    def test_evaluate_endpoint(self, client):
        chunk_ids = self._seed(client)
        cases = [
            {
                "case_id": "c1",
                "query": "What is hybrid search?",
                "relevant_chunk_ids": chunk_ids[:1],
            }
        ]
        r = client.post("/evaluate", json={"cases": cases, "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["n_cases"] == 1
        assert "recall@k" in body["retrieval_metrics_mean"]
        assert body["cases"][0]["error"] is None


class TestMetrics:
    def test_metrics_after_activity(self, client):
        client.post("/documents", json={"text": "hello"})
        client.post("/search", json={"query": "hello"})
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert "counters" in body
        assert body["counters"].get("ingest.documents", 0) == 1.0


class TestGetDocument:
    def test_get_existing(self, client):
        r = client.post("/documents", json={"text": "x", "title": "t"})
        did = r.json()["document_id"]
        r2 = client.get(f"/documents/{did}")
        assert r2.status_code == 200
        assert r2.json()["document_id"] == did

    def test_get_missing_returns_404(self, client):
        r = client.get("/documents/no-such-doc")
        assert r.status_code == 404
        body = r.json()
        assert body["code"] == "NOT_FOUND"


class TestErrorContract:
    def test_validation_error_contract(self, client):
        # Empty query → ValidationError → 422 with code.
        r = client.post("/query", json={"query": ""})
        assert r.status_code == 422
        # FastAPI's pydantic 422 or our domain 422 both fine; check shape.
        body = r.json()
        # Pydantic-style 422 has detail[], ours has code/message.
        assert "detail" in body or "code" in body
