"""Unit tests for the loader + ingestion service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.domain import DocumentParseError, DocumentSource, DocumentTooLargeError
from app.ingestion import DocumentLoader, IngestionService


class TestDocumentLoader:
    def test_load_text_plain(self):
        loader = DocumentLoader()
        doc = loader.load_text("Hello world", content_type="text/plain")
        assert doc.content == "Hello world"
        assert doc.metadata.content_type == "text/plain"

    def test_load_text_strips_whitespace(self):
        loader = DocumentLoader()
        doc = loader.load_text("  Hello\n\n  world  ")
        assert doc.content.startswith("Hello")
        assert doc.content.endswith("world")
        assert "\n\n" in doc.content  # interior blank line preserved

    def test_load_json(self):
        loader = DocumentLoader()
        data = {"a": 1, "b": [1, 2, 3]}
        doc = loader.load_text(json.dumps(data), content_type="application/json")
        assert "a: 1" in doc.content
        assert "b:" in doc.content

    def test_load_html(self):
        loader = DocumentLoader()
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        doc = loader.load_text(html, content_type="text/html")
        assert "Hello" in doc.content
        assert "world" in doc.content
        assert "<" not in doc.content

    def test_load_bytes_utf8(self):
        loader = DocumentLoader()
        doc = loader.load_bytes(b"Hello", content_type="text/plain")
        assert doc.content == "Hello"

    def test_too_large_raises(self):
        loader = DocumentLoader(max_bytes=10)
        with pytest.raises(DocumentTooLargeError):
            loader.load_text("x" * 100)

    def test_file_not_found(self, tmp_path: Path):
        loader = DocumentLoader()
        with pytest.raises(DocumentParseError):
            loader.load_file(tmp_path / "missing.txt")

    def test_file_load_txt(self, tmp_path: Path):
        p = tmp_path / "x.txt"
        p.write_text("file content", encoding="utf-8")
        loader = DocumentLoader()
        doc = loader.load_file(p)
        assert doc.content == "file content"
        assert doc.metadata.source == DocumentSource.file
        assert doc.metadata.title == "x"
        assert doc.metadata.extra["filename"] == "x.txt"

    def test_file_load_json_extension(self, tmp_path: Path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"k": "v"}), encoding="utf-8")
        loader = DocumentLoader()
        doc = loader.load_file(p)
        assert "k: v" in doc.content

    def test_empty_text_rejected(self):
        loader = DocumentLoader()
        with pytest.raises(ValueError):
            loader.load_text("   ")

    def test_parse_failure_raises_structured_error(self):
        loader = DocumentLoader()
        with pytest.raises(DocumentParseError):
            loader.load_bytes(b"\xff\xfe\x00bad", content_type="text/plain")


class TestIngestionService:
    def test_ingest_text_returns_chunks(self):
        svc = IngestionService()
        result = svc.ingest_text("This is a test document. " * 20, title="t")
        assert result.document_id.startswith("doc-")
        assert len(result.chunks) >= 1
        for c in result.chunks:
            assert c.document_id == result.document_id
            assert c.content.strip()
            assert c.id.startswith("chunk-")

    def test_ingest_many_failure_isolation(self):
        svc = IngestionService()
        items = [
            {"kind": "text", "text": "ok"},
            {"kind": "text", "text": "   "},  # invalid
            {"kind": "text", "text": "also ok"},
        ]
        results = svc.ingest_many(items)
        assert len(results) == 3
        assert results[1].warnings  # failure isolated
        assert results[0].chunks and results[2].chunks

    def test_ingest_unknown_kind_raises_value_error(self):
        svc = IngestionService()
        results = svc.ingest_many([{"kind": "bogus"}])
        assert results[0].warnings
