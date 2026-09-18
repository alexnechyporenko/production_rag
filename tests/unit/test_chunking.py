"""Unit tests for the chunking layer."""

from __future__ import annotations

import pytest
from app.chunking import FixedCharChunker, RecursiveChunker, SentenceChunker, get_chunker
from app.domain import Document, DocumentMetadata, deterministic_chunk_id


def _doc(text: str) -> Document:
    return Document(content=text, metadata=DocumentMetadata(title="t"))


class TestFixedCharChunker:
    def test_basic_chunking(self):
        ch = FixedCharChunker(chunk_size=10, overlap=0)
        chunks = ch.chunk(_doc("0123456789" * 3))
        assert len(chunks) == 3
        assert [c.sequence for c in chunks] == [0, 1, 2]
        assert chunks[0].content == "0123456789"
        assert chunks[1].char_start == 10
        assert chunks[2].char_end == 30

    def test_overlap(self):
        ch = FixedCharChunker(chunk_size=10, overlap=4)
        chunks = ch.chunk(_doc("0123456789" * 3))
        # With overlap 4 and step 6, we get 5 chunks covering positions 0,6,12,18,24
        assert len(chunks) == 5
        # First and second chunks share 4 characters.
        assert chunks[1].content[:4] == chunks[0].content[-4:]

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            FixedCharChunker(chunk_size=0)
        with pytest.raises(ValueError):
            FixedCharChunker(chunk_size=10, overlap=10)
        with pytest.raises(ValueError):
            FixedCharChunker(chunk_size=10, overlap=-1)


class TestSentenceChunker:
    def test_groups_sentences(self):
        text = "This is one. This is two. This is three. This is four."
        ch = SentenceChunker(target_chars=30, overlap_sentences=0)
        chunks = ch.chunk(_doc(text))
        assert len(chunks) >= 1
        for c in chunks:
            assert c.content.strip()

    def test_handles_single_oversized_sentence(self):
        text = "A very long sentence with no periods at all"
        ch = SentenceChunker(target_chars=5, overlap_sentences=0)
        chunks = ch.chunk(_doc(text))
        assert len(chunks) == 1
        assert chunks[0].content == text


class TestRecursiveChunker:
    def test_splits_on_paragraph_breaks(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        ch = RecursiveChunker(chunk_size=20, overlap=0)
        chunks = ch.chunk(_doc(text))
        assert len(chunks) >= 1
        # Each piece should be <= chunk_size or close to it.
        for c in chunks:
            assert len(c.content) <= 40  # some slack from joining

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=0)
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=10, overlap=10)


class TestDeterministicChunkIds:
    def test_same_input_same_id(self):
        doc = _doc("hello world")
        ch = RecursiveChunker(chunk_size=5, overlap=0)
        chunks1 = ch.chunk(doc)
        chunks2 = ch.chunk(doc)
        assert [c.id for c in chunks1] == [c.id for c in chunks2]

    def test_different_sequence_different_id(self):
        i1 = deterministic_chunk_id("doc1", 0, "content")
        i2 = deterministic_chunk_id("doc1", 1, "content")
        assert i1 != i2

    def test_different_content_different_id(self):
        i1 = deterministic_chunk_id("doc1", 0, "content A")
        i2 = deterministic_chunk_id("doc1", 0, "content B")
        assert i1 != i2


class TestFactory:
    def test_get_chunker_fixed(self):
        assert isinstance(get_chunker("fixed", chunk_size=10, overlap=0), FixedCharChunker)

    def test_get_chunker_recursive(self):
        assert isinstance(get_chunker("recursive", chunk_size=10, overlap=0), RecursiveChunker)

    def test_get_chunker_sentence(self):
        assert isinstance(get_chunker("sentence", target_chars=10), SentenceChunker)

    def test_unknown_chunker_raises(self):
        with pytest.raises(ValueError):
            get_chunker("nonexistent")
