"""
Unit tests for the pure helpers in index_workspace.

These deliberately avoid the heavy, optional dependencies
(`sentence_transformers`, `sqlite_vec`): importing `index_workspace` only pulls
in the standard library, because those packages are imported lazily inside the
functions that actually need them.
"""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import index_workspace as m  # noqa: E402


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert m.chunk_text("hello world", size=500) == ["hello world"]

    def test_strips_surrounding_whitespace(self):
        assert m.chunk_text("  hello world  ", size=500) == ["hello world"]

    def test_empty_text_yields_no_chunks(self):
        assert m.chunk_text("   ", size=10) == []

    def test_splits_on_word_boundary(self):
        # "aaaa bbbb cccc" with size 6 splits at spaces, never mid-word.
        chunks = m.chunk_text("aaaa bbbb cccc", size=6)
        assert chunks == ["aaaa", "bbbb", "cccc"]
        for c in chunks:
            assert " " not in c.strip() or len(c) <= 6

    def test_no_space_falls_back_to_hard_split(self):
        # A single long token with no spaces must still be chunked by size.
        chunks = m.chunk_text("abcdefghij", size=4)
        assert chunks == ["abcd", "efgh", "ij"]

    def test_reassembly_preserves_all_words(self):
        text = "the quick brown fox jumps over the lazy dog"
        chunks = m.chunk_text(text, size=12)
        assert " ".join(chunks).split() == text.split()

    def test_every_chunk_within_size_when_spaced(self):
        text = " ".join(["word"] * 50)
        size = 20
        for c in m.chunk_text(text, size=size):
            assert len(c) <= size


class TestVectorSerialization:
    def test_roundtrip_preserves_values(self):
        vec = [0.0, 1.5, -2.25, 100.0]
        data = m.serialize_vector(vec)
        assert m.deserialize_vector(data) == pytest.approx(vec)

    def test_serialized_length_is_four_bytes_per_float(self):
        vec = [1.0, 2.0, 3.0]
        assert len(m.serialize_vector(vec)) == 4 * len(vec)

    def test_serialize_uses_little_endian_float32(self):
        # Mirrors struct's native packing used by serialize_vector.
        vec = [1.0, 2.0]
        assert m.serialize_vector(vec) == struct.pack("2f", *vec)

    def test_empty_vector_roundtrips(self):
        assert m.deserialize_vector(m.serialize_vector([])) == []


class TestConfigConstants:
    def test_embedding_dim_matches_minilm(self):
        assert m.EMBEDDING_DIM == 384

    def test_indexed_extensions_are_lowercase_with_dot(self):
        for ext in m.EXTENSIONS:
            assert ext.startswith(".")
            assert ext == ext.lower()
