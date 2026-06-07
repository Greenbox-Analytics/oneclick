"""Tests for the book ingest orchestration."""

from unittest.mock import MagicMock, patch

import pytest

from knowledge.chunking import Chunk
from knowledge.ingest import (
    build_vector,
    delete_existing_source,
    ingest_chunks,
    preflight,
    slugify,
)


def _chunk(i):
    return Chunk(
        text=f"chunk {i}",
        section_path="Part V ▸ Record Deals",
        page_start=i + 1,
        page_end=i + 1,
        chunk_index=i,
        token_count=5,
    )


def test_slugify_cleans_messy_stem():
    assert slugify("All You Need to Know (Donald S. Passman)") == "all-you-need-to-know-donald-s-passman"


def test_preflight_rejects_wrong_dimension():
    pc = MagicMock()
    pc.describe_index.return_value = MagicMock(dimension=3072)
    with pytest.raises(ValueError, match="dimension"):
        preflight(pc, "idx", pages_text="a" * 1000, page_count=1)


def test_preflight_rejects_empty_extraction():
    pc = MagicMock()
    pc.describe_index.return_value = MagicMock(dimension=1536)
    with pytest.raises(ValueError, match="extract"):
        preflight(pc, "idx", pages_text="x", page_count=100)


def test_build_vector_has_full_metadata():
    vec = build_vector(_chunk(0), source="passman", book_title="The Music Business (Passman)", embedding=[0.1] * 1536)
    assert vec["id"].startswith("passman-0-")
    md = vec["metadata"]
    assert md["source"] == "passman"
    assert md["book_title"] == "The Music Business (Passman)"
    assert md["section_path"] == "Part V ▸ Record Deals"
    assert md["page_start"] == 1 and md["page_end"] == 1
    assert md["chunk_index"] == 0
    assert md["token_count"] == 5
    assert md["doc_type"] == "reference_book"
    assert md["text"] == "chunk 0"


def test_delete_existing_source_uses_prefix_list():
    index = MagicMock()
    index.list.return_value = iter([["passman-0-aaa", "passman-1-bbb"]])
    delete_existing_source(index, source="passman", namespace="ns")
    index.delete.assert_called_once_with(ids=["passman-0-aaa", "passman-1-bbb"], namespace="ns")


def test_ingest_interleaves_embed_then_upsert():
    chunks = [_chunk(i) for i in range(3)]
    index = MagicMock()
    openai_client = MagicMock()
    calls = []

    def _embed(**kwargs):
        n = len(kwargs["input"])
        calls.append(("embed", n))
        return MagicMock(data=[MagicMock(embedding=[0.0] * 1536) for _ in range(n)])

    def _upsert(**kwargs):
        calls.append(("upsert", len(kwargs["vectors"])))

    openai_client.embeddings.create.side_effect = _embed
    index.upsert.side_effect = _upsert

    ingest_chunks(
        chunks, source="passman", book_title="X", namespace="ns", index=index, openai_client=openai_client, batch_size=2
    )
    assert calls == [("embed", 2), ("upsert", 2), ("embed", 1), ("upsert", 1)]


@patch("knowledge.ingest.time.sleep", return_value=None)
def test_ingest_retries_transient_error(_sleep):
    chunks = [_chunk(0)]
    index = MagicMock()
    openai_client = MagicMock()
    seq = [TimeoutError("timeout"), MagicMock(data=[MagicMock(embedding=[0.0] * 1536)])]

    def _embed(**kwargs):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    openai_client.embeddings.create.side_effect = _embed
    ingest_chunks(
        chunks, source="s", book_title="X", namespace="ns", index=index, openai_client=openai_client, batch_size=1
    )
    assert index.upsert.call_count == 1
    assert openai_client.embeddings.create.call_count == 2  # failed once, retried, succeeded
    _sleep.assert_called_once()  # backoff fired


@patch("knowledge.ingest.time.sleep", return_value=None)
def test_ingest_does_not_retry_permanent_error(_sleep):
    chunks = [_chunk(0)]
    index = MagicMock()
    openai_client = MagicMock()
    openai_client.embeddings.create.side_effect = ValueError("bad request")
    with pytest.raises(ValueError):
        ingest_chunks(
            chunks, source="s", book_title="X", namespace="ns", index=index, openai_client=openai_client, batch_size=1
        )
    assert openai_client.embeddings.create.call_count == 1  # not retried


def test_slugify_raises_on_empty_result():
    with pytest.raises(ValueError):
        slugify("!!!")
