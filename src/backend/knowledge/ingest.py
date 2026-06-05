"""Ingest orchestration: pre-flight, idempotent replace, interleaved embed→upsert.

Pinecone + OpenAI handles are injected so this is unit-testable without network.
Only transient errors are retried; permanent errors surface immediately.
"""

from __future__ import annotations

import hashlib
import re
import time

from knowledge.chunking import Chunk

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DOC_TYPE = "reference_book"
MIN_CHARS_PER_PAGE = 200  # extraction-quality floor (guards scanned/empty PDFs)
_TRANSIENT_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "PineconeApiException",
}
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


def slugify(value: str) -> str:
    """Lowercase, hyphenate, strip non-alphanumerics — ID/prefix-safe source name."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    result = value.strip("-")
    if not result:
        raise ValueError(f"slugify({value!r}) produced an empty slug — pass an explicit --source slug")
    return result


def _is_transient(e: Exception) -> bool:
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    status = getattr(e, "status", None) or getattr(e, "status_code", None)
    if status in _TRANSIENT_STATUS:
        return True
    return type(e).__name__ in _TRANSIENT_NAMES


def preflight(pc, index_name: str, pages_text: str, page_count: int, expected_dim: int = EMBEDDING_DIM) -> None:
    """Fail fast before any embedding spend."""
    desc = pc.describe_index(index_name)
    dim = desc.dimension if hasattr(desc, "dimension") else desc["dimension"]
    if dim != expected_dim:
        raise ValueError(
            f"Index '{index_name}' has dimension {dim}, expected {expected_dim} "
            f"for {EMBEDDING_MODEL}. Use a 1536-d index."
        )
    if page_count > 0 and len(pages_text) / page_count < MIN_CHARS_PER_PAGE:
        raise ValueError(
            f"extraction produced only {len(pages_text)} chars across {page_count} pages "
            f"(< {MIN_CHARS_PER_PAGE}/page). The PDF may be scanned/image-only."
        )


def make_vector_id(source: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256(f"{source}|{chunk_index}|{text}".encode()).hexdigest()
    return f"{source}-{chunk_index}-{h[:16]}"


def build_vector(chunk: Chunk, source: str, book_title: str, embedding: list[float]) -> dict:
    return {
        "id": make_vector_id(source, chunk.chunk_index, chunk.text),
        "values": embedding,
        "metadata": {
            "source": source,
            "book_title": book_title,
            "section_path": chunk.section_path,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "doc_type": DOC_TYPE,
            "text": chunk.text,
        },
    }


def _with_retry(fn, *, attempts: int = 5, base_delay: float = 1.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — re-raise non-transient immediately below
            if not _is_transient(e) or i == attempts - 1:
                raise
            time.sleep(base_delay * (2**i))


def delete_existing_source(index, source: str, namespace: str) -> None:
    """Delete all vectors for a source by ID prefix (serverless-safe).

    WARNING: prefix matching means source='foo' also matches IDs for source='foo-bar'.
    Choose source slugs that are not prefixes of one another.
    """
    if not source:
        raise ValueError("source must be non-empty to compute a safe delete prefix")
    for ids in index.list(prefix=f"{source}-", namespace=namespace):
        if ids:
            _with_retry(lambda ids=ids: index.delete(ids=list(ids), namespace=namespace))


def ingest_chunks(
    chunks: list[Chunk],
    source: str,
    book_title: str,
    namespace: str,
    index,
    openai_client,
    batch_size: int = 100,
) -> int:
    """Embed and upsert chunks one batch at a time (resumable on failure)."""
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = [c.text for c in batch]
        resp = _with_retry(lambda inputs=inputs: openai_client.embeddings.create(model=EMBEDDING_MODEL, input=inputs))
        embeddings = [item.embedding for item in resp.data]
        vectors = [build_vector(c, source, book_title, e) for c, e in zip(batch, embeddings, strict=True)]
        _with_retry(lambda vectors=vectors: index.upsert(vectors=vectors, namespace=namespace))
        total += len(vectors)
        print(f"      ingested {total}/{len(chunks)}")
    return total
