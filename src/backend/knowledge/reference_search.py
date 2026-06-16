"""Shared, relevance-gated retrieval over the music-business reference namespace.

Both Zoe and (later) OneClick call `search_reference`. The book never overrides a
contract — callers label its output as background context only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from utils.ingestion.embeddings import create_query_embedding

# The single source of truth for which namespace holds the reference knowledge.
# Set REFERENCE_NAMESPACE in the backend .env to point Zoe/OneClick at a different
# namespace; the upload CLI defaults to the same var so the read/write sides stay in sync.
REFERENCE_NAMESPACE = os.getenv("REFERENCE_NAMESPACE", "music-business-reference")


@dataclass
class ReferencePassage:
    text: str
    section_path: str
    page_start: int
    page_end: int
    book_title: str
    score: float


_index = None


def _get_index():
    """Lazily build a Pinecone index handle from env (mirrors the upload script)."""
    global _index
    if _index is None:
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    return _index


def _normalize_matches(resp) -> list[tuple[float, dict]]:
    matches = resp["matches"] if isinstance(resp, dict) else resp.matches
    out: list[tuple[float, dict]] = []
    for m in matches:
        score = m["score"] if isinstance(m, dict) else m.score
        md = m["metadata"] if isinstance(m, dict) else m.metadata
        out.append((float(score), dict(md)))
    return out


def _to_passage(score: float, md: dict) -> ReferencePassage:
    return ReferencePassage(
        text=md.get("text", ""),
        section_path=md.get("section_path", ""),
        page_start=int(md.get("page_start", 0)),
        page_end=int(md.get("page_end", 0)),
        book_title=md.get("book_title", ""),
        score=score,
    )


def search_reference(
    query: str,
    top_k: int = 10,
    min_score: float = 0.45,  # tuned from live probe: relevant ~0.52-0.70, off-topic <=0.18
    floor_count: int = 0,
    floor_min: float = 0.2,
    namespace: str = REFERENCE_NAMESPACE,
    index=None,
) -> list[ReferencePassage]:
    """Retrieve reference passages with the gating policy.

    Keep every candidate >= `min_score`. Additionally keep up to `floor_count` of the
    highest-ranked candidates below `min_score` but >= `floor_min` (the "always feed
    the book's best, never feed nonsense" floor). Ordered by score, descending.
    """
    idx = index if index is not None else _get_index()
    vector = create_query_embedding(query)
    resp = idx.query(vector=vector, top_k=top_k, namespace=namespace, include_metadata=True)

    candidates = _normalize_matches(resp)
    candidates.sort(key=lambda c: c[0], reverse=True)

    selected: list[ReferencePassage] = []
    for rank, (score, md) in enumerate(candidates):
        if score >= min_score or (rank < floor_count and score >= floor_min):
            selected.append(_to_passage(score, md))
    return selected
