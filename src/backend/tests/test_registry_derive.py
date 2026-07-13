"""Mock-based tests for registry.derive_service.derive_for_collaborator.

A small store-backed fake Supabase client: db.table(name) returns a chainable
query over an in-memory dict keyed by table name. Supports select/insert with
.eq() filtering and .maybe_single(), plus db.storage.from_(bucket).download(path).
The real PDF parser is patched out — no real parsing happens.
"""

import asyncio
from copy import deepcopy
from unittest.mock import patch

from registry import derive_service


class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    """Chainable fake query over a single table's list of row dicts."""

    def __init__(self, store, table_name):
        self._store = store
        self._table = table_name
        self._filters = []
        self._op = "select"
        self._insert_payload = None
        self._single = False

    def select(self, *_args, **_kwargs):
        self._op = "select"
        return self

    def insert(self, payload):
        self._op = "insert"
        self._insert_payload = payload
        return self

    def eq(self, col, value):
        self._filters.append((col, value))
        return self

    def maybe_single(self):
        self._single = True
        return self

    def _matches(self, row):
        return all(row.get(col) == value for col, value in self._filters)

    def execute(self):
        rows = self._store.setdefault(self._table, [])
        if self._op == "insert":
            payload = self._insert_payload
            new_rows = payload if isinstance(payload, list) else [payload]
            for r in new_rows:
                rows.append(deepcopy(r))
            return _Result([deepcopy(r) for r in new_rows])
        # select
        matched = [deepcopy(r) for r in rows if self._matches(r)]
        if self._single:
            return _Result(matched[0] if matched else None)
        return _Result(matched)


class _Storage:
    def __init__(self, downloads):
        self._downloads = downloads  # path -> bytes (or raises if missing)

    def from_(self, _bucket):
        return self

    def download(self, path):
        if path not in self._downloads:
            raise FileNotFoundError(path)
        return self._downloads[path]


class _FakeDB:
    def __init__(self, store, downloads=None):
        self._store = store
        self.storage = _Storage(downloads or {})

    def table(self, name):
        return _Query(self._store, name)


def _make_db(downloads=None, **tables):
    return _FakeDB({name: list(rows) for name, rows in tables.items()}, downloads=downloads)


# ---------------------------------------------------------------------------
# Name matched, has percentage -> high confidence
# ---------------------------------------------------------------------------


def test_name_matched_with_pct_high_confidence():
    db = _make_db(
        downloads={"contracts/a.pdf": b"%PDF-bytes"},
        work_files=[{"work_id": "w1", "file_id": "f1"}],
        project_files=[{"id": "f1", "file_path": "contracts/a.pdf", "content_hash": "h1", "file_name": "a.pdf"}],
        contract_parse_cache=[],
    )
    parsed = {
        "parties": [
            {
                "name": "Marcus",
                "role": "producer",
                "master_pct": 30,
                "publishing_pct": 0,
                "soundexchange_pct": 15,
            },
            {"name": "Someone Else", "role": "writer", "master_pct": 10, "publishing_pct": 50},
        ],
        "main_artist_found": False,
    }
    with patch.object(derive_service, "_parse_pdf_bytes", return_value=parsed) as mock_parse:
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert mock_parse.call_count == 1  # cache miss -> parsed once
    assert result["found"] is True
    assert result["confidence"] == "high"
    assert result["master_pct"] == 30
    assert result["publishing_pct"] == 0
    assert result["soundexchange_pct"] == 15
    assert result["matched_file_ids"] == ["f1"]
    assert result["terms"] == []
    # Result was cached for the content_hash.
    assert any(r["content_hash"] == "h1" for r in db._store["contract_parse_cache"])


# ---------------------------------------------------------------------------
# Name not matched -> found False, low confidence, no matches
# ---------------------------------------------------------------------------


def test_name_not_matched_returns_not_found():
    db = _make_db(
        downloads={"contracts/a.pdf": b"%PDF-bytes"},
        work_files=[{"work_id": "w1", "file_id": "f1"}],
        project_files=[{"id": "f1", "file_path": "contracts/a.pdf", "content_hash": "h1", "file_name": "a.pdf"}],
        contract_parse_cache=[],
    )
    parsed = {
        "parties": [{"name": "Jane Doe", "role": "writer", "master_pct": 50, "publishing_pct": 50}],
        "main_artist_found": False,
    }
    with patch.object(derive_service, "_parse_pdf_bytes", return_value=parsed):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert result["found"] is False
    assert result["confidence"] == "low"
    assert result["master_pct"] is None
    assert result["publishing_pct"] is None
    assert result["soundexchange_pct"] is None
    assert result["matched_file_ids"] == []
    assert result["terms"] == []


# ---------------------------------------------------------------------------
# Cache hit -> parser NOT called, still derives from cached parties
# ---------------------------------------------------------------------------


def test_cache_hit_skips_parser():
    cached_parsed = {
        "parties": [{"name": "Marcus", "role": "producer", "master_pct": 25, "publishing_pct": 0}],
        "main_artist_found": False,
    }
    db = _make_db(
        # No download configured: if the code tried to download/parse, it would fail/return None.
        work_files=[{"work_id": "w1", "file_id": "f1"}],
        project_files=[{"id": "f1", "file_path": "contracts/a.pdf", "content_hash": "h1", "file_name": "a.pdf"}],
        contract_parse_cache=[{"content_hash": "h1", "parsed": cached_parsed}],
    )
    with patch.object(derive_service, "_parse_pdf_bytes") as mock_parse:
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert mock_parse.call_count == 0  # cache hit -> never parsed
    assert result["found"] is True
    assert result["confidence"] == "high"
    assert result["master_pct"] == 25
    assert result["matched_file_ids"] == ["f1"]


# ---------------------------------------------------------------------------
# Matched party but no percentage -> confidence "low"
# ---------------------------------------------------------------------------


def test_matched_no_pct_low_confidence():
    db = _make_db(
        downloads={"contracts/a.pdf": b"%PDF-bytes"},
        work_files=[{"work_id": "w1", "file_id": "f1"}],
        project_files=[{"id": "f1", "file_path": "contracts/a.pdf", "content_hash": "h1", "file_name": "a.pdf"}],
        contract_parse_cache=[],
    )
    parsed = {
        "parties": [{"name": "Marcus", "role": "producer", "master_pct": 0, "publishing_pct": None}],
        "main_artist_found": False,
    }
    with patch.object(derive_service, "_parse_pdf_bytes", return_value=parsed):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert result["found"] is True
    assert result["confidence"] == "low"
    assert result["master_pct"] == 0
    assert result["publishing_pct"] is None
    assert result["matched_file_ids"] == ["f1"]


# ---------------------------------------------------------------------------
# Alias matching
# ---------------------------------------------------------------------------


def test_collaborator_matched_via_alias():
    """A party listed by legal name with the collaborator's stage name in
    `aliases` still matches."""
    db = _make_db(
        downloads={"contracts/a.pdf": b"%PDF-bytes"},
        work_files=[{"work_id": "w1", "file_id": "f1"}],
        project_files=[{"id": "f1", "file_path": "contracts/a.pdf", "content_hash": "h1", "file_name": "a.pdf"}],
        contract_parse_cache=[],
    )
    parsed = {
        "parties": [
            {"name": "Marcus Adebayo", "role": "producer", "aliases": ["M-Bay"], "master_pct": 30, "publishing_pct": 0}
        ],
        "main_artist_found": False,
    }
    with patch.object(derive_service, "_parse_pdf_bytes", return_value=parsed):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "M-Bay"))

    assert result["found"] is True
    assert result["confidence"] == "high"
    assert result["master_pct"] == 30
    assert result["matched_file_ids"] == ["f1"]


def test_stale_cache_payload_without_aliases_still_matches_by_name():
    """Cached parses from before the aliases field behave exactly as before."""
    cached_parsed = {
        "parties": [{"name": "Marcus", "role": "producer", "master_pct": 25, "publishing_pct": 0}],
        "main_artist_found": False,
    }
    db = _make_db(
        work_files=[{"work_id": "w1", "file_id": "f1"}],
        project_files=[{"id": "f1", "file_path": "contracts/a.pdf", "content_hash": "h1", "file_name": "a.pdf"}],
        contract_parse_cache=[{"content_hash": "h1", "parsed": cached_parsed}],
    )
    with patch.object(derive_service, "_parse_pdf_bytes") as mock_parse:
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert mock_parse.call_count == 0
    assert result["found"] is True
    assert result["master_pct"] == 25
    # Cached payload predates the soundexchange bucket — tolerated as None.
    assert result["soundexchange_pct"] is None
