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
from utils.contract_parsing.cache import deserialize_contract_data
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare


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


def _cd(rows):
    """rows: list of (name, role, master_pct, publishing_pct). Builds a ContractData whose
    pivot reproduces those percentages (Master -> master bucket, Publishing -> publishing)."""
    parties, shares = [], []
    for name, role, master, publishing in rows:
        parties.append(Party(name=name, role=role))
        if master:
            shares.append(RoyaltyShare(party_name=name, royalty_type="Master", percentage=float(master)))
        if publishing:
            shares.append(RoyaltyShare(party_name=name, royalty_type="Publishing", percentage=float(publishing)))
    return ContractData(parties=parties, works=[], royalty_shares=shares, contract_summary="", default_basis=None)


def _db():
    return _make_db(
        work_files=[{"work_id": "w1", "file_id": "f1"}],
        project_files=[{"id": "f1", "file_path": "contracts/a.pdf", "content_hash": "h1", "file_name": "a.pdf"}],
    )


def test_name_matched_with_pct_high_confidence():
    db = _db()
    cd = _cd([("Marcus", "producer", 30, 0), ("Someone Else", "writer", 10, 50)])
    cd.royalty_shares.append(RoyaltyShare(party_name="Marcus", royalty_type="SoundExchange", percentage=15.0))
    with patch.object(derive_service, "get_or_parse", return_value=cd) as mock_gop:
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert mock_gop.call_count == 1
    assert result["found"] is True
    assert result["confidence"] == "high"
    assert result["master_pct"] == 30
    assert result["publishing_pct"] == 0
    assert result["soundexchange_pct"] == 15
    assert result["matched_file_ids"] == ["f1"]
    assert result["terms"] == []


def test_name_not_matched_returns_not_found():
    db = _db()
    cd = _cd([("Jane Doe", "writer", 50, 50)])
    with patch.object(derive_service, "get_or_parse", return_value=cd):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert result["found"] is False
    assert result["confidence"] == "low"
    assert result["master_pct"] is None
    assert result["publishing_pct"] is None
    assert result["soundexchange_pct"] is None
    assert result["matched_file_ids"] == []
    assert result["terms"] == []


def test_derivation_uses_get_or_parse_result():
    db = _db()  # no downloads configured; get_or_parse is patched so none happen
    cd = _cd([("Marcus", "producer", 25, 0)])
    with patch.object(derive_service, "get_or_parse", return_value=cd) as mock_gop:
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert mock_gop.call_count == 1
    assert result["found"] is True
    assert result["confidence"] == "high"
    assert result["master_pct"] == 25
    assert result["matched_file_ids"] == ["f1"]


def test_party_with_no_split_is_dropped_not_found():
    """A party listed in the contract with no royalty split at all is dropped
    by the pivot (merely mentioned), so the collaborator is not found."""
    db = _db()
    cd = _cd([("Marcus", "producer", 0, 0)])
    with patch.object(derive_service, "get_or_parse", return_value=cd):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert result["found"] is False
    assert result["confidence"] == "low"
    assert result["master_pct"] is None
    assert result["matched_file_ids"] == []


def test_parse_failure_returns_not_found():
    db = _db()
    with patch.object(derive_service, "get_or_parse", side_effect=RuntimeError("boom")):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert result["found"] is False
    assert result["matched_file_ids"] == []


# ---------------------------------------------------------------------------
# Alias matching
# ---------------------------------------------------------------------------


def test_collaborator_matched_via_alias():
    """A party listed by legal name with the collaborator's stage name in
    `aliases` still matches."""
    db = _db()
    cd = _cd([("Marcus Adebayo", "producer", 30, 0)])
    cd.parties[0].aliases = ["M-Bay"]
    with patch.object(derive_service, "get_or_parse", return_value=cd):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "M-Bay"))

    assert result["found"] is True
    assert result["confidence"] == "high"
    assert result["master_pct"] == 30
    assert result["matched_file_ids"] == ["f1"]


def test_stale_cache_payload_without_aliases_still_matches_by_name():
    """Cached parses from before the aliases field deserialize with empty
    aliases and behave exactly as before (matched by name)."""
    cached_payload = {
        "parties": [{"name": "Marcus", "role": "producer"}],  # no "aliases" key
        "works": [],
        "royalty_shares": [{"party_name": "Marcus", "royalty_type": "Master", "percentage": 25.0}],
        "contract_summary": "",
        "default_basis": None,
    }
    cd = deserialize_contract_data(cached_payload)
    db = _db()
    with patch.object(derive_service, "get_or_parse", return_value=cd):
        result = asyncio.run(derive_service.derive_for_collaborator(db, "w1", "Marcus"))

    assert result["found"] is True
    assert result["master_pct"] == 25
    assert result["matched_file_ids"] == ["f1"]
