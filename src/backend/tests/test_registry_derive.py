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


# ---------------------------------------------------------------------------
# Endpoint gating: Derive runs a full LLM extraction per contract, so it is a
# metered AI action. It previously charged NOTHING — only the Registry feature
# flag guarded it — which leaked the whole cost of the Derive dialog.
# ---------------------------------------------------------------------------


def _drive_derive_endpoint(monkeypatch, *, derive_result=None, burn_tokens=False):
    """Call the router coroutine directly with its deps stubbed, returning
    (gated_credits calls, debit_for_action calls, credits measured at debit)."""
    from unittest.mock import MagicMock

    from registry import router as registry_router
    from registry.models import DeriveFromContractsBody
    from utils.llm.tracking import TrackedOpenAI, credits_for_llm_usage

    gate_calls, debit_calls, measured = [], [], []

    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"id": "link-1"}]
    )
    db.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])

    async def _work_access(*_a, **_k):
        return MagicMock(can_manage=True)

    async def _derive(*_a, **_k):
        if burn_tokens:
            inner = MagicMock()
            inner.chat.completions.create.return_value = MagicMock(
                usage=MagicMock(prompt_tokens=200_000, completion_tokens=0, prompt_tokens_details=None)
            )
            TrackedOpenAI(inner, get_supabase=lambda: MagicMock()).chat.completions.create(model="gpt-5-mini")
        return derive_result or {"found": False, "confidence": "none", "matched_file_ids": []}

    ent = MagicMock()
    ent.debit_for_action.side_effect = lambda *a: (
        debit_calls.append(a),
        measured.append(credits_for_llm_usage()),
    )

    monkeypatch.setattr(registry_router, "gated_feature", lambda *a, **k: None)
    monkeypatch.setattr(registry_router, "gated_credits", lambda *a, **k: gate_calls.append((a, k)) or "grant")
    monkeypatch.setattr(registry_router, "_get_supabase", lambda: db)
    monkeypatch.setattr(registry_router, "get_work_access", _work_access)
    monkeypatch.setattr(registry_router, "analytics_capture", lambda *a, **k: None)
    monkeypatch.setattr(derive_service, "derive_for_collaborator", _derive)
    monkeypatch.setattr("subscriptions.deps._get_entitlements_service", lambda: ent)

    body = DeriveFromContractsBody(work_id="w1", name="Marcus", contract_file_ids=["f1"])
    asyncio.run(registry_router.derive_from_contracts(body, user_id="u1"))
    return gate_calls, debit_calls, measured


def test_derive_endpoint_is_credit_gated(monkeypatch):
    """It must take the REGISTRY_PARSE credit gate, and derive the billing org
    from the contracts it is about to parse."""
    from subscriptions.models import CreditAction

    gate_calls, debit_calls, _ = _drive_derive_endpoint(monkeypatch)

    assert len(gate_calls) == 1
    args, kwargs = gate_calls[0]
    assert args == ("u1", CreditAction.REGISTRY_PARSE)
    assert kwargs["resource_contract_ids"] == ["f1"]
    assert len(debit_calls) == 1  # charge-on-success


def test_derive_endpoint_debits_inside_the_tracking_scope(monkeypatch):
    """The debit must see the tokens the parses burned — otherwise the metered
    amount reads as 0 and the whole derive is free again."""
    _, _, measured = _drive_derive_endpoint(monkeypatch, burn_tokens=True)
    assert measured == [8]  # $0.05 of gpt-5-mini x 3 markup / $0.02 per credit


def test_derive_endpoint_charges_nothing_on_an_all_cached_derive(monkeypatch):
    """Every contract served from the parse cache burns no tokens -> 0 credits."""
    _, _, measured = _drive_derive_endpoint(monkeypatch, burn_tokens=False)
    assert measured == [0]
