"""POST /registry/v1/splits — contract PDFs in, the deal as data out
(contract_terms in the royalties input shape + the Registry splits pivot),
SSE with the debit after the result frame, idempotent under Idempotency-Key."""

import json
from unittest.mock import MagicMock

import pytest

from partner_api.service import PartnerContext

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
CTX = PartnerContext(org_id=ORG_ID, key_id="00000000-0000-0000-0000-0000000000bb")
PERIOD = "2026-09-30T00:00:00+00:00"
URL = "/registry/v1/splits"
H = {"Authorization": "Bearer mk_live_ok"}
PARSED = {
    "contract_terms": {
        "parties": [{"name": "Jane Doe", "role": "producer", "aliases": []}],
        "works": [{"title": "Blue Sky", "work_type": "song"}],
        "royalty_shares": [
            {"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50.0, "terms": None, "basis": "net"}
        ],
        "contract_summary": None,
        "default_basis": None,
    },
    "splits": {
        "main_artist": None,
        "parties": [
            {
                "name": "Jane Doe",
                "role": "producer",
                "master_pct": 50.0,
                "publishing_pct": 0.0,
                "soundexchange_pct": 0.0,
            }
        ],
    },
}


@pytest.fixture
def partner(monkeypatch):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    monkeypatch.setattr("partner_api.service.resolve_key", lambda sb, b: CTX)
    monkeypatch.setattr("partner_api.service.get_price", lambda sb, action: {"partner_registry_parse": 30}[action])
    monkeypatch.setattr(
        "partner_api.service.check_pool",
        lambda sb, org, price: {"ok": True, "balance": 100, "wallet_id": "w1", "period_end": PERIOD},
    )
    monkeypatch.setattr("partner_api.service.already_charged", lambda sb, rid: False)


@pytest.fixture
def parsed(monkeypatch):
    calls = []

    def fake(sb, **kw):
        calls.append(kw)
        return PARSED, None, None

    monkeypatch.setattr("partner_api.service.run_partner_parse", fake)
    return calls


@pytest.fixture
def debits(monkeypatch):
    seen = []
    monkeypatch.setattr("partner_api.service.debit_run", lambda sb, **kw: seen.append(kw))
    return seen


def _events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def _post(client, files=None, data=None, headers=None):
    return client.post(
        URL,
        files=files if files is not None else [("contracts", ("deal.pdf", b"%PDF-1.4 x", "application/pdf"))],
        data=data or {},
        headers={**H, **(headers or {})},
    )


def test_no_contracts_is_422(client, partner):
    r = client.post(URL, data={"main_artist_name": "x"}, headers=H)
    assert r.status_code == 422


def test_non_pdf_is_422_and_too_many_is_413(client, partner, parsed):
    r = _post(client, files=[("contracts", ("deal.docx", b"x", "application/octet-stream"))])
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_request"
    r = _post(client, files=[("contracts", (f"c{i}.pdf", b"%PDF", "application/pdf")) for i in range(11)])
    assert r.status_code == 413 and r.json()["detail"]["code"] == "too_many_contracts"
    assert parsed == []


def test_dry_pool_is_402_and_parse_never_runs(client, partner, parsed, monkeypatch):
    monkeypatch.setattr(
        "partner_api.service.check_pool",
        lambda sb, org, price: {"ok": False, "balance": 3, "wallet_id": "w1", "period_end": PERIOD},
    )
    r = _post(client)
    assert r.status_code == 402
    assert r.json()["detail"] == {"code": "insufficient_credits", "price": 30, "balance": 3}
    assert parsed == []


def test_result_event_carries_both_views_and_bills_the_registry_row(client, partner, parsed, debits):
    r = _post(client, data={"main_artist_name": "Jane Doe"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    (event,) = _events(r.text)
    assert event["type"] == "result"
    assert event["contract_terms"] == PARSED["contract_terms"]
    assert event["splits"] == PARSED["splits"]
    # The pivot is built for the named artist; the files went to the worker.
    assert parsed[0]["main_artist_name"] == "Jane Doe" and parsed[0]["org_id"] == ORG_ID
    assert len(parsed[0]["contract_paths"]) == 1
    # Billed once, on the API's own registry row, unmeasured -> the base.
    assert len(debits) == 1
    assert debits[0]["action"] == "partner_registry_parse" and debits[0]["amount"] == 30
    assert debits[0]["wallet_id"] == "w1" and debits[0]["key_id"] == CTX.key_id
    assert event["billing"] == {"credits": 30, "request_id": debits[0]["request_id"]}


def test_idempotency_key_binds_files_and_artist(client, partner, parsed, debits):
    _post(client, headers={"Idempotency-Key": "k1"})
    _post(client, headers={"Idempotency-Key": "k1"})
    assert debits[0]["request_id"] == debits[1]["request_id"]
    # Same files, different main artist: a different pivot, a different deliverable.
    _post(client, data={"main_artist_name": "Sam Ray"}, headers={"Idempotency-Key": "k1"})
    assert debits[2]["request_id"] != debits[0]["request_id"]
    # No header: every run pays.
    _post(client)
    _post(client)
    assert debits[3]["request_id"] != debits[4]["request_id"]


def test_unreadable_contract_is_an_error_event_and_unbilled(client, partner, debits, monkeypatch):
    def boom(sb, **kw):
        raise ValueError("no extractable text")

    monkeypatch.setattr("partner_api.service.run_partner_parse", boom)
    r = _post(client)
    assert r.status_code == 200
    (event,) = _events(r.text)
    assert event["type"] == "error" and event["code"] == "CONTRACT_UNREADABLE"
    assert event["details"] == {"reason": "no extractable text"}
    assert event["suggestion"]
    assert event["billing"] == {"credits": 0}
    assert debits == []


def test_internal_error_event_carries_request_id_and_is_unbilled(client, partner, debits, monkeypatch):
    monkeypatch.setattr("partner_api.service.run_partner_parse", MagicMock(side_effect=RuntimeError("llm down")))
    r = _post(client, headers={"Idempotency-Key": "k1"})
    (event,) = _events(r.text)
    assert event["type"] == "error" and event["code"] == "internal_error"
    assert event["request_id"]
    assert event["billing"] == {"credits": 0}
    assert debits == []


def test_replay_under_idempotency_key_reports_zero_credits(client, partner, parsed, debits, monkeypatch):
    asked = []
    monkeypatch.setattr("partner_api.service.already_charged", lambda sb, rid: asked.append(rid) or True)
    r = _post(client, headers={"Idempotency-Key": "k1"})
    (event,) = _events(r.text)
    assert event["billing"] == {"credits": 0, "request_id": asked[0], "replayed": True}
    assert debits[0]["request_id"] == asked[0]
    # No header => a fresh uuid4 that cannot have been charged: the read is skipped.
    _post(client)
    assert len(asked) == 1
