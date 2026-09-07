import json
from unittest.mock import MagicMock

import pytest

from oneclick.royalty_calculator import CalculationError
from partner_api.service import PartnerContext

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
CTX = PartnerContext(org_id=ORG_ID, key_id="00000000-0000-0000-0000-0000000000bb")

PERIOD = "2026-09-30T00:00:00+00:00"
RESULT = {"summary": {"payments": 0, "total_payable": 0.0, "expense_review_required": False}, "payments": []}
TERMS = json.dumps(
    {
        "parties": [{"name": "A", "role": "artist"}],
        "works": [{"title": "Song"}],
        "royalty_shares": [{"party_name": "A", "royalty_type": "Streaming", "percentage": 50.0}],
    }
)


@pytest.fixture
def partner(monkeypatch):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    monkeypatch.setattr("partner_api.router.psvc.resolve_key", lambda sb, b: CTX)
    monkeypatch.setattr("partner_api.router.psvc.get_price", lambda sb, action: 30)
    monkeypatch.setattr(
        "partner_api.router.psvc.check_pool",
        lambda sb, org, price: {"ok": True, "balance": 100, "wallet_id": "w1", "period_end": PERIOD},
    )
    monkeypatch.setattr("partner_api.router.psvc.already_charged", lambda sb, rid: False)


def _post(client, files=None, data=None, headers=None):
    h = {"Authorization": "Bearer mk_live_ok"}
    h.update(headers or {})
    return client.post(
        "/oneclick/v1/royalties",
        files=files or {"statement": ("s.csv", b"Title,Net\nSong,100\n", "text/csv")},
        data=data or {"contract_terms": TERMS},
        headers=h,
    )


def test_neither_or_both_contract_options_is_422(client, partner):
    r = client.post(
        "/oneclick/v1/royalties",
        files={"statement": ("s.csv", b"x", "text/csv")},
        headers={"Authorization": "Bearer mk_live_ok"},
    )
    assert r.status_code == 422


def test_non_pdf_contract_is_422(client, partner):
    r = client.post(
        "/oneclick/v1/royalties",
        files=[
            ("statement", ("s.csv", b"x", "text/csv")),
            ("contracts", ("c.docx", b"x", "application/octet-stream")),
        ],
        headers={"Authorization": "Bearer mk_live_ok"},
    )
    assert r.status_code == 422


def test_dry_pool_is_402_and_calc_never_runs(client, partner, monkeypatch):
    monkeypatch.setattr(
        "partner_api.router.psvc.check_pool",
        lambda sb, org, price: {"ok": False, "balance": 3, "wallet_id": "w1", "period_end": PERIOD},
    )
    ran = MagicMock()
    monkeypatch.setattr("partner_api.router.psvc.run_partner_calc", ran)
    r = _post(client)
    assert r.status_code == 402
    assert r.json()["detail"]["balance"] == 3
    ran.assert_not_called()


def test_success_streams_result_and_debits(client, partner, monkeypatch):
    monkeypatch.setattr(
        "partner_api.router.psvc.run_partner_calc",
        lambda sb, **kw: (dict(RESULT), None, None),  # terms mode: unmeasured
    )
    debits = []
    monkeypatch.setattr("partner_api.router.psvc.debit_run", lambda sb, **kw: debits.append(kw))
    r = _post(client, headers={"Idempotency-Key": "retry-1"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "result"
    assert events[-1]["summary"] == RESULT["summary"]
    assert events[-1]["billing"] == {"credits": 30, "request_id": debits[0]["request_id"]}
    assert debits and debits[0]["amount"] == 30
    assert debits[0]["metadata"]["base"] == 30  # compute_charge metadata rides on the debit

    # Same header + same payload => same request id (the debit RPC dedupes).
    _post(client, headers={"Idempotency-Key": "retry-1"})
    assert debits[1]["request_id"] == debits[0]["request_id"]
    # Same header + DIFFERENT payload => new request id (pays again).
    other_terms = json.dumps({**json.loads(TERMS), "default_basis": "net"})
    _post(client, data={"contract_terms": other_terms}, headers={"Idempotency-Key": "retry-1"})
    assert debits[2]["request_id"] != debits[0]["request_id"]


def test_request_id_is_scoped_to_the_pools_billing_period(client, partner, monkeypatch):
    # The pool's period_end reaches derive_request_id, so the SAME
    # Idempotency-Key + payload stops being a free duplicate next period.
    monkeypatch.setattr("partner_api.router.psvc.run_partner_calc", lambda sb, **kw: (dict(RESULT), None, None))
    debits = []
    monkeypatch.setattr("partner_api.router.psvc.debit_run", lambda sb, **kw: debits.append(kw))
    _post(client, headers={"Idempotency-Key": "nightly"})
    monkeypatch.setattr(
        "partner_api.router.psvc.check_pool",
        lambda sb, org, price: {
            "ok": True,
            "balance": 100,
            "wallet_id": "w1",
            "period_end": "2026-10-31T00:00:00+00:00",
        },
    )
    _post(client, headers={"Idempotency-Key": "nightly"})
    assert debits[1]["request_id"] != debits[0]["request_id"]


def test_replay_under_idempotency_key_reports_zero_credits(client, partner, monkeypatch):
    monkeypatch.setattr("partner_api.router.psvc.run_partner_calc", lambda sb, **kw: (dict(RESULT), None, None))
    debits = []
    monkeypatch.setattr("partner_api.router.psvc.debit_run", lambda sb, **kw: debits.append(kw))
    asked = []
    monkeypatch.setattr("partner_api.router.psvc.already_charged", lambda sb, rid: asked.append(rid) or True)
    r = _post(client, headers={"Idempotency-Key": "retry-1"})
    event = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")][-1]
    assert event["billing"] == {"credits": 0, "request_id": asked[0], "replayed": True}
    # The debit RPC is the authority and still runs at the full amount — the
    # read only reports; debit_credits' own dedupe is what makes it a no-op.
    assert debits[0]["amount"] == 30
    # No header => a fresh uuid4 that cannot have been charged: the read is skipped.
    _post(client)
    assert len(asked) == 1


def test_metered_cost_above_base_charges_metered(client, partner, monkeypatch):
    # The ONE wiring proof that measured/usage reach compute_charge. The
    # formula's other terms (size tail, unmeasured -> base) are pinned in
    # test_compute_charge.py; re-asserting them here would only pin them twice.
    monkeypatch.setattr("partner_api.router.psvc.run_partner_calc", lambda sb, **kw: (dict(RESULT), 45, {}))
    debits = []
    monkeypatch.setattr("partner_api.router.psvc.debit_run", lambda sb, **kw: debits.append(kw))
    r = _post(client)
    assert debits[0]["amount"] == 45  # metered beat the 30-credit base
    assert debits[0]["metadata"]["metered"] is True
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["billing"]["credits"] == 45  # the body reports THE charge, not the base


def test_calculation_error_is_terminal_sse_event(client, partner, monkeypatch):
    def boom(sb, **kw):
        raise CalculationError(
            code="NO_SONG_MATCHES",
            message="No songs matched.",
            suggestion="Check the statement titles.",
            details={"contract_works": ["A"]},
        )

    monkeypatch.setattr("partner_api.router.psvc.run_partner_calc", boom)
    debit = MagicMock()
    monkeypatch.setattr("partner_api.router.psvc.debit_run", debit)
    r = _post(client)
    assert r.status_code == 200
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == {
        "type": "error",
        "code": "NO_SONG_MATCHES",
        "message": "No songs matched.",
        "suggestion": "Check the statement titles.",
        "details": {"contract_works": ["A"]},
        "billing": {"credits": 0},
    }
    debit.assert_not_called()


def test_debit_failure_still_returns_result(client, partner, monkeypatch):
    monkeypatch.setattr("partner_api.router.psvc.run_partner_calc", lambda sb, **kw: (RESULT, None, None))

    def debit_boom(sb, **kw):
        raise RuntimeError("rpc down")

    monkeypatch.setattr("partner_api.router.psvc.debit_run", debit_boom)
    r = _post(client)
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "result"


def test_bad_statement_fails_before_any_llm_call(client, partner, monkeypatch):
    # Statement failures are CalculationErrors, and the router returns those
    # WITHOUT debiting — so validating the statement must come first, or a
    # junk statement buys unbilled contract parses on repeat.
    parsed = []
    monkeypatch.setattr("utils.ingestion.pdf_markdown.pdf_to_markdown", lambda p: parsed.append(p) or "md")
    monkeypatch.setattr("utils.contract_parsing.cache.get_or_parse", lambda sb, fn: parsed.append("parse"))

    class FakeCalc:
        def read_royalty_statement(self, path):
            raise CalculationError(code="STATEMENT_EMPTY", message="No rows.", suggestion="Add rows.")

    monkeypatch.setattr("oneclick.royalty_calculator.RoyaltyCalculator", FakeCalc)
    debit = MagicMock()
    monkeypatch.setattr("partner_api.router.psvc.debit_run", debit)

    r = client.post(
        "/oneclick/v1/royalties",
        files=[
            ("statement", ("s.csv", b"", "text/csv")),
            ("contracts", ("c.pdf", b"%PDF-1.4", "application/pdf")),
        ],
        headers={"Authorization": "Bearer mk_live_ok"},
    )
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["code"] == "STATEMENT_EMPTY"
    assert parsed == []  # the ordering, not just the error
    debit.assert_not_called()


def test_dry_pool_402_carries_price_and_balance(client, partner, monkeypatch):
    # A key is the org's own credential, so the balance is theirs to see.
    monkeypatch.setattr(
        "partner_api.router.psvc.check_pool",
        lambda sb, org, price: {"ok": False, "balance": 3, "wallet_id": "w1", "period_end": PERIOD},
    )
    assert _post(client).json()["detail"] == {"code": "insufficient_credits", "price": 30, "balance": 3}


def test_too_many_contracts_is_413(client, partner):
    files = [("statement", ("s.csv", b"x", "text/csv"))]
    files += [("contracts", (f"c{i}.pdf", b"x", "application/pdf")) for i in range(11)]
    r = client.post("/oneclick/v1/royalties", files=files, headers={"Authorization": "Bearer mk_live_ok"})
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "too_many_contracts"


def test_oversized_statement_is_413(client, partner, monkeypatch):
    # Cap shrunk rather than uploading 10 MB — _save_upload's guard is the subject.
    monkeypatch.setattr("partner_api.router.MAX_STATEMENT_BYTES", 4)
    r = _post(client, files={"statement": ("s.csv", b"way past the cap", "text/csv")})
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "file_too_large"


def test_client_that_disconnects_before_the_result_is_not_billed(partner, monkeypatch):
    # The rule: no delivered output, no charge (2026-09-04). Neither TestClient
    # nor httpx's ASGI transport can drop a connection mid-stream — both drain
    # the response — so drive the generator directly and close it at the result
    # frame, which is exactly what Starlette does when the client is gone.
    import asyncio
    import io

    from fastapi import UploadFile

    from partner_api.models import PartnerContractTerms
    from partner_api.router import partner_calculate

    monkeypatch.setattr("partner_api.router._get_supabase", lambda: MagicMock())
    monkeypatch.setattr("partner_api.router.psvc.run_partner_calc", lambda *a, **k: (RESULT, None, {}))
    debit = MagicMock()
    monkeypatch.setattr("partner_api.router.psvc.debit_run", debit)

    async def drive():
        resp = await partner_calculate(
            statement=UploadFile(file=io.BytesIO(b"Title,Net\nSong,100\n"), filename="s.csv"),
            contracts=[],
            contract_terms=PartnerContractTerms.model_validate_json(TERMS),
            expenses=None,
            idempotency_key=None,
            ctx=CTX,
        )
        it = resp.body_iterator
        chunk = await it.__anext__()
        await it.aclose()  # client went away before reading further
        return chunk

    chunk = asyncio.run(drive())
    assert '"type": "result"' in chunk
    debit.assert_not_called()
