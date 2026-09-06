"""POST /splitsheet/v1/documents — JSON in, the PDF or DOCX out, billed at
exactly the base after the file is delivered, idempotent under Idempotency-Key."""

import pytest

from partner_api.service import PartnerContext

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
CTX = PartnerContext(org_id=ORG_ID, key_id="00000000-0000-0000-0000-0000000000bb")
PERIOD = "2026-09-30T00:00:00+00:00"
URL = "/splitsheet/v1/documents"
H = {"Authorization": "Bearer mk_live_ok"}
BODY = {
    "work_title": "Blue Sky",
    "date": "6 September 2026",
    "contributors": [
        {"name": "Jane Doe", "role": "Producer", "publishing_share": 50, "master_percentage": 50},
        {"name": "Sam Ray", "role": "Writer", "publishing_share": 50, "master_percentage": 50},
    ],
}


@pytest.fixture
def partner(monkeypatch):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    monkeypatch.setattr("partner_api.service.resolve_key", lambda sb, b: CTX)
    monkeypatch.setattr("partner_api.service.get_price", lambda sb, action: {"partner_split_sheet": 20}[action])
    monkeypatch.setattr(
        "partner_api.service.check_pool",
        lambda sb, org, price: {"ok": True, "balance": 100, "wallet_id": "w1", "period_end": PERIOD},
    )


@pytest.fixture
def debits(monkeypatch):
    seen = []
    monkeypatch.setattr("partner_api.service.debit_run", lambda sb, **kw: seen.append(kw))
    return seen


def test_pdf_is_rendered_by_the_product_generator_and_billed_after_delivery(client, partner, debits):
    r = client.post(URL, json=BODY, headers=H)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"] == 'attachment; filename="Split_Sheet_Blue_Sky.pdf"'
    assert r.content.startswith(b"%PDF")
    assert int(r.headers["content-length"]) == len(r.content)
    assert len(debits) == 1
    assert debits[0]["action"] == "partner_split_sheet" and debits[0]["amount"] == 20
    assert debits[0]["wallet_id"] == "w1" and debits[0]["key_id"] == CTX.key_id


def test_docx_is_a_second_deliverable(client, partner, debits):
    r = client.post(URL, json={**BODY, "format": "docx"}, headers=H)
    assert r.status_code == 200
    assert r.headers["content-type"].endswith("wordprocessingml.document")
    assert r.headers["content-disposition"].endswith('.docx"')
    assert r.content.startswith(b"PK")
    assert len(debits) == 1


def test_validation_is_a_422(client, partner, debits):
    assert client.post(URL, json={**BODY, "contributors": []}, headers=H).status_code == 422
    assert client.post(URL, json={**BODY, "format": "xlsx"}, headers=H).status_code == 422
    assert client.post(URL, json={**BODY, "split_type": "sync"}, headers=H).status_code == 422
    assert client.post(URL, json={k: v for k, v in BODY.items() if k != "date"}, headers=H).status_code == 422
    bad = {**BODY, "contributors": [{"name": "A", "role": "x", "master_percentage": 120}]}
    assert client.post(URL, json=bad, headers=H).status_code == 422
    assert debits == []


def test_dry_pool_is_402_before_rendering(client, partner, debits, monkeypatch):
    monkeypatch.setattr(
        "partner_api.service.check_pool",
        lambda sb, org, price: {"ok": False, "balance": 5, "wallet_id": "w1", "period_end": PERIOD},
    )
    monkeypatch.setattr("partner_api.splitsheet.render", lambda req: (_ for _ in ()).throw(AssertionError("rendered")))
    r = client.post(URL, json=BODY, headers=H)
    assert r.status_code == 402
    assert r.json()["detail"] == {"code": "insufficient_credits", "price": 20, "balance": 5}
    assert debits == []


def test_idempotency_key_binds_the_body_including_format(client, partner, debits, monkeypatch):
    monkeypatch.setattr("partner_api.splitsheet.render", lambda req: b"%PDF-fake")
    k = {"Idempotency-Key": "sheet-1"}
    client.post(URL, json=BODY, headers={**H, **k})
    client.post(URL, json=BODY, headers={**H, **k})
    assert debits[0]["request_id"] == debits[1]["request_id"]
    client.post(URL, json={**BODY, "format": "docx"}, headers={**H, **k})
    assert debits[2]["request_id"] != debits[0]["request_id"]
    client.post(URL, json=BODY, headers=H)
    client.post(URL, json=BODY, headers=H)
    assert debits[3]["request_id"] != debits[4]["request_id"]


def test_generator_failure_is_a_500_with_request_id_and_unbilled(client, partner, debits, monkeypatch):
    def boom(req):
        raise RuntimeError("reportlab exploded")

    monkeypatch.setattr("partner_api.splitsheet.render", boom)
    r = client.post(URL, json=BODY, headers=H)
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "internal_error" and r.json()["detail"]["request_id"]
    assert debits == []
