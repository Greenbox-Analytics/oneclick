"""Tests for payout PDF documents (breakdown + receipt), the save-to-project
endpoint, and the payee receipt email.

All tests mock Supabase / storage / Resend — the only real work is ReportLab
PDF generation (deterministic, asserted via the %PDF magic bytes) and money
formatting (kept real for correctness).
"""

import base64
from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import emails, pdf, service

USER_ID = "user-aaa"
PAYEE_ID = "payee-111"
PAYOUT_ID = "payout-xyz"
ARTIST_ID = "artist-1"
PROJECT_ID = "proj-1"

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_snapshot():
    return {
        "payee": {"id": PAYEE_ID, "display_name": "Alice", "payout_currency": "USD"},
        "fx": {"rate_date": "2026-07-01", "rates_used": {"EUR->USD": 1.08}},
        "total_pay_ccy": 1234.56,
        "projects": [
            {
                "project_id": PROJECT_ID,
                "name": "Album One",
                "statements": [
                    {
                        "royalty_statement_id": "stmt-1",
                        "period_start": "2026-01-01",
                        "period_end": "2026-03-31",
                        "statement_currency": "EUR",
                        "statement_total": 5000.0,
                        "payee_subtotal_owed": 1143.11,
                        "payee_subtotal_pay_ccy": 1234.56,
                        "lines": [
                            {
                                "song": "Song A",
                                "role": "Writer",
                                "royalty_type": "mechanical",
                                "percentage": 50.0,
                                "amount_owed": 1143.11,
                                "amount_pay_ccy": 1234.56,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _make_payout(status="paid", pay_currency="USD", payment_method="paypal", snapshot=None):
    return {
        "id": PAYOUT_ID,
        "user_id": USER_ID,
        "payee_id": PAYEE_ID,
        "status": status,
        "pay_currency": pay_currency,
        "total_amount": 1234.56,
        "fx_rate_date": "2026-07-01",
        "created_at": "2026-07-01T00:00:00Z",
        "paid_at": "2026-07-01T01:00:00Z" if status == "paid" else None,
        "note": None,
        "breakdown_snapshot": _make_snapshot() if snapshot is None else snapshot,
        "payment_method": payment_method,
        "paypal_order_id": "ORDER-1" if payment_method == "paypal" else None,
        "paypal_capture_id": "CAPTURE-1" if payment_method == "paypal" else None,
        "orphan_state": "none",
    }


# ---------------------------------------------------------------------------
# PDF builders
# ---------------------------------------------------------------------------


class TestPdfBuilders:
    def test_breakdown_pdf_valid(self):
        out = pdf.generate_breakdown_pdf(_make_payout()).read()
        assert out.startswith(b"%PDF")
        assert len(out) > 1000

    def test_receipt_pdf_valid(self):
        out = pdf.generate_receipt_pdf(_make_payout(), payer_name="Kenji").read()
        assert out.startswith(b"%PDF")

    def test_empty_snapshot_handled(self):
        payout = _make_payout(snapshot={})
        assert pdf.generate_breakdown_pdf(payout).read().startswith(b"%PDF")
        assert pdf.generate_receipt_pdf(payout).read().startswith(b"%PDF")

    def test_missing_optional_fields_handled(self):
        snapshot = _make_snapshot()
        line = snapshot["projects"][0]["statements"][0]["lines"][0]
        line.update({"role": None, "royalty_type": None, "percentage": None})
        snapshot["projects"][0]["statements"][0]["statement_total"] = None
        payout = _make_payout(snapshot=snapshot)
        payout["paid_at"] = None
        assert pdf.generate_breakdown_pdf(payout).read().startswith(b"%PDF")
        assert pdf.generate_receipt_pdf(payout).read().startswith(b"%PDF")

    def test_manual_payment_method(self):
        out = pdf.generate_receipt_pdf(_make_payout(payment_method="manual")).read()
        assert out.startswith(b"%PDF")

    @pytest.mark.parametrize(
        ("amount", "ccy", "expected"),
        [
            (1234.56, "USD", "1,234.56 USD"),
            (1234.5, "JPY", "1,235 JPY"),  # zero-decimal, rounded
            (0, "USD", "0.00 USD"),
            (None, "JPY", "0 JPY"),
            (-5, "USD", "0.00 USD"),
        ],
    )
    def test_fmt_money(self, amount, ccy, expected):
        assert pdf.fmt_money(amount, ccy) == expected


# ---------------------------------------------------------------------------
# service.get_paid_payout
# ---------------------------------------------------------------------------


class _SelectDB:
    """Tiny mock: royalty_payouts select returns the given rows."""

    def __init__(self, rows):
        self._rows = rows

    def table(self, name):
        proxy = MagicMock()
        proxy.select.return_value = proxy
        proxy.eq.return_value = proxy
        filtered = self._rows
        proxy.execute.return_value = MagicMock(data=filtered)
        return proxy


class TestGetPaidPayout:
    def test_paid_ok(self):
        payout = _make_payout()
        assert service.get_paid_payout(_SelectDB([payout]), USER_ID, PAYOUT_ID) == payout

    def test_draft_raises_state_error(self):
        with pytest.raises(service.PayoutStateError):
            service.get_paid_payout(_SelectDB([_make_payout(status="draft")]), USER_ID, PAYOUT_ID)

    def test_missing_raises_permission_error(self):
        with pytest.raises(PermissionError):
            service.get_paid_payout(_SelectDB([]), USER_ID, PAYOUT_ID)


# ---------------------------------------------------------------------------
# Router endpoints
# ---------------------------------------------------------------------------


def _make_router_client(**patches):
    """Minimal app with the royalties router; auth/gating/supabase mocked.

    patches: {patch_target: {"side_effect"|"return_value": ...}} applied on top.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth import get_current_user_id
    from oneclick.royalties.router import router

    app = FastAPI()
    app.include_router(router)

    async def _mock_user_id():
        return USER_ID

    app.dependency_overrides[get_current_user_id] = _mock_user_id

    patchers = [
        patch("oneclick.royalties.router.gated_feature", return_value=None),
        patch("oneclick.royalties.router._get_supabase", return_value=MagicMock()),
    ]
    for target, kwargs in patches.items():
        patchers.append(patch(target, **kwargs))
    for p in patchers:
        p.start()

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    for p in patchers:
        p.stop()


class TestBreakdownPdfEndpoint:
    def test_200_pdf(self):
        for client in _make_router_client(
            **{"oneclick.royalties.service.get_payout": {"return_value": _make_payout()}}
        ):
            resp = client.get(f"/payouts/{PAYOUT_ID}/breakdown.pdf")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/pdf"
            assert resp.content.startswith(b"%PDF")
            assert "Payout_Breakdown_Alice" in resp.headers["content-disposition"]

    def test_404_foreign(self):
        for client in _make_router_client(
            **{"oneclick.royalties.service.get_payout": {"side_effect": PermissionError("nope")}}
        ):
            resp = client.get(f"/payouts/{PAYOUT_ID}/breakdown.pdf")
            assert resp.status_code == 404


class TestReceiptPdfEndpoint:
    def test_200_for_paid(self):
        for client in _make_router_client(
            **{"oneclick.royalties.service.get_paid_payout": {"return_value": _make_payout()}}
        ):
            resp = client.get(f"/payouts/{PAYOUT_ID}/receipt.pdf")
            assert resp.status_code == 200
            assert resp.content.startswith(b"%PDF")
            assert "Receipt_Alice" in resp.headers["content-disposition"]

    def test_409_for_draft(self):
        for client in _make_router_client(
            **{
                "oneclick.royalties.service.get_paid_payout": {
                    "side_effect": service.PayoutStateError("Receipts are only available for paid payouts")
                }
            }
        ):
            resp = client.get(f"/payouts/{PAYOUT_ID}/receipt.pdf")
            assert resp.status_code == 409

    def test_404_foreign(self):
        for client in _make_router_client(
            **{"oneclick.royalties.service.get_paid_payout": {"side_effect": PermissionError("nope")}}
        ):
            resp = client.get(f"/payouts/{PAYOUT_ID}/receipt.pdf")
            assert resp.status_code == 404


class TestSaveReceiptEndpoint:
    BODY = {"artist_id": ARTIST_ID, "project_id": PROJECT_ID}

    def test_200_saves_to_storage_and_project_files(self):
        db = MagicMock()
        storage_bucket = MagicMock()
        db.storage.from_.return_value = storage_bucket
        storage_bucket.get_public_url.return_value = "https://files/receipt.pdf"
        inserted_row = {"id": "file-1", "project_id": PROJECT_ID}
        db.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[inserted_row])
        # profiles lookup path (payer name) also goes through db.table — make select chain harmless
        db.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])

        for client in _make_router_client(
            **{
                "oneclick.royalties.router._get_supabase": {"return_value": db},
                "oneclick.royalties.service.get_paid_payout": {"return_value": _make_payout()},
                "main.verify_user_owns_artist": {"return_value": True},
                "main.verify_user_owns_project": {"return_value": True},
            }
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/receipt/save", json=self.BODY)
            assert resp.status_code == 200
            assert resp.json() == inserted_row

            db.storage.from_.assert_called_with("project-files")
            upload_path = storage_bucket.upload.call_args[0][0]
            assert upload_path.startswith(f"{ARTIST_ID}/{PROJECT_ID}/receipts/")
            uploaded_bytes = storage_bucket.upload.call_args[0][1]
            assert uploaded_bytes.startswith(b"%PDF")

            insert_record = db.table.return_value.insert.call_args[0][0]
            assert insert_record["folder_category"] == "other"
            assert insert_record["project_id"] == PROJECT_ID
            assert insert_record["file_type"] == "application/pdf"

    def test_403_when_not_owner_of_artist_or_project(self):
        for client in _make_router_client(
            **{
                "oneclick.royalties.service.get_paid_payout": {"return_value": _make_payout()},
                "main.verify_user_owns_artist": {"return_value": False},
                "main.verify_user_owns_project": {"return_value": True},
            }
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/receipt/save", json=self.BODY)
            assert resp.status_code == 403

    def test_409_for_draft(self):
        for client in _make_router_client(
            **{"oneclick.royalties.service.get_paid_payout": {"side_effect": service.PayoutStateError("not paid")}}
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/receipt/save", json=self.BODY)
            assert resp.status_code == 409

    def test_500_on_storage_failure(self):
        db = MagicMock()
        db.storage.from_.return_value.upload.side_effect = RuntimeError("bucket down")
        db.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        for client in _make_router_client(
            **{
                "oneclick.royalties.router._get_supabase": {"return_value": db},
                "oneclick.royalties.service.get_paid_payout": {"return_value": _make_payout()},
                "main.verify_user_owns_artist": {"return_value": True},
                "main.verify_user_owns_project": {"return_value": True},
            }
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/receipt/save", json=self.BODY)
            assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Receipt email
# ---------------------------------------------------------------------------


class TestSendPayoutReceiptEmail:
    ARGS = dict(
        recipient_email="alice@example.com",
        payee_name="Alice",
        payer_name="Kenji",
        amount_str="1,234.56 USD",
        paid_at="2026-07-01T01:00:00Z",
        paypal_capture_id="CAPTURE-1",
        receipt_pdf_bytes=b"%PDF-fake",
    )

    def test_returns_none_without_env(self, monkeypatch):
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        assert emails.send_payout_receipt_email(**self.ARGS) is None

    def test_sends_with_base64_attachment(self, monkeypatch):
        monkeypatch.setenv("RESEND_API_KEY", "re_test")
        monkeypatch.setenv("RESEND_FROM_EMAIL", "Msanii <noreply@test>")
        with patch("oneclick.royalties.emails.resend.Emails.send", return_value={"id": "email-1"}) as mock_send:
            result = emails.send_payout_receipt_email(**self.ARGS)
        assert result == {"id": "email-1"}
        payload = mock_send.call_args[0][0]
        assert payload["to"] == ["alice@example.com"]
        assert "1,234.56 USD" in payload["subject"]
        attachment = payload["attachments"][0]
        assert attachment["content_type"] == "application/pdf"
        assert base64.b64decode(attachment["content"]) == b"%PDF-fake"

    def test_send_failure_returns_none(self, monkeypatch):
        monkeypatch.setenv("RESEND_API_KEY", "re_test")
        monkeypatch.setenv("RESEND_FROM_EMAIL", "Msanii <noreply@test>")
        with patch("oneclick.royalties.emails.resend.Emails.send", side_effect=RuntimeError("api down")):
            assert emails.send_payout_receipt_email(**self.ARGS) is None
