"""Tests for the PayPal payout flow in oneclick/royalties/service.py + router.

All tests mock the Supabase client and the paypal_client HTTP functions —
no real DB or network calls. format_amount stays real (money correctness).

Critical invariants verified:
  1. Order creation passes the payee's email, the exact formatted amount
     string, and the payout currency; persists paypal_order_id but does NOT
     set payment_method (abandoned orders must not read as PayPal).
  2. Missing email / unsupported currency / non-draft / foreign payout are
     rejected BEFORE any PayPal call.
  3. Crash-window reconciliation: a stored order that is already COMPLETED
     marks the payout paid instead of creating a second order (no double pay).
  4. Capture marks paid only on a COMPLETED capture with the exact expected
     amount + currency; PENDING/DECLINED/mismatch leave the payout draft.
  5. Capture is double-click safe (already-paid short-circuit and
     ORDER_ALREADY_CAPTURED fallback).
  6. Router maps errors to 404 / 409 / 400 / 502.
"""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import service
from oneclick.royalties.paypal_client import PayPalError

USER_ID = "user-aaa"
OTHER_USER_ID = "user-bbb"
PAYEE_ID = "payee-111"
PAYOUT_ID = "payout-xyz"
ORDER_ID = "PAYPAL-ORDER-1"
CAPTURE_ID = "PAYPAL-CAPTURE-1"

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_payee(email="alice@example.com", payout_ccy="USD"):
    return {
        "id": PAYEE_ID,
        "user_id": USER_ID,
        "display_name": "Alice",
        "payout_currency": payout_ccy,
        "normalized_name": "alice",
        "registry_user_id": None,
        "email": email,
    }


def _make_payout(status="draft", total_amount=200.0, pay_currency="USD", paypal_order_id=None, payment_method="manual"):
    return {
        "id": PAYOUT_ID,
        "user_id": USER_ID,
        "payee_id": PAYEE_ID,
        "status": status,
        "pay_currency": pay_currency,
        "total_amount": total_amount,
        "fx_rate_date": "2026-06-23",
        "created_at": "2026-06-23T00:00:00Z",
        "paid_at": None,
        "note": None,
        "breakdown_snapshot": {"projects": []},
        "idempotency_key": None,
        "payment_method": payment_method,
        "paypal_order_id": paypal_order_id,
        "paypal_capture_id": None,
    }


def _completed_order(value="200.00", currency="USD", capture_status="COMPLETED"):
    return {
        "id": ORDER_ID,
        "status": "COMPLETED",
        "purchase_units": [
            {
                "payments": {
                    "captures": [
                        {
                            "id": CAPTURE_ID,
                            "status": capture_status,
                            "amount": {"currency_code": currency, "value": value},
                        }
                    ]
                }
            }
        ],
    }


# ---------------------------------------------------------------------------
# Mock DB — stateful select/update capture (clone of test_royalties_payouts)
# ---------------------------------------------------------------------------


class MockDB:
    def __init__(self, payouts=None, payees=None):
        self.payouts = payouts or []
        self.payees = payees or []
        self.updated_payouts = []  # dicts passed to royalty_payouts.update()

    def table(self, name):
        return _TableProxy(self, name)


class _TableProxy:
    def __init__(self, db, name):
        self._db = db
        self._name = name
        self._filters = {}
        self._pending_update = None

    def select(self, *a, **kw):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def update(self, data):
        self._pending_update = data
        return self

    def execute(self):
        db, name = self._db, self._name
        if self._pending_update is not None:
            if name == "royalty_payouts":
                matched = [p for p in db.payouts if all(p.get(c) == v for c, v in self._filters.items())]
                db.updated_payouts.append(self._pending_update)
                if matched:
                    updated = {**matched[0], **self._pending_update}
                    return MagicMock(data=[updated])
            return MagicMock(data=[])
        rows = db.payouts if name == "royalty_payouts" else db.payees if name == "royalty_payees" else []
        for col, val in self._filters.items():
            rows = [r for r in rows if r.get(col) == val]
        return MagicMock(data=rows)


# ---------------------------------------------------------------------------
# create_paypal_order_for_payout
# ---------------------------------------------------------------------------


class TestCreatePaypalOrder:
    def test_happy_path(self):
        db = MockDB(payouts=[_make_payout(total_amount=150.5)], payees=[_make_payee()])
        with patch.object(service.paypal_client, "create_order", return_value={"id": ORDER_ID}) as mock_create:
            result = service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)

        assert result == {"paypal_order_id": ORDER_ID}
        mock_create.assert_called_once_with(
            payee_email="alice@example.com",
            amount_value="150.50",
            currency="USD",
            reference_id=PAYOUT_ID,
            description="Royalty payout — Alice",
        )
        # paypal_order_id persisted; payment_method untouched until capture
        assert db.updated_payouts == [{"paypal_order_id": ORDER_ID}]

    def test_note_used_as_description(self):
        payout = _make_payout()
        payout["note"] = "Q1 2026 royalties"
        db = MockDB(payouts=[payout], payees=[_make_payee()])
        with patch.object(service.paypal_client, "create_order", return_value={"id": ORDER_ID}) as mock_create:
            service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert mock_create.call_args.kwargs["description"] == "Q1 2026 royalties"

    def test_missing_email_rejected_before_paypal_call(self):
        db = MockDB(payouts=[_make_payout()], payees=[_make_payee(email=None)])
        with (
            patch.object(service.paypal_client, "create_order") as mock_create,
            pytest.raises(ValueError, match="no email"),
        ):
            service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        mock_create.assert_not_called()
        assert db.updated_payouts == []

    def test_blank_email_rejected(self):
        db = MockDB(payouts=[_make_payout()], payees=[_make_payee(email="   ")])
        with pytest.raises(ValueError, match="no email"):
            service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)

    def test_unsupported_currency_rejected(self):
        db = MockDB(payouts=[_make_payout(pay_currency="NGN")], payees=[_make_payee()])
        with patch.object(service.paypal_client, "create_order") as mock_create, pytest.raises(ValueError, match="NGN"):
            service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        mock_create.assert_not_called()

    def test_non_draft_rejected(self):
        db = MockDB(payouts=[_make_payout(status="paid")], payees=[_make_payee()])
        with pytest.raises(service.PayoutStateError):
            service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)

    def test_foreign_payout_rejected(self):
        db = MockDB(payouts=[_make_payout()], payees=[_make_payee()])
        with patch.object(service.paypal_client, "create_order") as mock_create, pytest.raises(PermissionError):
            service.create_paypal_order_for_payout(db, OTHER_USER_ID, PAYOUT_ID)
        mock_create.assert_not_called()

    def test_crash_window_completed_order_marks_paid(self):
        """A stored order that already COMPLETED must mark the payout paid, not re-charge."""
        db = MockDB(payouts=[_make_payout(paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with (
            patch.object(service.paypal_client, "get_order", return_value=_completed_order()),
            patch.object(service.paypal_client, "create_order") as mock_create,
            pytest.raises(service.PayoutStateError, match="already paid"),
        ):
            service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)

        mock_create.assert_not_called()
        assert len(db.updated_payouts) == 1
        update = db.updated_payouts[0]
        assert update["status"] == "paid"
        assert update["payment_method"] == "paypal"
        assert update["paypal_capture_id"] == CAPTURE_ID

    def test_stale_uncompleted_order_replaced(self):
        """An abandoned (non-completed) order is replaced by a fresh one."""
        db = MockDB(payouts=[_make_payout(paypal_order_id="OLD-ORDER")], payees=[_make_payee()])
        with (
            patch.object(service.paypal_client, "get_order", return_value={"id": "OLD-ORDER", "status": "CREATED"}),
            patch.object(service.paypal_client, "create_order", return_value={"id": ORDER_ID}),
        ):
            result = service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert result == {"paypal_order_id": ORDER_ID}

    def test_expired_order_lookup_failure_replaced(self):
        """get_order failing on the stored id (expired/voided) still allows a fresh order."""
        db = MockDB(payouts=[_make_payout(paypal_order_id="OLD-ORDER")], payees=[_make_payee()])
        with (
            patch.object(service.paypal_client, "get_order", side_effect=PayPalError("gone", status_code=404)),
            patch.object(service.paypal_client, "create_order", return_value={"id": ORDER_ID}),
        ):
            result = service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert result == {"paypal_order_id": ORDER_ID}

    def test_jpy_amount_formatted_whole_number(self):
        db = MockDB(payouts=[_make_payout(total_amount=1234.5, pay_currency="JPY")], payees=[_make_payee()])
        with patch.object(service.paypal_client, "create_order", return_value={"id": ORDER_ID}) as mock_create:
            service.create_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert mock_create.call_args.kwargs["amount_value"] == "1235"
        assert mock_create.call_args.kwargs["currency"] == "JPY"


# ---------------------------------------------------------------------------
# capture_paypal_order_for_payout
# ---------------------------------------------------------------------------


class TestCapturePaypalOrder:
    def test_happy_path(self):
        db = MockDB(payouts=[_make_payout(paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with patch.object(service.paypal_client, "capture_order", return_value=_completed_order()) as mock_capture:
            updated = service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)

        mock_capture.assert_called_once_with(ORDER_ID)
        assert len(db.updated_payouts) == 1
        update = db.updated_payouts[0]
        assert update["status"] == "paid"
        assert update["paid_at"] == "now()"
        assert update["payment_method"] == "paypal"
        assert update["paypal_capture_id"] == CAPTURE_ID
        assert updated["status"] == "paid"

    def test_already_paid_via_paypal_is_idempotent(self):
        paid = _make_payout(status="paid", paypal_order_id=ORDER_ID, payment_method="paypal")
        db = MockDB(payouts=[paid], payees=[_make_payee()])
        with patch.object(service.paypal_client, "capture_order") as mock_capture:
            result = service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        mock_capture.assert_not_called()
        assert result == paid
        assert db.updated_payouts == []

    def test_already_paid_manually_rejected(self):
        db = MockDB(payouts=[_make_payout(status="paid", paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with pytest.raises(service.PayoutStateError, match="manually"):
            service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)

    def test_no_order_started_rejected(self):
        db = MockDB(payouts=[_make_payout()], payees=[_make_payee()])
        with pytest.raises(ValueError, match="No PayPal payment"):
            service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)

    def test_foreign_payout_rejected(self):
        db = MockDB(payouts=[_make_payout(paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with pytest.raises(PermissionError):
            service.capture_paypal_order_for_payout(db, OTHER_USER_ID, PAYOUT_ID)

    def test_order_already_captured_falls_back_to_get_order(self):
        db = MockDB(payouts=[_make_payout(paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with (
            patch.object(
                service.paypal_client,
                "capture_order",
                side_effect=PayPalError("dup", status_code=422, issue="ORDER_ALREADY_CAPTURED"),
            ),
            patch.object(service.paypal_client, "get_order", return_value=_completed_order()),
        ):
            updated = service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert updated["status"] == "paid"
        assert db.updated_payouts[0]["paypal_capture_id"] == CAPTURE_ID

    def test_other_paypal_errors_propagate(self):
        db = MockDB(payouts=[_make_payout(paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with (
            patch.object(service.paypal_client, "capture_order", side_effect=PayPalError("boom", status_code=500)),
            pytest.raises(PayPalError),
        ):
            service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert db.updated_payouts == []

    @pytest.mark.parametrize("capture_status", ["PENDING", "DECLINED", "FAILED"])
    def test_incomplete_capture_stays_draft(self, capture_status):
        db = MockDB(payouts=[_make_payout(paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with (
            patch.object(
                service.paypal_client, "capture_order", return_value=_completed_order(capture_status=capture_status)
            ),
            pytest.raises(PayPalError),
        ):
            service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert db.updated_payouts == []  # no DB write — payout stays draft

    @pytest.mark.parametrize(
        ("value", "currency"),
        [
            ("199.99", "USD"),  # wrong amount
            ("200.00", "EUR"),  # wrong currency
        ],
    )
    def test_amount_mismatch_stays_draft(self, value, currency):
        db = MockDB(payouts=[_make_payout(paypal_order_id=ORDER_ID)], payees=[_make_payee()])
        with (
            patch.object(
                service.paypal_client, "capture_order", return_value=_completed_order(value=value, currency=currency)
            ),
            pytest.raises(PayPalError, match="mismatch"),
        ):
            service.capture_paypal_order_for_payout(db, USER_ID, PAYOUT_ID)
        assert db.updated_payouts == []


# ---------------------------------------------------------------------------
# Router HTTP layer — error mapping
# ---------------------------------------------------------------------------


def _make_router_client(service_fn, side_effect=None, return_value=None):
    """Minimal app with the royalties router; auth/gating/supabase mocked."""
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
        patch(service_fn, side_effect=side_effect, return_value=return_value if side_effect is None else None),
    ]
    for p in patchers:
        p.start()

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    for p in patchers:
        p.stop()


CREATE_FN = "oneclick.royalties.service.create_paypal_order_for_payout"
CAPTURE_FN = "oneclick.royalties.service.capture_paypal_order_for_payout"


class TestPaypalRouter:
    def test_order_200(self):
        for client in _make_router_client(CREATE_FN, return_value={"paypal_order_id": ORDER_ID}):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/order")
            assert resp.status_code == 200
            assert resp.json() == {"paypal_order_id": ORDER_ID}

    def test_order_404_on_permission_error(self):
        for client in _make_router_client(CREATE_FN, side_effect=PermissionError("not yours")):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/order")
            assert resp.status_code == 404

    def test_order_409_on_state_error(self):
        for client in _make_router_client(CREATE_FN, side_effect=service.PayoutStateError("already paid")):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/order")
            assert resp.status_code == 409

    def test_order_400_on_value_error(self):
        for client in _make_router_client(CREATE_FN, side_effect=ValueError("This payee has no email address yet.")):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/order")
            assert resp.status_code == 400
            assert "email" in resp.json()["detail"]

    def test_order_502_on_paypal_error(self):
        for client in _make_router_client(CREATE_FN, side_effect=PayPalError("internal issue details")):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/order")
            assert resp.status_code == 502
            # Raw PayPal error details must not leak to the client
            assert "internal issue details" not in resp.json()["detail"]

    def test_capture_200(self):
        paid = _make_payout(status="paid", paypal_order_id=ORDER_ID, payment_method="paypal")
        for client in _make_router_client(CAPTURE_FN, return_value=paid):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/capture")
            assert resp.status_code == 200
            assert resp.json()["status"] == "paid"

    def test_capture_404_on_permission_error(self):
        for client in _make_router_client(CAPTURE_FN, side_effect=PermissionError("not yours")):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/capture")
            assert resp.status_code == 404

    def test_capture_502_on_paypal_error(self):
        for client in _make_router_client(CAPTURE_FN, side_effect=PayPalError("PENDING")):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/capture")
            assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Receipt email scheduling on capture
# ---------------------------------------------------------------------------


def _make_receipt_db(payee_email="alice@example.com", payer_name="Kenji"):
    """Supabase mock whose royalty_payees / profiles selects return real rows."""

    def _table(name):
        proxy = MagicMock()
        proxy.select.return_value = proxy
        proxy.eq.return_value = proxy
        if name == "royalty_payees":
            rows = [{"email": payee_email, "display_name": "Alice"}]
        elif name == "profiles":
            rows = [{"full_name": payer_name}]
        else:
            rows = []
        proxy.execute.return_value = MagicMock(data=rows)
        return proxy

    db = MagicMock()
    db.table.side_effect = _table
    return db


def _make_capture_client(db, capture_side_effect=None, capture_return=None, extra_patchers=()):
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
        patch("oneclick.royalties.router._get_supabase", return_value=db),
        patch(
            CAPTURE_FN,
            side_effect=capture_side_effect,
            return_value=capture_return if capture_side_effect is None else None,
        ),
        *extra_patchers,
    ]
    for p in patchers:
        p.start()

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    for p in patchers:
        p.stop()


class TestCaptureReceiptEmail:
    def _paid_payout(self):
        return _make_payout(status="paid", paypal_order_id=ORDER_ID, payment_method="paypal")

    def test_email_scheduled_on_successful_capture(self):
        email_mock = MagicMock()
        for client in _make_capture_client(
            _make_receipt_db(),
            capture_return=self._paid_payout(),
            extra_patchers=(patch("oneclick.royalties.router._send_receipt_email_background", email_mock),),
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/capture")
            assert resp.status_code == 200
        # TestClient runs background tasks synchronously after the response.
        email_mock.assert_called_once()
        assert email_mock.call_args.kwargs["payee_email"] == "alice@example.com"
        assert email_mock.call_args.kwargs["payee_name"] == "Alice"
        assert email_mock.call_args.kwargs["payer_name"] == "Kenji"

    def test_no_email_when_payee_has_none(self):
        email_mock = MagicMock()
        for client in _make_capture_client(
            _make_receipt_db(payee_email=None),
            capture_return=self._paid_payout(),
            extra_patchers=(patch("oneclick.royalties.router._send_receipt_email_background", email_mock),),
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/capture")
            assert resp.status_code == 200
        email_mock.assert_not_called()

    def test_no_email_on_capture_failure(self):
        email_mock = MagicMock()
        for client in _make_capture_client(
            _make_receipt_db(),
            capture_side_effect=PayPalError("PENDING"),
            extra_patchers=(patch("oneclick.royalties.router._send_receipt_email_background", email_mock),),
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/capture")
            assert resp.status_code == 502
        email_mock.assert_not_called()

    def test_email_send_failure_does_not_break_capture(self):
        # Patch the underlying send (not the background wrapper) so the real
        # swallow inside _send_receipt_email_background is exercised.
        for client in _make_capture_client(
            _make_receipt_db(),
            capture_return=self._paid_payout(),
            extra_patchers=(
                patch(
                    "oneclick.royalties.router.emails.send_payout_receipt_email",
                    side_effect=RuntimeError("resend down"),
                ),
            ),
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/paypal/capture")
            assert resp.status_code == 200
            assert resp.json()["status"] == "paid"
