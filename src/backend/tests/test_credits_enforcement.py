# src/backend/tests/test_credits_enforcement.py
"""gated_credits: grant minting, structured 402s, legacy fallback."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import subscriptions.enforcement as enforcement
from subscriptions.models import CheckResult, CreditAction, CreditCheckResult

TEST_USER = "00000000-0000-0000-0000-000000000001"


@pytest.fixture()
def fake_service(monkeypatch):
    svc = MagicMock()
    monkeypatch.setattr(enforcement, "_service", lambda: svc)
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    return svc


class TestGatedCredits:
    def test_allowed_returns_grant(self, fake_service):
        fake_service.check_credits.return_value = CreditCheckResult(allowed=True, price=3)
        grant = enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert grant.enabled and grant.price == 3 and grant.kind == "debit"
        assert len(grant.request_id) == 36  # uuid4

    def test_fresh_request_id_per_invocation(self, fake_service):
        fake_service.check_credits.return_value = CreditCheckResult(allowed=True, price=3)
        g1 = enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        g2 = enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert g1.request_id != g2.request_id

    def test_overage_allowed_sets_kind(self, fake_service):
        fake_service.check_credits.return_value = CreditCheckResult(allowed=True, price=21, use_overage=True)
        grant = enforcement.gated_credits(TEST_USER, CreditAction.ONECLICK_RUN)
        assert grant.kind == "overage_debit" and grant.enabled

    def test_zero_price_grant_disabled(self, fake_service):
        # Admin short-circuit / degraded-open return price 0 → debit must no-op.
        fake_service.check_credits.return_value = CreditCheckResult(allowed=True, price=0)
        grant = enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert grant.enabled is False

    def test_denied_raises_402_with_structured_detail(self, fake_service):
        reset_date = datetime(2026, 8, 1, tzinfo=UTC)
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False,
            price=3,
            reason="You've used this month's credits.",
            upgrade_required=True,
            reset_date=reset_date,
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert exc.value.status_code == 402
        assert exc.value.detail["upgradeRequired"] is True
        assert exc.value.detail["overageAvailable"] is False
        assert "credits" in exc.value.detail["reason"]
        assert exc.value.detail["price"] == 3
        assert exc.value.detail["resetDate"] == reset_date.isoformat()

    def test_denied_paid_carries_overage_flag(self, fake_service):
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False,
            price=21,
            reason="You've used your included credits for this month.",
            overage_available=True,
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ONECLICK_RUN)
        assert exc.value.detail["overageAvailable"] is True
        assert exc.value.detail["upgradeRequired"] is False

    def test_disabled_falls_back_to_legacy_gate(self, fake_service, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        fake_service.can.return_value = CheckResult(allowed=True, reason=None, upgrade_required=False)
        grant = enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert grant.enabled is False  # no debit under legacy mode
        fake_service.can.assert_called_once()
        fake_service.check_credits.assert_not_called()

    def test_disabled_legacy_denial_still_raises(self, fake_service, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        fake_service.can.return_value = CheckResult(
            allowed=False, reason="Zoe is a Pro feature.", upgrade_required=True
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert exc.value.status_code == 402


def test_free_credit_grant_is_disabled_and_priced_zero():
    from subscriptions.enforcement import free_credit_grant
    from subscriptions.models import CreditAction

    g = free_credit_grant(CreditAction.ZOE_MESSAGE)
    assert g.enabled is False
    assert g.price == 0
    assert g.kind == "debit"
    assert g.action == "zoe_message"
    assert g.request_id == ""
    # Licensing Phase B: free grants carry no targeted wallet.
    assert g.wallet_id is None


# ---------------------------------------------------------------------------
# Licensing Phase B — Task 6: seat wall threads managedByOrg + requestUrl, the
# grant carries the seat wallet_id, and paywall_blocked carries managed_by_org.
# ---------------------------------------------------------------------------


class TestSeatWall:
    def test_allowed_seat_grant_carries_wallet_id(self, fake_service):
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=True, price=3, wallet_id="wallet-seat", managed_by_org=True
        )
        grant = enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert grant.enabled and grant.price == 3
        assert grant.wallet_id == "wallet-seat"

    def test_seat_dry_402_detail_has_managed_by_org_and_request_url(self, fake_service):
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False,
            price=3,
            managed_by_org=True,
            reason="You've used the credits your organization allocated. Ask your admin for more.",
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        detail = exc.value.detail
        assert exc.value.status_code == 402
        assert detail["managedByOrg"] is True
        assert detail["requestUrl"] == "/organization"
        # Seat wall has NO overage / upgrade path and no reset (rule 8).
        assert detail["upgradeRequired"] is False
        assert detail["overageAvailable"] is False
        assert detail["resetDate"] is None
        assert "organization" in detail["reason"].lower()

    def test_personal_denial_has_no_managed_by_org_keys(self, fake_service):
        """Regression: a personal (non-org) denial keeps the pre-licensing detail
        shape — no managedByOrg / requestUrl leak."""
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False, price=3, reason="You've used this month's credits.", upgrade_required=True
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert "managedByOrg" not in exc.value.detail
        assert "requestUrl" not in exc.value.detail

    def test_paywall_blocked_carries_managed_by_org(self, fake_service, monkeypatch):
        captured = {}

        def _capture(uid, event, props):
            captured["props"] = props

        monkeypatch.setattr(enforcement, "analytics_capture", _capture)
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False, price=3, managed_by_org=True, reason="Ask your admin for more."
        )
        with pytest.raises(HTTPException):
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert captured["props"]["managed_by_org"] is True

    def test_personal_paywall_blocked_managed_by_org_false(self, fake_service, monkeypatch):
        captured = {}

        def _capture(uid, event, props):
            captured["props"] = props

        monkeypatch.setattr(enforcement, "analytics_capture", _capture)
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False, price=3, upgrade_required=True, reason="You've used this month's credits."
        )
        with pytest.raises(HTTPException):
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        assert captured["props"]["managed_by_org"] is False

    def test_seat_wall_fires_paywall_blocked_and_402_detail_together(self, fake_service, monkeypatch):
        """Combined regression (Task 14 event audit): a single seat-context
        denial must produce BOTH signals from the SAME call — the analytics
        event (`paywall_blocked` with `managed_by_org: True` in properties)
        and the structured 402 (`managedByOrg`/`requestUrl`). The tests above
        check each half in isolation against separate calls; this pins them
        together so a future refactor can't silently drop one side."""
        captured = {}

        def _capture(uid, event, props):
            captured["event"] = event
            captured["props"] = props

        monkeypatch.setattr(enforcement, "analytics_capture", _capture)
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False,
            price=3,
            managed_by_org=True,
            reason="You've used the credits your organization allocated. Ask your admin for more.",
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)

        assert captured["event"] == "paywall_blocked"
        assert captured["props"]["managed_by_org"] is True

        detail = exc.value.detail
        assert detail["managedByOrg"] is True
        assert detail["requestUrl"] == "/organization"


# ---------------------------------------------------------------------------
# Licensing Phase C — Task 6: owner-aware dry-seat wall. The 402 detail maps
# CreditCheckResult.owner_can_unlink/project_id/project_name to ownerCanUnlink/
# projectId/projectName, CO-OCCURRING with managedByOrg/requestUrl (round 4).
# ---------------------------------------------------------------------------


class TestOwnerAwareDrySeatWall:
    def test_owner_dry_seat_402_carries_unlink_alongside_managed_by_org(self, fake_service):
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False,
            price=3,
            managed_by_org=True,
            owner_can_unlink=True,
            project_id="proj-123",
            reason=(
                "You've used the credits your organization allocated. Ask your admin for more. "
                "Or unlink this project in its settings to use your own plan here."
            ),
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        detail = exc.value.detail
        # Owner CTA present, projectId REQUIRED alongside the flag (round 5).
        assert detail["ownerCanUnlink"] is True
        assert detail["projectId"] == "proj-123"
        assert "projectName" not in detail  # absent when the read didn't carry it
        # CO-OCCURS with the buy/request affordance — never mutually exclusive.
        assert detail["managedByOrg"] is True
        assert detail["requestUrl"] == "/organization"

    def test_seat_wall_without_owner_has_no_owner_keys(self, fake_service):
        """A non-owner seat wall keeps the plain managed-by-org shape — no owner
        keys leak in."""
        fake_service.check_credits.return_value = CreditCheckResult(
            allowed=False,
            price=3,
            managed_by_org=True,
            reason="You've used the credits your organization allocated. Ask your admin for more.",
        )
        with pytest.raises(HTTPException) as exc:
            enforcement.gated_credits(TEST_USER, CreditAction.ZOE_MESSAGE)
        detail = exc.value.detail
        assert "ownerCanUnlink" not in detail
        assert "projectId" not in detail
        assert detail["managedByOrg"] is True


# ---------------------------------------------------------------------------
# Licensing Phase C — Task 7: gated_create / gated_upload thread the optional
# resource_project_id into can() so caps derivation can fire (rule 9). The kwarg
# defaults None → byte-identical to today.
# ---------------------------------------------------------------------------


class TestResourceProjectIdThreading:
    def _svc(self, monkeypatch):
        svc = MagicMock()
        svc.can.return_value = CheckResult(allowed=True, reason=None, upgrade_required=False)
        monkeypatch.setattr(enforcement, "_service", lambda: svc)
        return svc

    def test_gated_create_threads_resource_project_id(self, monkeypatch):
        svc = self._svc(monkeypatch)
        enforcement.gated_create(TEST_USER, "work", 5, resource_project_id="proj-1")
        _, kwargs = svc.can.call_args
        assert kwargs["resource_project_id"] == "proj-1"
        assert kwargs["current_count"] == 5

    def test_gated_create_none_call_is_byte_identical(self, monkeypatch):
        # When no project id is supplied, the kwarg is omitted entirely so the
        # can() call matches pre-Phase-C exactly (existing exact-arg tests unmodified).
        svc = self._svc(monkeypatch)
        enforcement.gated_create(TEST_USER, "work", 5)
        _, kwargs = svc.can.call_args
        assert "resource_project_id" not in kwargs
        assert kwargs["current_count"] == 5

    def test_gated_create_unknown_resource_never_calls_can(self, monkeypatch):
        """A typo'd resource still no-ops before can() even with a project id."""
        svc = self._svc(monkeypatch)
        enforcement.gated_create(TEST_USER, "bogus", 5, resource_project_id="proj-1")
        svc.can.assert_not_called()

    def test_gated_upload_threads_project_id_and_host(self, monkeypatch):
        svc = self._svc(monkeypatch)
        enforcement.gated_upload(TEST_USER, size=100, host_user_id=TEST_USER, resource_project_id="proj-1")
        _, kwargs = svc.can.call_args
        assert kwargs["resource_project_id"] == "proj-1"
        assert kwargs["host_user_id"] == TEST_USER
        assert kwargs["size"] == 100

    def test_gated_upload_none_call_is_byte_identical(self, monkeypatch):
        # No project id → kwarg omitted, byte-identical to pre-Phase-C.
        svc = self._svc(monkeypatch)
        enforcement.gated_upload(TEST_USER, size=100)
        _, kwargs = svc.can.call_args
        assert "resource_project_id" not in kwargs
        assert kwargs["host_user_id"] is None
        assert kwargs["size"] == 100
