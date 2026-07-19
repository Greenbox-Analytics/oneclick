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
