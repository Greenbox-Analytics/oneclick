"""Tests that AdminService tester-grant mutations fire analytics.identify."""

from unittest.mock import MagicMock

from subscriptions.admin_service import AdminService
from subscriptions.service import EntitlementsService


def _make_service_with_user(monkeypatch, identify_calls, user_id="user-123", email="t@example.com"):
    """Mock sb.auth.admin.list_users() — that's what create_tester_grant uses
    (admin_service.py:~182), NOT sb.table('users').select()."""
    monkeypatch.setattr(
        "subscriptions.admin_service.analytics_identify",
        lambda uid, props: identify_calls.append((uid, dict(props))),
    )
    sb = MagicMock()
    fake_user = MagicMock()
    fake_user.id = user_id
    fake_user.email = email
    sb.auth.admin.list_users.return_value = [fake_user]
    return AdminService(sb, EntitlementsService(sb)), sb


def test_create_tester_grant_calls_identify(monkeypatch):
    identify_calls: list = []
    svc, _ = _make_service_with_user(monkeypatch, identify_calls)

    svc.create_tester_grant(email="t@example.com", expires_at=None, reason="tester")

    assert len(identify_calls) == 1
    uid, props = identify_calls[0]
    assert uid == "user-123"
    assert props["is_tester"] is True
    assert "tester_granted_at" in props
    assert "tester_expires_at" in props


def test_revoke_tester_grant_calls_identify(monkeypatch):
    identify_calls: list = []
    monkeypatch.setattr(
        "subscriptions.admin_service.analytics_identify",
        lambda uid, props: identify_calls.append((uid, dict(props))),
    )
    svc = AdminService(MagicMock(), EntitlementsService(MagicMock()))

    svc.revoke_tester_grant("user-456")

    assert len(identify_calls) == 1
    uid, props = identify_calls[0]
    assert uid == "user-456"
    assert props["is_tester"] is False
    assert props["tester_granted_at"] is None
    assert props["tester_expires_at"] is None


def test_create_tester_grant_swallows_analytics_errors(monkeypatch):
    """Analytics failures must not break the grant."""

    def explode(*a, **kw):
        raise RuntimeError("posthog down")

    monkeypatch.setattr("subscriptions.admin_service.analytics_identify", explode)

    sb = MagicMock()
    fake_user = MagicMock()
    fake_user.id = "user-789"
    fake_user.email = "x@y.z"
    sb.auth.admin.list_users.return_value = [fake_user]
    svc = AdminService(sb, EntitlementsService(sb))

    # Should not raise
    svc.create_tester_grant(email="x@y.z", expires_at=None, reason="tester")
