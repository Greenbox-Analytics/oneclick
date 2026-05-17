"""Tests for GET /me/analytics-context."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from auth import get_current_user_id
from main import app

TEST_USER_ID = "00000000-0000-0000-0000-000000000001"


def _override_user(user_id: str = TEST_USER_ID):
    app.dependency_overrides[get_current_user_id] = lambda: user_id


def _clear_overrides():
    app.dependency_overrides.clear()


def _mock_supabase_chain(tier_overrides_rows=None, profile_row=None, subscription_rows=None):
    """Build a minimal Supabase client mock for the analytics-context handler.

    Handler runs three reads:
      1. tier_overrides (.eq().like().execute()) -> list  -> is_tester
      2. profiles (.eq().single().execute()) -> row -> role, email, created_at
         AND (.eq().limit().execute()) -> list -> is_admin check
      3. subscriptions (.eq().execute()) -> list -> plan (from .tier)
    """
    tier_exec = MagicMock(data=tier_overrides_rows or [])
    profile_exec = MagicMock(data=profile_row or {})
    subs_exec = MagicMock(data=subscription_rows or [])

    def table(name):
        chain = MagicMock()
        if name == "tier_overrides":
            # Handler now uses: .table().select().eq().like().execute()
            chain.select.return_value.eq.return_value.like.return_value.execute.return_value = tier_exec
        elif name == "profiles":
            chain.select.return_value.eq.return_value.single.return_value.execute.return_value = profile_exec
            # is_user_admin → is_db_admin → profiles.select("is_admin").eq().limit(1).execute()
            chain.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        elif name == "subscriptions":
            chain.select.return_value.eq.return_value.execute.return_value = subs_exec
        return chain

    client = MagicMock()
    client.table.side_effect = table
    return client


def test_returns_unauthenticated_when_no_user(monkeypatch):
    # Don't override get_current_user_id — TestClient should hit the real dependency
    # which raises 401/403 when no bearer token is provided.
    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    assert resp.status_code in (401, 403)


def test_is_tester_true_when_active_grant(monkeypatch):
    _override_user()
    future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
    granted = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    monkeypatch.delenv("ADMIN_EMAILS", raising=False)
    sb = _mock_supabase_chain(
        tier_overrides_rows=[{"reason": "tester", "granted_at": granted, "expires_at": future}],
        profile_row={"role": "artist", "email": "tester@example.com", "created_at": "2026-01-01T00:00:00+00:00"},
        subscription_rows=[{"tier": "free"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    _clear_overrides()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_tester"] is True
    assert body["is_admin"] is False
    assert body["tester_granted_at"] == granted
    assert body["tester_expires_at"] == future
    assert body["plan"] == "free"
    assert body["role"] == "artist"
    assert body["email"] == "tester@example.com"
    assert body["signed_up_at"] == "2026-01-01T00:00:00+00:00"


def test_is_admin_true_when_email_in_admin_emails(monkeypatch):
    _override_user()
    monkeypatch.setenv("ADMIN_EMAILS", "boss@example.com, other@example.com")
    sb = _mock_supabase_chain(
        profile_row={"role": "manager", "email": "boss@example.com", "created_at": "2026-01-01T00:00:00+00:00"},
        subscription_rows=[{"tier": "pro"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    _clear_overrides()

    body = resp.json()
    assert body["is_admin"] is True
    assert body["plan"] == "pro"


def test_is_admin_false_when_email_not_in_admin_emails(monkeypatch):
    _override_user()
    monkeypatch.setenv("ADMIN_EMAILS", "boss@example.com")
    sb = _mock_supabase_chain(
        profile_row={"role": "artist", "email": "other@example.com", "created_at": "2026-01-01T00:00:00+00:00"},
        subscription_rows=[{"tier": "free"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    _clear_overrides()

    assert resp.json()["is_admin"] is False


def test_is_tester_false_when_grant_expired(monkeypatch):
    _override_user()
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    sb = _mock_supabase_chain(
        tier_overrides_rows=[{"reason": "tester", "granted_at": past, "expires_at": past}],
        profile_row={"role": "artist", "email": "x@y.z", "created_at": past},
        subscription_rows=[{"tier": "free"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    _clear_overrides()

    assert resp.json()["is_tester"] is False


def test_is_tester_false_when_no_override(monkeypatch):
    _override_user()
    sb = _mock_supabase_chain(
        tier_overrides_rows=[],
        profile_row={"role": "manager", "email": "p@q.r", "created_at": "2025-12-01T00:00:00+00:00"},
        subscription_rows=[{"tier": "pro"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    _clear_overrides()

    body = resp.json()
    assert body["is_tester"] is False
    assert body["tester_granted_at"] is None
    assert body["tester_expires_at"] is None
    assert body["plan"] == "pro"


def test_plan_defaults_to_free_when_no_subscriptions_row(monkeypatch):
    _override_user()
    sb = _mock_supabase_chain(
        profile_row={"role": "artist", "email": "n@new.com", "created_at": "2026-05-01T00:00:00+00:00"},
        subscription_rows=[],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    _clear_overrides()

    assert resp.json()["plan"] == "free"


def test_endpoint_does_not_call_identify(monkeypatch):
    """GETs must be side-effect-free — verify identify() is never called."""
    _override_user()
    sb = _mock_supabase_chain(
        profile_row={"role": "artist", "email": "a@b.c", "created_at": "2026-01-01T00:00:00+00:00"},
        subscription_rows=[{"tier": "free"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    identify_calls = []
    monkeypatch.setattr("analytics.identify", lambda *a, **kw: identify_calls.append((a, kw)))

    client = TestClient(app)
    client.get("/me/analytics-context")
    _clear_overrides()

    assert identify_calls == []
