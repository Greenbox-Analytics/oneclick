"""Tests for GET /me/analytics-context."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from auth import get_current_user_email, get_current_user_id
from main import app

TEST_USER_ID = "00000000-0000-0000-0000-000000000001"
TEST_EMAIL = "tester@example.com"


def _override_user(user_id: str = TEST_USER_ID, email: str = TEST_EMAIL):
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_current_user_email] = lambda: email


def _clear_overrides():
    app.dependency_overrides.clear()


def _mock_supabase_chain(
    tier_overrides_rows=None,
    profile_row=None,
    subscription_rows=None,
    auth_created_at: str | None = "2026-01-01T00:00:00+00:00",
):
    """Build a minimal Supabase client mock for the analytics-context handler.

    Reads the handler now makes:
      1. tier_overrides (.eq().like().execute())              -> is_tester
      2. profiles (.eq().maybe_single().execute())            -> role (tolerant — None if column missing)
         AND (.eq().limit().execute())                        -> is_admin check (profiles.is_admin)
      3. subscriptions (.eq().execute())                      -> plan (from .tier)
      4. auth.admin.get_user_by_id(user_id)                   -> signed_up_at (auth.users.created_at)

    `profile_row` populates the `.maybe_single()` result; `.role` is the only
    field the handler reads from it (created_at + email come from auth.users,
    not profiles, in this schema).
    """
    tier_exec = MagicMock(data=tier_overrides_rows or [])
    profile_exec = MagicMock(data=profile_row or {})
    subs_exec = MagicMock(data=subscription_rows or [])

    def table(name):
        chain = MagicMock()
        if name == "tier_overrides":
            chain.select.return_value.eq.return_value.like.return_value.execute.return_value = tier_exec
        elif name == "profiles":
            # Tolerant role read: maybe_single() (no exception on missing row)
            chain.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = profile_exec
            # is_user_admin → is_db_admin → profiles.select("is_admin").eq().limit(1).execute()
            chain.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        elif name == "subscriptions":
            chain.select.return_value.eq.return_value.execute.return_value = subs_exec
        return chain

    client = MagicMock()
    client.table.side_effect = table

    # auth.admin.get_user_by_id(user_id) → returns a user object with .created_at
    auth_user = MagicMock()
    auth_user.user = MagicMock(created_at=auth_created_at)
    client.auth.admin.get_user_by_id = MagicMock(return_value=auth_user)
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
        profile_row={"role": "artist"},
        subscription_rows=[{"tier": "free"}],
        auth_created_at="2026-01-01T00:00:00+00:00",
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
    # Override with the JWT email we want is_user_admin() to see.
    _override_user(email="boss@example.com")
    monkeypatch.setenv("ADMIN_EMAILS", "boss@example.com, other@example.com")
    sb = _mock_supabase_chain(
        profile_row={"role": "manager"},
        subscription_rows=[{"tier": "pro"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.get("/me/analytics-context")
    _clear_overrides()

    body = resp.json()
    assert body["is_admin"] is True
    assert body["plan"] == "pro"
    assert body["email"] == "boss@example.com"  # echoes the JWT email


def test_is_admin_false_when_email_not_in_admin_emails(monkeypatch):
    _override_user(email="other@example.com")
    monkeypatch.setenv("ADMIN_EMAILS", "boss@example.com")
    sb = _mock_supabase_chain(
        profile_row={"role": "artist"},
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
        profile_row={"role": "artist"},
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
        profile_row={"role": "manager"},
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
        profile_row={"role": "artist"},
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
        profile_row={"role": "artist"},
        subscription_rows=[{"tier": "free"}],
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    identify_calls = []
    monkeypatch.setattr("analytics.identify", lambda *a, **kw: identify_calls.append((a, kw)))

    client = TestClient(app)
    client.get("/me/analytics-context")
    _clear_overrides()

    assert identify_calls == []
