"""Tests for /users/welcome — fires signup_completed + identify exactly once per user."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from auth import get_current_user_id
from main import app

TEST_USER_ID = "00000000-0000-0000-0000-000000000099"


def _mock_sb(profile_row: dict, subscription_rows: list | None = None):
    """Build a Supabase mock for the /users/welcome handler.

    Handler reads:
      1. profiles.select(...).eq().maybe_single().execute() — gate + names
      2. subscriptions.select("tier").eq().execute() — plan
      3. tier_overrides.select(...).eq().like().execute() — is_tester
      4. profiles.select("is_admin").eq().limit().execute() — is_user_admin path
    Handler writes:
      5. profiles.update(...).eq().execute() — mark welcome_email_sent_at
    """
    sb = MagicMock()
    subs_rows = subscription_rows if subscription_rows is not None else [{"tier": "free"}]

    def table(name):
        chain = MagicMock()
        if name == "profiles":
            # maybe_single() path returns the profile row (or None when missing)
            chain.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
                data=profile_row
            )
            # is_user_admin → is_db_admin → .select("is_admin").eq().limit().execute()
            chain.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
            # .update(...).eq().execute() — the idempotent mark
            chain.update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])
        elif name == "subscriptions":
            chain.select.return_value.eq.return_value.execute.return_value = MagicMock(data=subs_rows)
        elif name == "tier_overrides":
            chain.select.return_value.eq.return_value.like.return_value.execute.return_value = MagicMock(data=[])
        else:
            chain.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        return chain

    sb.table.side_effect = table
    # Auth admin lookup (used only when sending the welcome email; safe stub).
    sb.auth.admin.get_user_by_id.return_value = MagicMock(user=MagicMock(email=profile_row.get("email")))
    return sb


def test_first_call_fires_signup_completed_and_identify(monkeypatch):
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID

    capture_calls = []
    identify_calls = []
    monkeypatch.setattr(
        "users.router.analytics_capture",
        lambda uid, event, props=None: capture_calls.append((uid, event, props or {})),
    )
    monkeypatch.setattr(
        "users.router.analytics_identify",
        lambda uid, props: identify_calls.append((uid, props)),
    )
    # Welcome email sender is unrelated to analytics — stub it so we don't hit Resend.
    monkeypatch.setattr("users.router.send_welcome_email", lambda *a, **kw: {"id": "stub"})
    monkeypatch.delenv("ADMIN_EMAILS", raising=False)

    sb = _mock_sb(
        profile_row={
            "email": "new@example.com",
            "role": "artist",
            "created_at": "2026-05-16T00:00:00+00:00",
            "welcome_email_sent_at": None,
            "first_name": "Ada",
            "given_name": None,
            "full_name": "Ada Lovelace",
        }
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.post("/users/welcome")
    app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    signup_events = [c for c in capture_calls if c[1] == "signup_completed"]
    assert len(signup_events) == 1
    assert signup_events[0][0] == TEST_USER_ID

    assert len(identify_calls) == 1
    assert identify_calls[0][0] == TEST_USER_ID
    props = identify_calls[0][1]
    assert props["email"] == "new@example.com"
    assert props["plan"] == "free"
    assert props["role"] == "artist"
    assert props["signed_up_at"] == "2026-05-16T00:00:00+00:00"
    assert props["is_admin"] is False
    assert props["is_tester"] is False


def test_second_call_is_idempotent_no_events(monkeypatch):
    """If welcome_email_sent_at is already set, no analytics events fire."""
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID

    capture_calls = []
    identify_calls = []
    monkeypatch.setattr(
        "users.router.analytics_capture",
        lambda uid, event, props=None: capture_calls.append((uid, event, props or {})),
    )
    monkeypatch.setattr(
        "users.router.analytics_identify",
        lambda uid, props: identify_calls.append((uid, props)),
    )
    monkeypatch.setattr("users.router.send_welcome_email", lambda *a, **kw: {"id": "stub"})

    sb = _mock_sb(
        profile_row={
            "email": "old@example.com",
            "role": "artist",
            "created_at": "2026-04-01T00:00:00+00:00",
            "welcome_email_sent_at": "2026-04-02T00:00:00+00:00",  # already set
            "first_name": "Old",
            "given_name": None,
            "full_name": "Old User",
        }
    )
    monkeypatch.setattr("main.get_supabase_client", lambda: sb)

    client = TestClient(app)
    resp = client.post("/users/welcome")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert [c for c in capture_calls if c[1] == "signup_completed"] == []
    assert identify_calls == []
