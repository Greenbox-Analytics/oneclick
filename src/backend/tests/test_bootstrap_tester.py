"""Tests for POST /me/bootstrap-tester — auto-grants tester from TESTER_EMAILS env."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with mocked auth, ANALYTICS off."""
    from fastapi.testclient import TestClient

    from tests.conftest import TEST_USER_ID

    monkeypatch.delenv("POSTHOG_ENABLED", raising=False)  # analytics no-op

    import main
    from auth import get_current_user_email, get_current_user_id

    main.app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    main.app.dependency_overrides[get_current_user_email] = lambda: "tester@example.com"
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _wire_sb(existing_tester_rows: list[dict]):
    """Build a supabase mock: select tier_overrides returns given rows; upsert is captured."""
    sb = MagicMock()
    upserted = {}

    def _table(name):
        b = MagicMock()
        b.select.return_value = b
        b.eq.return_value = b
        b.like.return_value = b
        b.execute.return_value = MagicMock(data=existing_tester_rows if name == "tier_overrides" else [])

        def _upsert(payload, **kwargs):
            upserted["payload"] = payload
            upserted["on_conflict"] = kwargs.get("on_conflict")
            ub = MagicMock()
            ub.execute.return_value = MagicMock(data=[payload])
            return ub

        b.upsert.side_effect = _upsert
        return b

    sb.table.side_effect = _table
    return sb, upserted


class TestBootstrapTester:
    def test_grants_when_email_in_env_and_no_existing_row(self, client, monkeypatch):
        import main

        monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
        sb, upserted = _wire_sb(existing_tester_rows=[])
        monkeypatch.setattr(main, "get_supabase_client", lambda: sb)

        resp = client.post("/me/bootstrap-tester")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"granted": True, "source": "env"}
        assert upserted["payload"]["reason"] == "tester_env"
        assert upserted["payload"]["max_oneclick_runs_per_month"] == -1
        assert upserted["on_conflict"] == "user_id"

    def test_no_grant_when_email_not_in_env(self, client, monkeypatch):
        import main

        monkeypatch.setenv("TESTER_EMAILS", "someone@else.com")
        sb, upserted = _wire_sb(existing_tester_rows=[])
        monkeypatch.setattr(main, "get_supabase_client", lambda: sb)

        resp = client.post("/me/bootstrap-tester")
        assert resp.status_code == 200
        assert resp.json() == {"granted": False, "reason": "not_in_allowlist"}
        assert "payload" not in upserted

    def test_no_grant_when_already_tester(self, client, monkeypatch):
        import main

        monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
        existing = [{"reason": "tester", "expires_at": None, "granted_at": "2026-04-01T00:00:00Z"}]
        sb, upserted = _wire_sb(existing_tester_rows=existing)
        monkeypatch.setattr(main, "get_supabase_client", lambda: sb)

        resp = client.post("/me/bootstrap-tester")
        assert resp.status_code == 200
        assert resp.json() == {"granted": False, "reason": "already_tester"}
        assert "payload" not in upserted

    def test_empty_env_no_grant(self, client, monkeypatch):
        import main

        monkeypatch.delenv("TESTER_EMAILS", raising=False)
        sb, upserted = _wire_sb(existing_tester_rows=[])
        monkeypatch.setattr(main, "get_supabase_client", lambda: sb)

        resp = client.post("/me/bootstrap-tester")
        assert resp.status_code == 200
        assert resp.json() == {"granted": False, "reason": "not_in_allowlist"}

    def test_fresh_grant_also_grants_initial_credits(self, client, monkeypatch):
        """The fresh-grant path should also trigger the one-time initial
        tester credit allocation (best-effort — see main.py bootstrap_tester)."""
        import main

        monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
        sb, upserted = _wire_sb(existing_tester_rows=[])
        monkeypatch.setattr(main, "get_supabase_client", lambda: sb)

        with patch("subscriptions.admin_service.grant_initial_tester_credits") as gitc:
            resp = client.post("/me/bootstrap-tester")

        assert resp.status_code == 200
        assert resp.json()["granted"] is True
        gitc.assert_called_once()

    def test_already_tester_does_not_regrant_credits(self, client, monkeypatch):
        """When bootstrap short-circuits because the user is already a
        tester, the initial-credits helper must NOT be invoked again."""
        import main

        monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
        existing = [{"reason": "tester", "expires_at": None, "granted_at": "2026-04-01T00:00:00Z"}]
        sb, upserted = _wire_sb(existing_tester_rows=existing)
        monkeypatch.setattr(main, "get_supabase_client", lambda: sb)

        with patch("subscriptions.admin_service.grant_initial_tester_credits") as gitc:
            resp = client.post("/me/bootstrap-tester")

        assert resp.json()["granted"] is False
        gitc.assert_not_called()

    def test_bootstrap_respects_revoked_marker(self, client, monkeypatch):
        """When user has a tester_revoked row, bootstrap should not re-grant
        even if email is in TESTER_EMAILS — admin's revoke decision is sticky."""
        import main

        monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
        existing = [{"reason": "tester_revoked", "expires_at": None, "granted_at": "2026-04-01T00:00:00Z"}]
        sb, upserted = _wire_sb(existing_tester_rows=existing)
        monkeypatch.setattr(main, "get_supabase_client", lambda: sb)

        resp = client.post("/me/bootstrap-tester")
        assert resp.status_code == 200
        assert resp.json() == {"granted": False, "reason": "revoked"}
        assert "payload" not in upserted  # no new upsert
