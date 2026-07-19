"""Endpoint tests for /admin/users/{id}/credits/*. Exercises require_admin +
delegation to admin_service.grant_user_credits / get_user_credit_ledger."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def admin_client(client, monkeypatch):
    """TestClient with require_admin satisfied — mirror tests/test_admin_router.py's override."""
    import main
    from subscriptions.admin_auth import require_admin

    async def _pass():
        return "admin@example.com"

    main.app.dependency_overrides[require_admin] = _pass
    yield client
    main.app.dependency_overrides.pop(require_admin, None)


class TestAdminCredits:
    def test_grant_calls_rpc(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        # grant_credits returns jsonb {duplicate, balance_after}; configure the
        # mock with that realistic shape so the happy path (not the isinstance
        # fallback) is what's exercised, and assert the endpoint surfaces it.
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False, "balance_after": 200})
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": 200, "reason": "support goodwill"},
        )
        assert resp.status_code == 200
        rpc_names = [c.args[0] for c in mock_supabase.rpc.call_args_list]
        assert "grant_credits" in rpc_names
        args = [c.args[1] for c in mock_supabase.rpc.call_args_list if c.args[0] == "grant_credits"][0]
        assert args["p_amount"] == 200 and args["p_bucket"] == "reserve" and args["p_kind"] == "admin_grant"
        assert args["p_metadata"]["granted_by"] == "admin@example.com"
        # RPC result is surfaced verbatim under "result".
        assert resp.json()["result"] == {"duplicate": False, "balance_after": 200}
        assert resp.json()["granted"] == 200

    def test_grant_passes_idempotency_key(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": 50, "reason": "retry-safe", "idempotency_key": "abc"},
        )
        assert resp.status_code == 200
        args = [c.args[1] for c in mock_supabase.rpc.call_args_list if c.args[0] == "grant_credits"][0]
        assert args["p_request_id"] == "admin-grant:abc"

    def test_grant_without_idempotency_key_passes_none(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": 50, "reason": "no-key"},
        )
        assert resp.status_code == 200
        args = [c.args[1] for c in mock_supabase.rpc.call_args_list if c.args[0] == "grant_credits"][0]
        assert args["p_request_id"] is None

    def test_amount_over_ceiling_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": 2_000_000, "reason": "fat finger"},
        )
        assert resp.status_code == 422

    def test_negative_amount_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": -5, "reason": "oops"},
        )
        assert resp.status_code == 422

    def test_zero_amount_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": 0, "reason": "nope"},
        )
        assert resp.status_code == 422

    def test_grant_missing_wallet_404(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from tests.conftest import _default_table_side_effect

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "credit_wallets":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        mock_supabase.table.side_effect = side_effect
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": 50, "reason": "x"},
        )
        assert resp.status_code == 404

    def test_ledger_returns_rows(self, admin_client):
        resp = admin_client.get("/admin/users/00000000-0000-0000-0000-000000000002/credits/ledger")
        assert resp.status_code == 200
        assert isinstance(resp.json()["entries"], list)


class TestGrantInitialTesterCredits:
    @staticmethod
    def _sb_with_wallet(wallet_rows):
        sb = MagicMock()
        b = MagicMock()
        for m in ("select", "eq"):
            getattr(b, m).return_value = b
        b.execute.return_value = MagicMock(data=wallet_rows)
        sb.table.return_value = b
        return sb

    def test_credits_off_is_noop(self, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions.admin_service import grant_initial_tester_credits

        sb = self._sb_with_wallet([{"id": "w1"}])
        assert grant_initial_tester_credits(sb, "u1") is False
        sb.rpc.assert_not_called()

    def test_grants_reserve_with_once_per_user_key(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.admin_service import grant_initial_tester_credits

        sb = self._sb_with_wallet([{"id": "w1"}])
        assert grant_initial_tester_credits(sb, "u1", 500) is True
        name, params = sb.rpc.call_args[0]
        assert name == "grant_credits"
        assert params["p_wallet_id"] == "w1"
        assert params["p_amount"] == 500
        assert params["p_kind"] == "admin_grant"
        assert params["p_bucket"] == "reserve"
        # Once-per-user FOREVER: the ledger's unique request_id index dedupes
        # bootstrap re-runs and admin re-grants. Top-ups use the grant endpoint.
        assert params["p_request_id"] == "tester-init:u1"

    def test_amount_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("TESTER_INITIAL_CREDITS", "250")
        from subscriptions.admin_service import grant_initial_tester_credits

        sb = self._sb_with_wallet([{"id": "w1"}])
        grant_initial_tester_credits(sb, "u1")
        assert sb.rpc.call_args[0][1]["p_amount"] == 250

    def test_bad_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("TESTER_INITIAL_CREDITS", "not-a-number")
        from subscriptions.admin_service import grant_initial_tester_credits

        sb = self._sb_with_wallet([{"id": "w1"}])
        grant_initial_tester_credits(sb, "u1")
        assert sb.rpc.call_args[0][1]["p_amount"] == 1000

    def test_missing_wallet_skips_without_raising(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions.admin_service import grant_initial_tester_credits

        sb = self._sb_with_wallet([])
        assert grant_initial_tester_credits(sb, "u1") is False
        sb.rpc.assert_not_called()


class TestAdminCreditsGrantRequiresAdmin:
    def test_grant_requires_admin(self, client, monkeypatch):
        """No require_admin override → the dependency's real logic runs.

        The plain `client` fixture authenticates as TEST_USER_EMAIL
        ("test@example.com") with no ADMIN_EMAILS set and no DB-admin
        profile row. With ADMIN_EMAILS completely empty, require_admin's
        fail-loud "no admins configured" branch fires and returns 500
        (operator misconfig), NOT 403 — see subscriptions/admin_auth.py.
        To exercise the actual "not authorized" 403 path (mirrors
        test_admin_router.py's `_set_admin_emails` pattern), pin
        ADMIN_EMAILS to a *different* address so env_admin_emails() is
        non-empty but doesn't match the caller.
        """
        monkeypatch.setenv("ADMIN_EMAILS", "someoneelse@example.com")
        resp = client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/grant",
            json={"amount": 10, "reason": "x"},
        )
        assert resp.status_code == 403
