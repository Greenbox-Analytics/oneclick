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


class TestAdjustUserCredits:
    """Service-level tests for the admin clawback path (adjust_user_credits).

    The clamp/reserve-only/never-negative semantics live in the debit_credits
    RPC itself (migration 20260720000001_credit_packs.sql) and can't be
    asserted against a mocked `sb.rpc` — these tests pin the CALL SHAPE the
    service sends, the 404-on-no-wallet path, and the raise-on-unexpected-
    RPC-shape path. Real clamp/bucket behavior is a real-DB launch-gate item
    (spec §10), not something a unit test can fake honestly.
    """

    def test_removes_credits_with_clawback_kind(self, monkeypatch):
        from subscriptions.admin_service import adjust_user_credits

        sb = TestGrantInitialTesterCredits._sb_with_wallet([{"id": "w1"}])
        sb.rpc.return_value.execute.return_value = MagicMock(
            data={"duplicate": False, "balance_after": 1500, "removed": 500, "shortfall": 0}
        )
        result = adjust_user_credits(sb, "u1", 500, "stripe refund re_123", "admin@x.com", request_id="re_123")
        name, params = sb.rpc.call_args[0]
        assert name == "debit_credits"
        assert params["p_amount"] == 500
        assert params["p_kind"] == "clawback"
        assert params["p_action"] == "admin_adjust"
        assert params["p_request_id"] == "admin-adjust:re_123"
        assert params["p_metadata"]["adjusted_by"] == "admin@x.com"
        assert params["p_metadata"]["reason"] == "stripe refund re_123"
        # Verbatim relay — no massaging of the RPC's response shape.
        assert result == {"duplicate": False, "balance_after": 1500, "removed": 500, "shortfall": 0}

    def test_no_wallet_raises_value_error(self):
        import pytest as _pytest

        from subscriptions.admin_service import adjust_user_credits

        sb = TestGrantInitialTesterCredits._sb_with_wallet([])
        with _pytest.raises(ValueError):
            adjust_user_credits(sb, "u1", 500, "r", "admin@x.com", request_id="req1")

    def test_raises_on_unexpected_rpc_shape(self):
        """A non-dict RPC result must raise, not be swallowed into a fabricated
        success — we can never claim a removal we can't confirm."""
        import pytest as _pytest

        from subscriptions.admin_service import adjust_user_credits

        sb = TestGrantInitialTesterCredits._sb_with_wallet([{"id": "w1"}])
        sb.rpc.return_value.execute.return_value = MagicMock(data="not-a-dict")
        with _pytest.raises(RuntimeError):
            adjust_user_credits(sb, "u1", 500, "r", "admin@x.com", request_id="req1")


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


class TestAdminCreditsAdjust:
    """Endpoint tests for POST /admin/users/{id}/credits/adjust (clawback).

    Mirrors TestAdminCredits's grant-endpoint coverage: RPC call shape,
    idempotency-key enforcement (REQUIRED here, unlike grant), 404 mapping,
    and verbatim relay of the RPC result (fresh-clawback and duplicate-replay
    shapes both flow through untouched).
    """

    def test_adjust_calls_rpc_with_clawback_kind(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(
            data={"duplicate": False, "balance_after": 1500, "removed": 500, "shortfall": 0}
        )
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 500, "reason": "stripe refund re_123", "idempotency_key": "re_123"},
        )
        assert resp.status_code == 200
        rpc_names = [c.args[0] for c in mock_supabase.rpc.call_args_list]
        assert "debit_credits" in rpc_names
        args = [c.args[1] for c in mock_supabase.rpc.call_args_list if c.args[0] == "debit_credits"][0]
        assert args["p_amount"] == 500
        assert args["p_kind"] == "clawback"
        assert args["p_action"] == "admin_adjust"
        assert args["p_request_id"] == "admin-adjust:re_123"
        assert args["p_metadata"]["adjusted_by"] == "admin@example.com"
        assert args["p_metadata"]["reason"] == "stripe refund re_123"
        # RPC result surfaced verbatim under "result" — {removed, shortfall,
        # balance_after} on a fresh clawback.
        assert resp.json()["result"] == {"duplicate": False, "balance_after": 1500, "removed": 500, "shortfall": 0}
        assert resp.json()["requested"] == 500

    def test_adjust_relays_duplicate_replay_verbatim(self, admin_client, mock_supabase, monkeypatch):
        """A replayed idempotency key returns only {duplicate, balance_after}
        from the RPC — the endpoint must relay exactly that, not fabricate
        removed/shortfall fields that weren't actually recomputed."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": True, "balance_after": 1500})
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 500, "reason": "retry", "idempotency_key": "re_123"},
        )
        assert resp.status_code == 200
        assert resp.json()["result"] == {"duplicate": True, "balance_after": 1500}

    def test_adjust_missing_idempotency_key_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 500, "reason": "no key"},
        )
        assert resp.status_code == 422

    def test_adjust_blank_idempotency_key_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 500, "reason": "blank key", "idempotency_key": ""},
        )
        assert resp.status_code == 422

    def test_adjust_amount_over_ceiling_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 2_000_000, "reason": "fat finger", "idempotency_key": "k1"},
        )
        assert resp.status_code == 422

    def test_adjust_negative_amount_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": -5, "reason": "oops", "idempotency_key": "k1"},
        )
        assert resp.status_code == 422

    def test_adjust_zero_amount_rejected(self, admin_client):
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 0, "reason": "nope", "idempotency_key": "k1"},
        )
        assert resp.status_code == 422

    def test_adjust_missing_wallet_404(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from tests.conftest import _default_table_side_effect

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "credit_wallets":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        mock_supabase.table.side_effect = side_effect
        resp = admin_client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 50, "reason": "x", "idempotency_key": "k1"},
        )
        assert resp.status_code == 404


class TestAdminCreditsAdjustRequiresAdmin:
    def test_adjust_requires_admin(self, client, monkeypatch):
        """Mirrors TestAdminCreditsGrantRequiresAdmin — ADMIN_EMAILS pinned to
        a different address so require_admin's real 403 path (not the
        no-admins-configured 500) is what's exercised."""
        monkeypatch.setenv("ADMIN_EMAILS", "someoneelse@example.com")
        resp = client.post(
            "/admin/users/00000000-0000-0000-0000-000000000002/credits/adjust",
            json={"amount": 10, "reason": "x", "idempotency_key": "k1"},
        )
        assert resp.status_code == 403


class TestAdjustOrgPool:
    """Service-level tests for adjust_org_pool (org-wallet clawback —
    follow-ups plan 2026-07-22, Task 2). Mirrors TestAdjustUserCredits: pins
    the RPC call shape against the ORG wallet (distinct owner_type from the
    per-user path), the 404-no-wallet path, and the raise-on-unexpected-
    RPC-shape path. The clamp/reserve-only semantics themselves live in the
    debit_credits RPC and are a real-DB launch-gate item, not something a
    mocked client can honestly assert.
    """

    @staticmethod
    def _sb_with_org_wallet(wallet_rows):
        sb = MagicMock()
        b = MagicMock()
        for m in ("select", "eq"):
            getattr(b, m).return_value = b
        b.execute.return_value = MagicMock(data=wallet_rows)
        sb.table.return_value = b
        return sb

    def test_removes_credits_with_clawback_kind_on_org_wallet(self):
        from subscriptions.admin_service import adjust_org_pool

        sb = self._sb_with_org_wallet([{"id": "org-w1"}])
        sb.rpc.return_value.execute.return_value = MagicMock(
            data={"duplicate": False, "balance_after": 4500, "removed": 500, "shortfall": 0}
        )
        result = adjust_org_pool(sb, "org-1", 500, "refund re_999", "admin@x.com", request_id="re_999")

        name, params = sb.rpc.call_args[0]
        assert name == "debit_credits"
        assert params["p_wallet_id"] == "org-w1"
        assert params["p_amount"] == 500
        assert params["p_kind"] == "clawback"
        assert params["p_action"] == "admin_adjust"
        # Distinct key namespace from the per-user path's `admin-adjust:` —
        # no collision class between the two clawback flows.
        assert params["p_request_id"] == "admin-org-adjust:re_999"
        assert params["p_metadata"] == {"reason": "refund re_999", "adjusted_by": "admin@x.com", "org_id": "org-1"}
        # Verbatim relay — no massaging of the RPC's response shape.
        assert result == {"duplicate": False, "balance_after": 4500, "removed": 500, "shortfall": 0}

        # Wallet lookup targeted the ORG wallet, not a user wallet.
        eq_calls = [c.args for c in sb.table.return_value.eq.call_args_list]
        assert ("owner_type", "org") in eq_calls
        assert ("owner_id", "org-1") in eq_calls

    def test_no_wallet_raises_value_error(self):
        from subscriptions.admin_service import adjust_org_pool

        sb = self._sb_with_org_wallet([])
        with pytest.raises(ValueError):
            adjust_org_pool(sb, "org-1", 500, "r", "admin@x.com", request_id="req1")

    def test_raises_on_unexpected_rpc_shape(self):
        """A non-dict RPC result must raise, not be swallowed into a
        fabricated success — we can never claim a removal we can't confirm."""
        from subscriptions.admin_service import adjust_org_pool

        sb = self._sb_with_org_wallet([{"id": "org-w1"}])
        sb.rpc.return_value.execute.return_value = MagicMock(data="not-a-dict")
        with pytest.raises(RuntimeError):
            adjust_org_pool(sb, "org-1", 500, "r", "admin@x.com", request_id="req1")


def _cached_table_builder(table_data: dict):
    """Returns (side_effect_fn, builders): a `.table(name)` side_effect where
    repeated calls for the SAME table name return the SAME MagicMock builder
    (mirrors test_credits_stripe.py's `_mock_supabase` helper). Needed here
    because get_org_pool issues two separate queries against `credit_ledger`
    (the last-50 listing AND cumulative_purchased's SUM) that must see the
    same configured rows.
    """
    builders: dict = {}

    def _get(name):
        if name not in builders:
            b = MagicMock()
            for m in ("select", "eq", "order", "limit", "in_"):
                getattr(b, m).return_value = b
            b.execute.return_value = MagicMock(data=list(table_data.get(name, [])))
            builders[name] = b
        return builders[name]

    return _get, builders


class TestGetOrgPool:
    """Tests for admin_service.get_org_pool — the support-visibility
    snapshot read before disposing of a pool via adjust_org_pool."""

    ORG_ID = "org-1"

    def test_shape_and_cumulative_purchased_matches_shared_helper(self):
        """NOTE: the `credit_ledger` fixture below is all `purchase`-kind
        rows deliberately — the shared MagicMock builder can't distinguish
        the ledger-listing query (unfiltered by kind) from
        cumulative_purchased's `kind='purchase'`-filtered SUM (both are
        `.eq()` no-ops against the same mock), so mixing kinds here would
        make the sum reflect the mock's inability to filter, not the
        service's real behavior. The real per-kind filtering is exercised
        against the actual RPC/SQL in the launch-gate list (spec §10); this
        test's job is only to prove get_org_pool's cumulativePurchased comes
        from the SAME helper function stripe_events uses, not a duplicate
        reimplementation that can drift.
        """
        from orgs.wallets import cumulative_purchased
        from subscriptions.admin_service import get_org_pool

        side_effect, builders = _cached_table_builder(
            {
                "organizations": [{"id": self.ORG_ID, "status": "active", "archived_at": None}],
                "credit_wallets": [
                    {
                        "id": "org-w1",
                        "owner_type": "org",
                        "owner_id": self.ORG_ID,
                        "bundle_balance": 0,
                        "reserve_balance": 4500,
                    }
                ],
                "credit_ledger": [
                    {"id": "l1", "delta": 5000, "kind": "purchase", "created_at": "2026-07-01T00:00:00+00:00"},
                ],
            }
        )
        sb = MagicMock()
        sb.table.side_effect = side_effect

        result = get_org_pool(sb, self.ORG_ID)

        assert result["orgId"] == self.ORG_ID
        assert result["status"] == "active"
        assert result["archivedAt"] is None
        assert result["poolBalance"] == 4500
        # No drift: this must equal the SAME helper the stripe activation
        # check uses, computed independently against the same fixture rows.
        assert result["cumulativePurchased"] == cumulative_purchased(sb, "org-w1") == 5000
        assert result["ledger"] == builders["credit_ledger"].execute.return_value.data

    def test_no_wallet_yet_reports_zeroes_not_an_error(self):
        """A fresh 'pending' org that's never been topped up has no pool
        wallet row yet — that's a normal state, not a 404."""
        from subscriptions.admin_service import get_org_pool

        side_effect, _ = _cached_table_builder(
            {
                "organizations": [{"id": self.ORG_ID, "status": "pending", "archived_at": None}],
                "credit_wallets": [],
            }
        )
        sb = MagicMock()
        sb.table.side_effect = side_effect

        result = get_org_pool(sb, self.ORG_ID)
        assert result["poolBalance"] == 0
        assert result["cumulativePurchased"] == 0
        assert result["ledger"] == []

    def test_unknown_org_raises_value_error(self):
        from subscriptions.admin_service import get_org_pool

        side_effect, _ = _cached_table_builder({"organizations": []})
        sb = MagicMock()
        sb.table.side_effect = side_effect

        with pytest.raises(ValueError):
            get_org_pool(sb, self.ORG_ID)

    def test_reports_archived_org(self):
        from subscriptions.admin_service import get_org_pool

        side_effect, _ = _cached_table_builder(
            {
                "organizations": [{"id": self.ORG_ID, "status": "active", "archived_at": "2026-07-15T00:00:00+00:00"}],
                "credit_wallets": [
                    {
                        "id": "org-w1",
                        "owner_type": "org",
                        "owner_id": self.ORG_ID,
                        "bundle_balance": 0,
                        "reserve_balance": 0,
                    }
                ],
                "credit_ledger": [],
            }
        )
        sb = MagicMock()
        sb.table.side_effect = side_effect

        result = get_org_pool(sb, self.ORG_ID)
        assert result["archivedAt"] == "2026-07-15T00:00:00+00:00"


class TestOrgPoolEndpoints:
    """Endpoint tests for POST /admin/orgs/{id}/pool/clawback and
    GET /admin/orgs/{id}/pool — same `admin_client` fixture (require_admin
    overridden only) used by TestAdminCredits/TestAdminCreditsAdjust above.
    Flag-independence (LICENSING_ENABLED unset) is asserted explicitly since
    these endpoints — like org suspend/reactivate — are gated ONLY by
    PLATFORM require_admin, never licensing_enabled().
    """

    ORG_ID = "org-1"

    def _install_org_pool_tables(self, mock_supabase, table_data):
        side_effect, builders = _cached_table_builder(table_data)
        mock_supabase.table.side_effect = side_effect
        return builders

    def test_clawback_relays_rpc_result_verbatim(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        self._install_org_pool_tables(
            mock_supabase, {"credit_wallets": [{"id": "org-w1", "owner_type": "org", "owner_id": self.ORG_ID}]}
        )
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(
            data={"duplicate": False, "balance_after": 4000, "removed": 1000, "shortfall": 0}
        )
        resp = admin_client.post(
            f"/admin/orgs/{self.ORG_ID}/pool/clawback",
            json={"amount": 1000, "reason": "refund re_1", "idempotency_key": "re_1"},
        )
        assert resp.status_code == 200
        # Relayed verbatim — NOT wrapped like the per-user adjust endpoint's
        # {"requested": ..., "result": ...} shape.
        assert resp.json() == {"duplicate": False, "balance_after": 4000, "removed": 1000, "shortfall": 0}
        rpc_names = [c.args[0] for c in mock_supabase.rpc.call_args_list]
        assert "debit_credits" in rpc_names
        args = [c.args[1] for c in mock_supabase.rpc.call_args_list if c.args[0] == "debit_credits"][0]
        assert args["p_request_id"] == "admin-org-adjust:re_1"
        assert args["p_kind"] == "clawback"

    def test_clawback_missing_wallet_404(self, admin_client, mock_supabase):
        self._install_org_pool_tables(mock_supabase, {"credit_wallets": []})
        resp = admin_client.post(
            f"/admin/orgs/{self.ORG_ID}/pool/clawback",
            json={"amount": 500, "reason": "x", "idempotency_key": "k1"},
        )
        assert resp.status_code == 404

    def test_clawback_missing_idempotency_key_422(self, admin_client):
        resp = admin_client.post(
            f"/admin/orgs/{self.ORG_ID}/pool/clawback",
            json={"amount": 500, "reason": "no key"},
        )
        assert resp.status_code == 422

    def test_clawback_blank_idempotency_key_422(self, admin_client):
        resp = admin_client.post(
            f"/admin/orgs/{self.ORG_ID}/pool/clawback",
            json={"amount": 500, "reason": "blank key", "idempotency_key": ""},
        )
        assert resp.status_code == 422

    def test_clawback_amount_over_ceiling_422(self, admin_client):
        resp = admin_client.post(
            f"/admin/orgs/{self.ORG_ID}/pool/clawback",
            json={"amount": 2_000_000, "reason": "fat finger", "idempotency_key": "k1"},
        )
        assert resp.status_code == 422

    def test_get_pool_returns_full_shape(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        self._install_org_pool_tables(
            mock_supabase,
            {
                "organizations": [{"id": self.ORG_ID, "status": "active", "archived_at": None}],
                "credit_wallets": [
                    {
                        "id": "org-w1",
                        "owner_type": "org",
                        "owner_id": self.ORG_ID,
                        "bundle_balance": 0,
                        "reserve_balance": 2500,
                    }
                ],
                "credit_ledger": [
                    {"id": "l1", "delta": 5000, "kind": "purchase", "created_at": "2026-07-01T00:00:00+00:00"}
                ],
            },
        )
        resp = admin_client.get(f"/admin/orgs/{self.ORG_ID}/pool")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {
            "orgId": self.ORG_ID,
            "status": "active",
            "archivedAt": None,
            "poolBalance": 2500,
            "cumulativePurchased": 5000,
            "ledger": [{"id": "l1", "delta": 5000, "kind": "purchase", "created_at": "2026-07-01T00:00:00+00:00"}],
        }

    def test_get_pool_unknown_org_404(self, admin_client, mock_supabase):
        self._install_org_pool_tables(mock_supabase, {"organizations": []})
        resp = admin_client.get(f"/admin/orgs/{self.ORG_ID}/pool")
        assert resp.status_code == 404
