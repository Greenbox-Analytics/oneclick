"""Endpoint tests for POST /admin/orgs/{org_id}/pool/grant + PAID_IN_KINDS."""

from unittest.mock import MagicMock

import pytest

ORG_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def admin_client(client, monkeypatch):
    """TestClient with require_admin satisfied — mirrors test_admin_credits.py."""
    import main
    from subscriptions.admin_auth import require_admin

    async def _pass():
        return "admin@example.com"

    main.app.dependency_overrides[require_admin] = _pass
    yield client
    main.app.dependency_overrides.pop(require_admin, None)


def _wire_org_tables(mock_supabase, org_exists=True, org_status="pending", ledger_rows=None):
    """organizations lookup + org wallet read used by read_or_create_org_wallet.

    Caches one MagicMock per table name (rather than a fresh mock per call) so
    that "organizations" — hit three times in the grant+activate flow
    (existence check, maybe_activate_org's select, maybe_activate_org's
    update) — is the SAME object throughout a request. That's what lets a
    caller assert against the returned builder's `.update` call.

    `ledger_rows` feeds cumulative_paid_in's SUM read inside
    maybe_activate_org — defaults to empty (no paid-in credits, org stays
    pending).

    Returns the {table_name: MagicMock} builder map.
    """
    builders: dict = {}

    def table_side_effect(name):
        if name not in builders:
            builders[name] = MagicMock()
        t = builders[name]
        if name == "organizations":
            t.select.return_value.eq.return_value.execute.return_value.data = (
                [{"id": ORG_ID, "status": org_status, "min_initial_purchase_credits": None}] if org_exists else []
            )
        elif name == "credit_wallets":
            t.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
                {"id": "wallet-1", "owner_type": "org", "owner_id": ORG_ID}
            ]
        elif name == "credit_ledger":
            # cumulative_paid_in SUM read inside maybe_activate_org
            t.select.return_value.eq.return_value.in_.return_value.execute.return_value.data = ledger_rows or []
        return t

    mock_supabase.table.side_effect = table_side_effect
    return builders


class TestOrgPoolGrant:
    def test_grant_calls_rpc_with_admin_grant_kind(self, admin_client, mock_supabase):
        _wire_org_tables(mock_supabase)
        mock_supabase.rpc.return_value.execute.return_value.data = {"balance_after": 500}

        resp = admin_client.post(
            f"/admin/orgs/{ORG_ID}/pool/grant",
            json={"amount": 500, "reason": "pilot comp", "idempotency_key": "k-1"},
        )
        assert resp.status_code == 200
        args = mock_supabase.rpc.call_args
        assert args.args[0] == "grant_credits"
        params = args.args[1]
        assert params["p_kind"] == "admin_grant"
        assert params["p_bucket"] == "reserve"
        assert params["p_request_id"] == "admin-org-grant:k-1"
        assert params["p_amount"] == 500

    def test_missing_idempotency_key_is_422(self, admin_client, mock_supabase):
        resp = admin_client.post(
            f"/admin/orgs/{ORG_ID}/pool/grant",
            json={"amount": 500, "reason": "pilot comp"},
        )
        assert resp.status_code == 422

    def test_unknown_org_is_404(self, admin_client, mock_supabase):
        _wire_org_tables(mock_supabase, org_exists=False)
        resp = admin_client.post(
            f"/admin/orgs/{ORG_ID}/pool/grant",
            json={"amount": 500, "reason": "x", "idempotency_key": "k-2"},
        )
        assert resp.status_code == 404

    def test_non_admin_rejected(self, client, mock_supabase):
        resp = client.post(
            f"/admin/orgs/{ORG_ID}/pool/grant",
            json={"amount": 500, "reason": "x", "idempotency_key": "k-3"},
        )
        assert resp.status_code in (401, 403)

    def test_activation_flips_org_when_paid_in_crosses_floor(self, admin_client, mock_supabase, monkeypatch):
        """Guards the maybe_activate_org wiring specifically: deleting that
        call (and the result["activated"] assignment) from grant_org_credits
        would leave every other test in this class green, since none of them
        put the org's cumulative paid-in at/past the activation floor."""
        monkeypatch.delenv("ENTERPRISE_MIN_INITIAL_CREDITS", raising=False)  # deterministic 10000 default
        builders = _wire_org_tables(mock_supabase, org_status="pending", ledger_rows=[{"delta": 10000}])
        mock_supabase.rpc.return_value.execute.return_value.data = {"balance_after": 10000}

        resp = admin_client.post(
            f"/admin/orgs/{ORG_ID}/pool/grant",
            json={"amount": 10000, "reason": "pilot activation", "idempotency_key": "k-act"},
        )
        assert resp.status_code == 200
        builders["organizations"].update.assert_called_with({"status": "active"})
        assert resp.json()["result"]["activated"] is True


class TestPaidInKinds:
    def test_admin_grant_is_paid_in(self):
        from orgs.wallets import PAID_IN_KINDS

        assert "admin_grant" in PAID_IN_KINDS
