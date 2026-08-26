"""Endpoint tests for GET /admin/orgs (admin Organizations tab)."""

from unittest.mock import MagicMock

import pytest

ORG_A = "aaaaaaaa-0000-0000-0000-000000000001"


@pytest.fixture
def admin_client(client, monkeypatch):
    import main
    from subscriptions.admin_auth import require_admin

    async def _pass():
        return "admin@example.com"

    main.app.dependency_overrides[require_admin] = _pass
    yield client
    main.app.dependency_overrides.pop(require_admin, None)


def _wire_tables(mock_supabase, wallet_rows=None):
    """Wires organizations/credit_wallets/org_members/credit_ledger mocks.

    `wallet_rows` defaults to a single pool-wallet row; pass `[]` to simulate
    a fresh org with no pool wallet yet (the normal state of a pending org
    that hasn't been topped up). Returns the list of table names requested,
    in call order, so a test can assert `credit_ledger` was never queried
    when there's no wallet to look its id up from.
    """
    if wallet_rows is None:
        wallet_rows = [{"id": "w-1", "bundle_balance": 100, "reserve_balance": 400}]
    called_tables: list[str] = []

    def table_side_effect(name):
        called_tables.append(name)
        t = MagicMock()
        if name == "organizations":
            t.select.return_value.order.return_value.execute.return_value.data = [
                {
                    "id": ORG_A,
                    "name": "Archived Corp",
                    "status": "archived",
                    "archived_at": "2026-07-01T00:00:00Z",
                    "monthly_dispersal_credits": 5000,
                    "min_initial_purchase_credits": None,
                    "created_at": "2026-06-01T00:00:00Z",
                }
            ]
        elif name == "credit_wallets":
            t.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = wallet_rows
        elif name == "org_members":
            res = MagicMock()
            res.count = 3
            t.select.return_value.eq.return_value.eq.return_value.execute.return_value = res
        elif name == "credit_ledger":
            t.select.return_value.eq.return_value.in_.return_value.execute.return_value.data = [{"delta": 12000}]
        return t

    mock_supabase.table.side_effect = table_side_effect
    return called_tables


class TestAdminOrgsList:
    def test_shape_includes_archived_and_floor_fallback(self, admin_client, mock_supabase, monkeypatch):
        # conftest documents prior incidents of local env leaking into tests —
        # pin the default-floor branch deterministically.
        monkeypatch.delenv("ENTERPRISE_MIN_INITIAL_CREDITS", raising=False)
        _wire_tables(mock_supabase)
        resp = admin_client.get("/admin/orgs")
        assert resp.status_code == 200
        orgs = resp.json()["orgs"]
        assert len(orgs) == 1
        org = orgs[0]
        assert org["status"] == "archived"
        assert org["archivedAt"] == "2026-07-01T00:00:00Z"
        assert org["memberCount"] == 3
        assert org["bundleBalance"] == 100
        assert org["reserveBalance"] == 400
        assert org["monthlyDispersalCredits"] == 5000
        assert org["activationFloor"] == 10000  # env default fallback
        assert org["cumulativePaidIn"] == 12000

    def test_no_wallet_defaults_to_zero_and_skips_ledger_query(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.delenv("ENTERPRISE_MIN_INITIAL_CREDITS", raising=False)
        called_tables = _wire_tables(mock_supabase, wallet_rows=[])
        resp = admin_client.get("/admin/orgs")
        assert resp.status_code == 200
        org = resp.json()["orgs"][0]
        assert org["bundleBalance"] == 0
        assert org["reserveBalance"] == 0
        assert org["cumulativePaidIn"] == 0
        assert "credit_ledger" not in called_tables

    def test_non_admin_rejected(self, client, mock_supabase):
        resp = client.get("/admin/orgs")
        assert resp.status_code in (401, 403)
