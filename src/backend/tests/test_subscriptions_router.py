"""Tests for /me/entitlements endpoint."""

from unittest.mock import MagicMock

import pytest

from tests.conftest import TEST_USER_ID, MockQueryBuilder

FREE_TIER_ROW = {
    "tier": "free",
    "max_artists": 3,
    "max_projects": 3,
    "max_boards": 3,
    "max_tasks": 50,
    "max_storage_bytes": 1073741824,
    "max_split_sheets_per_month": 5,
    "max_oneclick_runs_per_month": 1,
    "zoe_enabled": False,
    "oneclick_enabled": True,
    "registry_enabled": False,
    "integrations_allowed": ["google_drive"],
    "updated_at": "2026-05-09T00:00:00+00:00",
}
FREE_SUB_ROW = {
    "id": "00000000-0000-0000-0000-000000000aaa",
    "user_id": TEST_USER_ID,
    "tier": "free",
    "status": "active",
    "stripe_customer_id": None,
    "stripe_subscription_id": None,
    "stripe_price_id": None,
    "current_period_start": None,
    "current_period_end": None,
    "cancel_at_period_end": False,
    "canceled_at": None,
    "created_at": "2026-05-09T00:00:00+00:00",
    "updated_at": "2026-05-09T00:00:00+00:00",
}
ZERO_USAGE_ROW = {
    "user_id": TEST_USER_ID,
    "total_storage_bytes": 0,
    "split_sheets_this_period": 0,
    "period_start": "2026-05-09T00:00:00+00:00",
    "period_end": "2099-05-09T00:00:00+00:00",  # far future so no rollover
    "updated_at": "2026-05-09T00:00:00+00:00",
}


def _wire_supabase(mock_supabase, rows_by_table):
    def _table(name):
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=rows_by_table.get(name, []), count=len(rows_by_table.get(name, [])))
        return b

    mock_supabase.table.side_effect = _table


@pytest.fixture(autouse=True)
def _reset_service_singleton():
    """Force a fresh EntitlementsService for each test so cache+wiring don't leak."""
    from subscriptions import deps

    deps._entitlements_service = None
    yield
    deps._entitlements_service = None


class TestGetEntitlements:
    def test_returns_200_with_correct_shape(self, client, mock_supabase):
        _wire_supabase(
            mock_supabase,
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            },
        )

        response = client.get("/me/entitlements")

        assert response.status_code == 200
        data = response.json()
        assert data["tier"] == "free"
        assert data["status"] == "active"
        assert data["caps"]["maxArtists"] == 3
        assert data["features"]["zoeEnabled"] is False
        assert data["features"]["integrationsAllowed"] == ["google_drive"]
        assert data["usage"]["totalStorageBytes"] == 0
        assert data["hasOverrides"] is False
        assert data["degraded"] is False

    def test_unauthenticated_returns_401(self, mock_supabase):
        from fastapi.testclient import TestClient

        import main

        original = dict(main.app.dependency_overrides)
        main.app.dependency_overrides.clear()
        try:
            with TestClient(main.app) as tc:
                response = tc.get("/me/entitlements")
            assert response.status_code == 401
        finally:
            main.app.dependency_overrides.update(original)

    def test_service_error_returns_degraded_not_500(self, client, mock_supabase):
        mock_supabase.table.side_effect = RuntimeError("DB blew up")

        response = client.get("/me/entitlements")

        assert response.status_code == 200
        data = response.json()
        assert data["degraded"] is True
        assert data["tier"] == "free"


class TestNoRefreshEndpoint:
    def test_no_refresh_endpoint(self, client, mock_supabase):
        """The /refresh endpoint was intentionally omitted — POST should 404 or 405."""
        _wire_supabase(
            mock_supabase,
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            },
        )
        response = client.post("/me/entitlements/refresh")
        assert response.status_code in (404, 405)


class TestBillingPrefs:
    def test_entitlements_includes_credits_when_enabled(self, client, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        resp = client.get("/me/entitlements")
        assert resp.status_code == 200
        assert resp.json()["credits"] is not None
        assert resp.json()["credits"]["prices"]["zoeMessage"] == 3

    def test_entitlements_credits_null_when_disabled(self, client):
        resp = client.get("/me/entitlements")
        assert resp.status_code == 200
        assert resp.json()["credits"] is None

    def test_set_overage_prefs(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from unittest.mock import MagicMock

        from tests.conftest import _default_table_side_effect

        persisted = {
            "user_id": TEST_USER_ID,
            "tier": "pro",
            "status": "active",
            "overage_enabled": True,
            "overage_cap_credits": 500,
            "storage_overage_enabled": False,
        }

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "subscriptions":
                # Same builder shape serves both the tier read and the upsert
                # result (MockQueryBuilder.upsert returns self, sharing execute).
                b.execute.return_value = MagicMock(data=[persisted], count=1)
            return b

        mock_supabase.table.side_effect = side_effect
        resp = client.post(
            "/me/billing-prefs",
            json={"overage_enabled": True, "overage_cap_credits": 500},
        )
        assert resp.status_code == 200
        assert resp.json()["overageEnabled"] is True
        assert resp.json()["overageCapCredits"] == 500
        assert resp.json()["storageOverageEnabled"] is False

    def test_free_tier_cannot_enable_credit_overage(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from unittest.mock import MagicMock

        from tests.conftest import _default_table_side_effect

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "subscriptions":
                b.execute.return_value = MagicMock(
                    data=[{"user_id": "00000000-0000-0000-0000-000000000001", "tier": "free", "status": "active"}],
                    count=1,
                )
            return b

        mock_supabase.table.side_effect = side_effect
        resp = client.post("/me/billing-prefs", json={"overage_enabled": True})
        assert resp.status_code == 400

    def test_free_tier_can_disable_overage(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from unittest.mock import MagicMock

        from tests.conftest import _default_table_side_effect

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "subscriptions":
                b.execute.return_value = MagicMock(
                    data=[
                        {
                            "user_id": "00000000-0000-0000-0000-000000000001",
                            "tier": "free",
                            "status": "active",
                            "overage_enabled": False,
                            "overage_cap_credits": None,
                            "storage_overage_enabled": False,
                        }
                    ],
                    count=1,
                )
            return b

        mock_supabase.table.side_effect = side_effect
        resp = client.post("/me/billing-prefs", json={"overage_enabled": False})
        assert resp.status_code == 200
        assert resp.json()["overageEnabled"] is False

    def test_missing_subscriptions_row_creates_it(self, client, mock_supabase, monkeypatch):
        """No subscriptions row: the endpoint upserts one instead of a silent no-op."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from unittest.mock import MagicMock

        from tests.conftest import _default_table_side_effect

        sub_builders = []

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[], count=0)
                # MockQueryBuilder.upsert is a plain chainable method; replace it
                # with a MagicMock so the payload can be asserted.
                b.upsert = MagicMock(return_value=b)
                sub_builders.append(b)
            return b

        mock_supabase.table.side_effect = side_effect
        resp = client.post("/me/billing-prefs", json={"storage_overage_enabled": True})
        assert resp.status_code == 200

        upserts = [b.upsert for b in sub_builders if b.upsert.called]
        assert len(upserts) == 1
        args, kwargs = upserts[0].call_args
        # Plain shape: user_id + provided prefs only — tier/status/overage columns
        # all have NOT NULL DEFAULTs in the schema, so the insert path is safe.
        assert args[0] == {"storage_overage_enabled": True, "user_id": TEST_USER_ID}
        assert kwargs.get("on_conflict") == "user_id"

    def test_sparse_update_response_reflects_row_not_request(self, client, mock_supabase, monkeypatch):
        """Posting only a cap returns the row's actual overage_enabled, not null."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from unittest.mock import MagicMock

        from tests.conftest import _default_table_side_effect

        persisted = {
            "user_id": TEST_USER_ID,
            "tier": "pro",
            "status": "active",
            "overage_enabled": False,
            "overage_cap_credits": 250,
            "storage_overage_enabled": True,
        }

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[persisted], count=1)
            return b

        mock_supabase.table.side_effect = side_effect
        resp = client.post("/me/billing-prefs", json={"overage_cap_credits": 250})
        assert resp.status_code == 200
        data = resp.json()
        assert data["overageEnabled"] is False  # from the persisted row, not the request
        assert data["overageCapCredits"] == 250
        assert data["storageOverageEnabled"] is True

    def test_empty_payload_400(self, client, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        resp = client.post("/me/billing-prefs", json={})
        assert resp.status_code == 400

    def test_negative_cap_422(self, client, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        resp = client.post("/me/billing-prefs", json={"overage_cap_credits": -5})
        assert resp.status_code == 422


class TestCreditUsageEndpoint:
    def test_disabled_returns_enabled_false(self, client, mock_supabase):
        resp = client.get("/me/credits/usage")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False}
        tables_touched = {c.args[0] for c in mock_supabase.table.call_args_list}
        assert "credit_ledger" not in tables_touched

    def test_enabled_aggregates_per_tool_spend(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from tests.conftest import _default_table_side_effect

        ledger_rows = [
            {"action": "oneclick_run", "delta": -21, "kind": "debit", "metadata": {}},
            {"action": "zoe_message", "delta": -3, "kind": "debit", "metadata": {}},
            {"action": "oneclick_run", "delta": 0, "kind": "overage_debit", "metadata": {"credits_billed": 21}},
        ]

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "credit_ledger":
                b.execute.return_value = MagicMock(data=ledger_rows, count=len(ledger_rows))
            return b

        mock_supabase.table.side_effect = side_effect
        resp = client.get("/me/credits/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True

        tools = {t["action"]: t for t in data["tools"]}
        assert tools["oneclick_run"]["count"] == 2
        assert tools["oneclick_run"]["spent"] == 42
        assert tools["oneclick_run"]["price"] == 21
        assert tools["zoe_message"]["count"] == 1
        assert tools["zoe_message"]["spent"] == 3
        assert tools["zoe_message"]["price"] == 3
        assert tools["registry_parse"]["count"] == 0
        assert tools["registry_parse"]["spent"] == 0
        assert tools["registry_parse"]["price"] == 12

        assert "monthlyGrant" in data
        assert "balance" in data
