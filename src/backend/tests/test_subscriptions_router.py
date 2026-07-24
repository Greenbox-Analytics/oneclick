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
