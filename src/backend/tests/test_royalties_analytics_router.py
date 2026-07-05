"""Tests for the analytics router: GET /analytics/overview|artist/{id}|payee/{id}.

All tests mock the Supabase client, analytics_service, and ownership helpers.
No real DB or network calls.

Key invariants verified:
  1. GET /overview → 200 and returns the (mocked) service object.
  2. GET /artist/{artist_id} with a foreign artist → 404.
  3. GET /artist/{artist_id} with an owned artist → 200 and returns the (mocked) service object.
  4. GET /payee/{payee_id} when analytics_service raises PermissionError → 404.
  5. GET /payee/{payee_id} normal path → 200 and returns the (mocked) service object.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-aaa"
ARTIST_ID = "artist-111"
PAYEE_ID = "payee-222"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OVERVIEW_RESULT = {
    "base": "USD",
    "outstanding_total": 500.0,
    "payees_owed_count": 2,
    "drafted_total": 100.0,
    "draft_count": 1,
    "paid_total": 1000.0,
    "paid_last_30d": 200.0,
    "paid_by_month": [],
    "top_owed": [],
    "unconvertible_count": 0,
}

ARTIST_RESULT = {
    "artist_id": ARTIST_ID,
    "base": "USD",
    "summary": {"earned_total": 300.0, "owed_now": 100.0, "paid_total": 200.0},
    "by_month": [],
    "unconvertible_count": 0,
}

PAYEE_RESULT = {
    "payee_id": PAYEE_ID,
    "display_name": "Alice",
    "base": "USD",
    "summary": {"earned_total": 150.0, "paid_total": 50.0, "owed": 100.0},
    "by_month": [],
    "unconvertible_count": 0,
}


def _make_router_client(
    owns_artist: bool = True,
    overview_return=None,
    artist_return=None,
    payee_return=None,
    payee_side_effect=None,
):
    """Build a minimal FastAPI app with the analytics router. Auth/gating bypassed.

    Patches:
      - auth dependency → returns USER_ID
      - gated_feature → no-op
      - _get_supabase → trivial MagicMock
      - oneclick.royalties.analytics_router._verify_owns_artist → controlled return
      - analytics_service.overview → controlled return
      - analytics_service.artist_analytics → controlled return
      - analytics_service.payee_analytics → controlled return / side_effect
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth import get_current_user_id
    from oneclick.royalties.analytics_router import router

    app = FastAPI()
    app.include_router(router)

    async def _mock_user_id():
        return USER_ID

    app.dependency_overrides[get_current_user_id] = _mock_user_id

    gating_patcher = patch("oneclick.royalties.analytics_router.gated_feature", return_value=None)
    supabase_patcher = patch("oneclick.royalties.analytics_router._get_supabase", return_value=MagicMock())
    ownership_patcher = patch(
        "oneclick.royalties.analytics_router._verify_owns_artist",
        return_value=owns_artist,
    )
    overview_patcher = patch(
        "oneclick.royalties.analytics_router.analytics_service.overview",
        return_value=overview_return if overview_return is not None else OVERVIEW_RESULT,
    )
    artist_patcher = patch(
        "oneclick.royalties.analytics_router.analytics_service.artist_analytics",
        return_value=artist_return if artist_return is not None else ARTIST_RESULT,
    )

    if payee_side_effect is not None:
        payee_patcher = patch(
            "oneclick.royalties.analytics_router.analytics_service.payee_analytics",
            side_effect=payee_side_effect,
        )
    else:
        payee_patcher = patch(
            "oneclick.royalties.analytics_router.analytics_service.payee_analytics",
            return_value=payee_return if payee_return is not None else PAYEE_RESULT,
        )

    gating_patcher.start()
    supabase_patcher.start()
    ownership_patcher.start()
    overview_patcher.start()
    artist_patcher.start()
    payee_patcher.start()

    client = TestClient(app, raise_server_exceptions=False)

    yield client

    gating_patcher.stop()
    supabase_patcher.stop()
    ownership_patcher.stop()
    overview_patcher.stop()
    artist_patcher.stop()
    payee_patcher.stop()


# ---------------------------------------------------------------------------
# Tests: GET /overview
# ---------------------------------------------------------------------------


class TestOverviewEndpoint:
    def test_overview_returns_200(self):
        """GET /overview must return 200."""
        for client in _make_router_client():
            resp = client.get("/overview")
            assert resp.status_code == 200

    def test_overview_returns_service_object(self):
        """GET /overview body must match the (mocked) service return value."""
        for client in _make_router_client(overview_return=OVERVIEW_RESULT):
            resp = client.get("/overview")
            assert resp.status_code == 200
            body = resp.json()
            assert body["outstanding_total"] == 500.0
            assert body["payees_owed_count"] == 2

    def test_overview_with_base_query_param(self):
        """GET /overview?base=GBP must pass through and return 200."""
        for client in _make_router_client():
            resp = client.get("/overview?base=GBP")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: GET /artist/{artist_id}
# ---------------------------------------------------------------------------


class TestArtistAnalyticsEndpoint:
    def test_foreign_artist_returns_404(self):
        """When _verify_owns_artist returns False, endpoint must return 404."""
        for client in _make_router_client(owns_artist=False):
            resp = client.get(f"/artist/{ARTIST_ID}")
            assert resp.status_code == 404

    def test_owned_artist_returns_200(self):
        """When _verify_owns_artist returns True, endpoint must return 200."""
        for client in _make_router_client(owns_artist=True):
            resp = client.get(f"/artist/{ARTIST_ID}")
            assert resp.status_code == 200

    def test_owned_artist_returns_service_object(self):
        """200 body must match the (mocked) analytics_service.artist_analytics return value."""
        for client in _make_router_client(owns_artist=True, artist_return=ARTIST_RESULT):
            resp = client.get(f"/artist/{ARTIST_ID}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["artist_id"] == ARTIST_ID
            assert body["summary"]["earned_total"] == 300.0

    def test_artist_with_base_query_param(self):
        """GET /artist/{id}?base=EUR must return 200 for an owned artist."""
        for client in _make_router_client(owns_artist=True):
            resp = client.get(f"/artist/{ARTIST_ID}?base=EUR")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: GET /payee/{payee_id}
# ---------------------------------------------------------------------------


class TestPayeeAnalyticsEndpoint:
    def test_permission_error_returns_404(self):
        """When analytics_service.payee_analytics raises PermissionError, endpoint must return 404."""
        for client in _make_router_client(payee_side_effect=PermissionError("Payee not found")):
            resp = client.get(f"/payee/{PAYEE_ID}")
            assert resp.status_code == 404

    def test_normal_path_returns_200(self):
        """Normal (owned) payee must return 200."""
        for client in _make_router_client():
            resp = client.get(f"/payee/{PAYEE_ID}")
            assert resp.status_code == 200

    def test_normal_path_returns_service_object(self):
        """200 body must match the (mocked) analytics_service.payee_analytics return value."""
        for client in _make_router_client(payee_return=PAYEE_RESULT):
            resp = client.get(f"/payee/{PAYEE_ID}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["payee_id"] == PAYEE_ID
            assert body["display_name"] == "Alice"
            assert body["summary"]["owed"] == 100.0

    def test_payee_with_base_query_param(self):
        """GET /payee/{id}?base=CAD must return 200."""
        for client in _make_router_client():
            resp = client.get(f"/payee/{PAYEE_ID}?base=CAD")
            assert resp.status_code == 200

    def test_service_called_for_owned_payee(self):
        """analytics_service.payee_analytics must be invoked on the normal path."""
        with (
            patch(
                "oneclick.royalties.analytics_router.analytics_service.payee_analytics",
                return_value=PAYEE_RESULT,
            ) as mock_svc,
            patch("oneclick.royalties.analytics_router.gated_feature", return_value=None),
            patch("oneclick.royalties.analytics_router._get_supabase", return_value=MagicMock()),
            patch("oneclick.royalties.analytics_router._verify_owns_artist", return_value=True),
            patch(
                "oneclick.royalties.analytics_router.analytics_service.overview",
                return_value=OVERVIEW_RESULT,
            ),
            patch(
                "oneclick.royalties.analytics_router.analytics_service.artist_analytics",
                return_value=ARTIST_RESULT,
            ),
        ):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from auth import get_current_user_id
            from oneclick.royalties.analytics_router import router

            app = FastAPI()
            app.include_router(router)

            async def _mock_user_id():
                return USER_ID

            app.dependency_overrides[get_current_user_id] = _mock_user_id
            client = TestClient(app, raise_server_exceptions=False)
            client.get(f"/payee/{PAYEE_ID}")
            mock_svc.assert_called_once()
