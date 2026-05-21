"""Tests for GET /admin/analytics/behavior endpoint."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


@pytest.fixture
def admin_client(monkeypatch):
    from fastapi.testclient import TestClient

    import main
    from subscriptions.admin_auth import require_admin

    main.app.dependency_overrides[require_admin] = lambda: "admin@example.com"
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


class TestBehaviorEndpoint:
    def test_returns_unavailable_when_posthog_unconfigured(self, admin_client, monkeypatch):
        from admin import analytics_router

        if hasattr(analytics_router, "_behavior_cache"):
            analytics_router._behavior_cache.clear()

        unconfigured = MagicMock()
        unconfigured.configured = False
        monkeypatch.setattr(analytics_router, "PostHogClient", lambda: unconfigured)

        resp = admin_client.get("/admin/analytics/behavior?window=7d&cohort=testers")
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert "POSTHOG_PERSONAL_API_KEY" in body["reason"]

    def test_returns_unavailable_on_http_error(self, admin_client, monkeypatch):
        from admin import analytics_router

        if hasattr(analytics_router, "_behavior_cache"):
            analytics_router._behavior_cache.clear()

        client = MagicMock()
        client.configured = True
        client.query = AsyncMock(side_effect=httpx.RequestError("boom"))
        monkeypatch.setattr(analytics_router, "PostHogClient", lambda: client)

        resp = admin_client.get("/admin/analytics/behavior?window=7d&cohort=all")
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert "posthog_query_failed" in body["reason"]

    def test_parses_successful_response(self, admin_client, monkeypatch):
        from admin import analytics_router

        if hasattr(analytics_router, "_behavior_cache"):
            analytics_router._behavior_cache.clear()

        client = MagicMock()
        client.configured = True

        async def _query(q: str):
            if "lagInFrame" in q:
                return {"results": [["/dashboard", "/projects", 12], ["/projects", "/tools/oneclick", 7]]}
            if "toDate" in q:
                return {"results": [["2026-05-18", 50, 12], ["2026-05-19", 65, 15]]}
            if "page_time_spent" in q:
                return {"results": [["/dashboard", 100, 25, 4500, 0.12], ["/projects", 80, 22, 3200, 0.20]]}
            return {"results": [[180, 30, 6.0]]}

        client.query = AsyncMock(side_effect=_query)
        monkeypatch.setattr(analytics_router, "PostHogClient", lambda: client)

        resp = admin_client.get("/admin/analytics/behavior?window=7d&cohort=testers")
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is True
        assert body["total_pageviews"] == 180
        assert body["unique_visitors"] == 30
        assert body["pageviews_per_visitor"] == 6.0
        assert len(body["top_pages"]) == 2
        assert body["top_pages"][0]["path"] == "/dashboard"
        assert body["top_pages"][0]["avg_dwell_ms"] == 4500
        assert len(body["top_flows"]) == 2
        assert body["top_flows"][0]["from_path"] == "/dashboard"
        assert body["top_flows"][0]["to_path"] == "/projects"

    def test_cache_hit_skips_queries(self, admin_client, monkeypatch):
        from admin import analytics_router
        from admin.analytics_models import BehaviorSummary

        if hasattr(analytics_router, "_behavior_cache"):
            analytics_router._behavior_cache.clear()
        analytics_router._behavior_cache[("7d", "testers")] = BehaviorSummary(
            available=True, window="7d", cohort="testers", total_pageviews=999, unique_visitors=10
        )

        called = {"n": 0}

        def _make():
            called["n"] += 1
            return MagicMock()

        monkeypatch.setattr(analytics_router, "PostHogClient", _make)

        resp = admin_client.get("/admin/analytics/behavior?window=7d&cohort=testers")
        assert resp.status_code == 200
        assert resp.json()["total_pageviews"] == 999
        assert called["n"] == 0
