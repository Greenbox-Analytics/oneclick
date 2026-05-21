"""Tests for /admin/analytics/summary."""

import httpx
from fastapi.testclient import TestClient

from main import app


def _override_admin():
    from subscriptions.admin_auth import require_admin

    app.dependency_overrides[require_admin] = lambda: "admin@example.com"


def _clear_overrides():
    app.dependency_overrides.clear()


def _fake_query_factory():
    """Returns a fake .query() coroutine + a call counter.

    Dispatches different results based on the HogQL query content so each
    sub-query gets a sensible response.
    """
    calls = {"n": 0}

    async def fake_query(self, hogql):
        calls["n"] += 1
        if "uniqExact(person_id)" in hogql and "GROUP BY day" in hogql:
            return {"results": [["2026-05-15", 3], ["2026-05-16", 5]]}
        if "uniqExact(person_id)" in hogql:
            return {"results": [[7]]}
        if "FROM persons" in hogql:
            return {"results": [[24]]}
        return {"results": [["oneclick", 10, 5, "2026-05-16T12:00:00Z"]]}

    return fake_query, calls


def test_returns_unavailable_when_key_missing(monkeypatch):
    _override_admin()
    monkeypatch.delenv("POSTHOG_PERSONAL_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_PROJECT_ID", raising=False)
    from admin.analytics_router import _cache

    _cache.clear()

    client = TestClient(app)
    resp = client.get("/admin/analytics/summary")
    _clear_overrides()

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert "POSTHOG_PERSONAL_API_KEY" in (body.get("reason") or "")


def test_cache_hits_within_ttl(monkeypatch):
    _override_admin()
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "test-key")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")
    from admin.analytics_router import _cache

    _cache.clear()

    fake_query, calls = _fake_query_factory()
    monkeypatch.setattr("admin.posthog_client.PostHogClient.query", fake_query)

    client = TestClient(app)
    r1 = client.get("/admin/analytics/summary?window=7d&cohort=testers")
    r2 = client.get("/admin/analytics/summary?window=7d&cohort=testers")
    _clear_overrides()

    assert r1.status_code == 200 and r2.status_code == 200
    # First call fires 4 sub-queries; second is fully cached.
    assert calls["n"] == 4
    body = r1.json()
    assert body["active_users"] == 7
    assert body["total_users"] == 24
    assert len(body["sparkline"]) == 2


def test_cache_keyed_on_window_and_cohort(monkeypatch):
    _override_admin()
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "test-key")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")
    from admin.analytics_router import _cache

    _cache.clear()

    fake_query, calls = _fake_query_factory()
    monkeypatch.setattr("admin.posthog_client.PostHogClient.query", fake_query)

    client = TestClient(app)
    client.get("/admin/analytics/summary?window=7d&cohort=testers")
    client.get("/admin/analytics/summary?window=30d&cohort=testers")
    client.get("/admin/analytics/summary?window=7d&cohort=all")
    _clear_overrides()

    # Three distinct cache keys x 4 sub-queries each.
    assert calls["n"] == 12


def test_non_admin_returns_403_or_401(monkeypatch):
    # No override - require_admin runs the real check.
    # In this codebase, when caller has no email and ADMIN_EMAILS is empty,
    # require_admin can return 500 (operator misconfig signal); otherwise 403.
    monkeypatch.delenv("ADMIN_EMAILS", raising=False)
    client = TestClient(app)
    resp = client.get("/admin/analytics/summary")
    # Accept any auth-style failure code
    assert resp.status_code in (401, 403, 500)


def test_returns_unavailable_when_posthog_query_fails(monkeypatch):
    _override_admin()
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "test-key")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")
    from admin.analytics_router import _cache

    _cache.clear()

    async def fake_query(self, hogql):
        raise httpx.RequestError("PostHog unreachable", request=httpx.Request("POST", "http://example"))

    monkeypatch.setattr("admin.posthog_client.PostHogClient.query", fake_query)

    client = TestClient(app)
    resp = client.get("/admin/analytics/summary?window=7d&cohort=testers")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert "posthog_query_failed" in (body.get("reason") or "")
    # Not cached — second call should retry
    resp2 = client.get("/admin/analytics/summary?window=7d&cohort=testers")
    assert resp2.status_code == 200
    assert resp2.json()["available"] is False
    _clear_overrides()


def test_query_builders_include_env_and_date_filter():
    """Every HogQL builder must filter to dev/prod and date >= cutoff."""
    from admin.analytics_router import (
        _ENV_DATE_FILTER,
        _build_active_users_query,
        _build_sparkline_query,
        _build_summary_query,
        _build_total_users_query,
    )

    builders_with_events = [
        _build_summary_query(7, "all"),
        _build_summary_query(None, "testers"),
        _build_active_users_query(30, "all"),
        _build_sparkline_query(7, "testers"),
    ]
    for sql in builders_with_events:
        # Assert the exact constant, not a substring — catches typos like `proprties.`
        assert _ENV_DATE_FILTER in sql, f"missing exact env+date filter in: {sql}"
        # Also verify the property access uses the existing `properties.<key>` pattern
        assert "properties.environment" in sql

    # _build_total_users_query targets the persons table, not events — env/date filter
    # not applicable. Sanity-check it does NOT contain the predicate (would be wrong).
    persons_sql = _build_total_users_query("all")
    assert "environment" not in persons_sql
