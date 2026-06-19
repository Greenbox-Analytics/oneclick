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
        if "work_created" in hogql:  # registry lifecycle
            return {"results": [[6, 4, 2]]}
        if "oneclick_calc_started" in hogql:  # session funnel
            return {"results": [["oneclick", 3, 2, 2, 5, 1]]}
        # Summary query row shape: [tool, opens, completions, openers, converters, last_used]
        return {"results": [["oneclick", 10, 5, 3, 2, "2026-05-16T12:00:00Z"]]}

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
    # First call fires 6 sub-queries (4 core + 2 new); second is fully cached.
    assert calls["n"] == 6
    body = r1.json()
    assert body["active_users"] == 7
    assert body["total_users"] == 24
    assert len(body["sparkline"]) == 2
    assert body["registry_lifecycle"] == {"created": 6, "submitted": 4, "registered": 2}
    assert len(body["funnels"]) == 1
    assert body["funnels"][0]["tool"] == "oneclick"
    assert [s["label"] for s in body["funnels"][0]["steps"]] == ["opened", "started", "completed"]


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

    # Three distinct cache keys x 6 sub-queries each.
    assert calls["n"] == 18


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


def test_parse_results_completion_rate_is_per_user_funnel():
    """completion_rate must be converters/openers (bounded 0-1), not the old
    completions/opens ratio which could exceed 100% on repeat usage."""
    from admin.analytics_router import _parse_results

    # zoe: 1 opener, 1 converter, but 3 raw completions (3 answers in one session).
    # Old formula (3 completions / 1 open) = 300%; new formula = 1 converter / 1 opener = 100%.
    rows = [
        ["zoe", 1, 3, 1, 1, "2026-05-16T12:00:00Z"],
        ["oneclick", 3, 5, 2, 2, "2026-05-16T12:00:00Z"],
        ["registry", 4, 0, 3, 0, "2026-05-16T12:00:00Z"],
    ]
    summary = _parse_results(
        rows,
        active_users=2,
        total_users=5,
        sparkline_rows=[],
        window="7d",
        cohort="testers",
        funnel_rows=[],
        lifecycle_rows=[],
    )

    by_tool = {r.tool: r for r in summary.per_tool}
    assert by_tool["zoe"].completion_rate == 1.0
    assert by_tool["oneclick"].completion_rate == 1.0
    assert by_tool["registry"].completion_rate == 0.0
    # Every per-tool rate is bounded at 100%.
    assert all(0.0 <= r.completion_rate <= 1.0 for r in summary.per_tool)
    # Funnel avg is the mean of named-tool rates with openers > 0: (1 + 1 + 0) / 3.
    assert abs(summary.funnel_completion_avg - (2 / 3)) < 1e-9
    # Raw volume columns are preserved for the headline counts.
    assert summary.tool_actions == 8  # 3 + 5 + 0 raw completions
    assert by_tool["oneclick"].opens == 3 and by_tool["oneclick"].completions == 5


def test_parse_funnels_nesting_labels_and_error_rate():
    """Row shape: [tool, opened, started, completed, completed_events, failed_events]."""
    from admin.analytics_router import _parse_funnels

    # oneclick — 3-step, error_rate = failed/(completed+failed) = 1/(5+1).
    one = _parse_funnels([["oneclick", 3, 2, 2, 5, 1]])[0]
    assert [s.label for s in one.steps] == ["opened", "started", "completed"]
    assert [s.users for s in one.steps] == [3, 2, 2]
    assert abs(one.error_rate - (1 / 6)) < 1e-9
    assert (one.completed_events, one.failed_events) == (5, 1)

    # zoe — 3-step with zoe-specific labels; no failures -> error_rate 0.
    zoe = _parse_funnels([["zoe", 1, 1, 1, 3, 0]])[0]
    assert [s.label for s in zoe.steps] == ["opened", "submitted", "answered"]
    assert zoe.error_rate == 0.0

    # splitsheet — 2-step (started ignored). The real query returns started == opened == 4
    # for splitsheet (the started gate is forced open); the parser ignores it regardless.
    split = _parse_funnels([["splitsheet", 4, 4, 2, 2, 0]])[0]
    assert [s.label for s in split.steps] == ["opened", "generated"]
    assert [s.users for s in split.steps] == [4, 2]

    # error_rate guard: zero finished attempts -> 0.0, not a division error.
    no_attempts = _parse_funnels([["zoe", 2, 0, 0, 0, 0]])[0]
    assert no_attempts.error_rate == 0.0

    # Zero opens omitted; unknown (non-funnel) tool omitted.
    assert _parse_funnels([["oneclick", 0, 0, 0, 0, 0]]) == []
    assert _parse_funnels([["boards", 5, 0, 0, 0, 0]]) == []


def test_parse_registry_lifecycle():
    from admin.analytics_router import _parse_registry_lifecycle

    assert _parse_registry_lifecycle([]) is None
    # countIf always returns one row; an all-zero row must be treated as no data.
    assert _parse_registry_lifecycle([[0, 0, 0]]) is None
    lc = _parse_registry_lifecycle([[6, 4, 2]])
    assert (lc.created, lc.submitted, lc.registered) == (6, 4, 2)


def test_new_query_failure_does_not_blank_the_card(monkeypatch):
    """A funnel/lifecycle query failure must NOT take down the table/sparkline — it should
    leave available=True with the table populated and the funnel section empty."""
    _override_admin()
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "test-key")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")
    from admin.analytics_router import _cache

    _cache.clear()

    async def fake_query(self, hogql):
        # Fail ONLY the two new queries; core queries succeed.
        if "oneclick_calc_started" in hogql or "work_created" in hogql:
            raise httpx.HTTPStatusError(
                "bad query",
                request=httpx.Request("POST", "http://example"),
                response=httpx.Response(400),
            )
        if "GROUP BY day" in hogql:
            return {"results": [["2026-05-16", 5]]}
        if "uniqExact(person_id)" in hogql:
            return {"results": [[7]]}
        if "FROM persons" in hogql:
            return {"results": [[24]]}
        return {"results": [["oneclick", 10, 5, 3, 2, "2026-05-16T12:00:00Z"]]}

    monkeypatch.setattr("admin.posthog_client.PostHogClient.query", fake_query)

    client = TestClient(app)
    resp = client.get("/admin/analytics/summary?window=7d&cohort=testers")
    _clear_overrides()

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True  # card still works
    assert len(body["per_tool"]) == 1  # table populated
    assert body["funnels"] == []  # new section gracefully empty
    assert body["registry_lifecycle"] is None


def test_query_builders_include_env_and_date_filter():
    """Every HogQL builder must filter to dev/prod and date >= cutoff."""
    from admin.analytics_router import (
        _ENV_DATE_FILTER,
        _build_active_users_query,
        _build_registry_lifecycle_query,
        _build_session_funnel_query,
        _build_sparkline_query,
        _build_summary_query,
        _build_total_users_query,
    )

    builders_with_events = [
        _build_summary_query(7, "all"),
        _build_summary_query(None, "testers"),
        _build_active_users_query(30, "all"),
        _build_sparkline_query(7, "testers"),
        _build_session_funnel_query(7, "testers"),
        _build_session_funnel_query(None, "all"),
        _build_registry_lifecycle_query(7, "testers"),
    ]
    for sql in builders_with_events:
        # Assert the exact constant, not a substring — catches typos like `proprties.`
        assert _ENV_DATE_FILTER in sql, f"missing exact env+date filter in: {sql}"
        # Also verify the property access uses the existing `properties.<key>` pattern
        assert "properties.environment" in sql

    # Session funnel must scan only the allowlisted events.
    funnel_sql = _build_session_funnel_query(7, "testers")
    assert "oneclick_calc_started" in funnel_sql
    assert "splitsheet_generation_failed" in funnel_sql

    # _build_total_users_query targets the persons table, not events — env/date filter
    # not applicable. Sanity-check it does NOT contain the predicate (would be wrong).
    persons_sql = _build_total_users_query("all")
    assert "environment" not in persons_sql
