"""Unit tests for the PostHog dashboard backfill script."""

from unittest.mock import MagicMock

import pytest

from scripts import posthog_apply_env_filter as backfill


def test_mutate_filters_adds_env_and_date():
    """A fresh insight gets env predicate + date_from."""
    insight_filters = {"date_from": "-30d", "properties": []}
    new_filters = backfill._mutate_filters(insight_filters)

    assert new_filters["date_from"] == backfill.POSTHOG_DATA_CUTOFF
    props = new_filters["properties"]
    assert props["type"] == "AND"
    env_values = [v for v in props["values"] if v.get("key") == "environment"]
    assert len(env_values) == 1
    # Compare against the source of truth, not a hardcoded list, so the test
    # stays correct if ENV_FILTER_VALUES is ever extended or reordered.
    assert sorted(env_values[0]["value"]) == sorted(backfill.ENV_FILTER_VALUES)
    assert env_values[0]["operator"] == "exact"
    assert env_values[0]["type"] == "event"


def test_mutate_filters_is_idempotent():
    """Re-applying does not duplicate the environment predicate."""
    insight_filters = {"date_from": "-30d", "properties": []}
    once = backfill._mutate_filters(insight_filters)
    twice = backfill._mutate_filters(once)
    env_values = [v for v in twice["properties"]["values"] if v.get("key") == "environment"]
    assert len(env_values) == 1


def test_mutate_filters_preserves_other_predicates():
    """Existing non-environment property filters survive the mutation."""
    insight_filters = {
        "date_from": "-30d",
        "properties": {
            "type": "AND",
            "values": [{"key": "tool", "value": ["oneclick"], "operator": "exact", "type": "event"}],
        },
    }
    out = backfill._mutate_filters(insight_filters)
    tool_values = [v for v in out["properties"]["values"] if v.get("key") == "tool"]
    assert len(tool_values) == 1
    assert tool_values[0]["value"] == ["oneclick"]


def test_already_patched_requires_env_and_date():
    """_is_already_patched needs BOTH env predicate AND date_from at cutoff."""
    fully_patched = {
        "date_from": backfill.POSTHOG_DATA_CUTOFF,
        "properties": {
            "type": "AND",
            "values": [{"key": "environment", "value": ["dev", "prod"], "operator": "exact", "type": "event"}],
        },
    }
    assert backfill._is_already_patched(fully_patched) is True

    # Half-patched: env added but date missing — must NOT be considered patched
    half_patched_no_date = {
        "properties": {
            "type": "AND",
            "values": [{"key": "environment", "value": ["dev", "prod"], "operator": "exact", "type": "event"}],
        },
    }
    assert backfill._is_already_patched(half_patched_no_date) is False

    # Half-patched: date set but env missing
    half_patched_no_env = {"date_from": backfill.POSTHOG_DATA_CUTOFF, "properties": []}
    assert backfill._is_already_patched(half_patched_no_env) is False

    # Untouched
    not_patched = {"properties": []}
    assert backfill._is_already_patched(not_patched) is False


def test_exits_when_env_missing(monkeypatch):
    monkeypatch.delenv("POSTHOG_PERSONAL_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_PROJECT_ID", raising=False)
    with pytest.raises(SystemExit) as exc:
        backfill.run(dashboard_ids=[1593101], dry_run=True)
    assert exc.value.code == 2


def _make_fake_client(monkeypatch, insight_payload: dict, patches: list):
    """Helper: fake httpx.Client where GET /dashboards/{id}/ returns one tile and
    GET /insights/{id}/ returns insight_payload."""

    def get_(url, **kw):
        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = lambda: None
        if "/dashboards/" in url and "/tiles/" not in url:
            m.json = lambda: {"id": 1, "tiles": [{"insight": {"id": 100}}]}
        else:
            m.json = lambda: insight_payload
        return m

    def patch_(url, **kw):
        patches.append((url, kw.get("json")))
        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = lambda: None
        return m

    fake = MagicMock()
    fake.__enter__ = lambda self: fake
    fake.__exit__ = lambda *a: None
    fake.get.side_effect = get_
    fake.patch.side_effect = patch_
    monkeypatch.setattr(backfill.httpx, "Client", lambda *a, **kw: fake)
    return fake


def test_dry_run_does_not_patch(monkeypatch):
    """In --dry-run mode, the script issues GETs but no PATCHes."""
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "k")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")

    patches: list = []
    _make_fake_client(monkeypatch, {"id": 100, "filters": {"properties": []}}, patches)

    backfill.run(dashboard_ids=[1593101], dry_run=True)
    assert patches == [], "dry-run should not issue PATCH"


def test_query_based_insight_is_skipped(monkeypatch):
    """Insights with non-null `query` field use the newer API; PATCH to filters
    would be a no-op. The script must skip and warn rather than silently fail."""
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "k")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")

    patches: list = []
    _make_fake_client(
        monkeypatch,
        {
            "id": 100,
            "filters": {"properties": []},
            "query": {"kind": "HogQLQuery", "source": {"kind": "EventsQuery"}},
        },
        patches,
    )

    backfill.run(dashboard_ids=[1593101], dry_run=False)
    assert patches == [], "query-based insights must not be PATCHed via filters"


def test_apply_patches_filter_based_insight(monkeypatch):
    """Standard filter-based insight gets PATCHed with the new filter tree."""
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "k")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")

    patches: list = []
    _make_fake_client(
        monkeypatch,
        {"id": 100, "filters": {"properties": []}, "query": None},
        patches,
    )

    backfill.run(dashboard_ids=[1593101], dry_run=False)
    assert len(patches) == 1
    _url, body = patches[0]
    new_filters = body["filters"]
    assert new_filters["date_from"] == backfill.POSTHOG_DATA_CUTOFF
    env_values = [v for v in new_filters["properties"]["values"] if v.get("key") == "environment"]
    assert len(env_values) == 1
