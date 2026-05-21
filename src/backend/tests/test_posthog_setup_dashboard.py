"""Smoke tests for the PostHog dashboard setup script."""

import json
from unittest.mock import MagicMock

import pytest

from scripts import posthog_setup_dashboard as setup


def test_exits_when_env_missing(monkeypatch):
    monkeypatch.delenv("POSTHOG_PERSONAL_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_PROJECT_ID", raising=False)
    with pytest.raises(SystemExit) as exc:
        setup.run(env="test", adopt=False)
    assert exc.value.code == 2


def _make_fake_client(call_log: list, next_id: dict):
    """Build a MagicMock that records calls and returns fresh ids for POST."""

    def post(url, **kwargs):
        call_log.append(("POST", url))
        rid = f"id-{next_id['n']}"
        next_id["n"] += 1
        m = MagicMock()
        m.status_code = 201
        m.raise_for_status = lambda: None
        m.json = lambda: {"id": rid}
        return m

    def patch_(url, **kwargs):
        call_log.append(("PATCH", url))
        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = lambda: None
        m.json = lambda: {"id": "existing"}
        return m

    def get_(url, **kwargs):
        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = lambda: None
        m.json = lambda: {"results": []}
        return m

    fake = MagicMock()
    fake.__enter__ = lambda self: fake
    fake.__exit__ = lambda *a: None
    fake.post.side_effect = post
    fake.patch.side_effect = patch_
    fake.get.side_effect = get_
    return fake


def test_creates_entities_on_first_run(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "key")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")
    monkeypatch.setattr(setup, "STATE_DIR", tmp_path)

    call_log: list = []
    next_id = {"n": 1}
    fake = _make_fake_client(call_log, next_id)
    monkeypatch.setattr(setup.httpx, "Client", lambda *a, **kw: fake)

    setup.run(env="test", adopt=False)

    state_file = tmp_path / ".posthog_dashboard_state.test.json"
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert len(state["cohorts"]) == len(setup.COHORTS)
    assert len(state["insights"]) == len(setup.INSIGHTS)
    assert len(state["dashboards"]) == len(setup.DASHBOARDS)
    # Tile-mounting pass: every insight bound to every dashboard.
    assert "dashboard_tiles" in state
    for dash_logical in setup.DASHBOARDS:
        assert dash_logical in state["dashboard_tiles"]
        assert len(state["dashboard_tiles"][dash_logical]) == len(setup.INSIGHTS)


def test_second_run_patches_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "key")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")
    monkeypatch.setattr(setup, "STATE_DIR", tmp_path)

    # Pre-populate state file with one cohort ID + empty tile map
    state_file = tmp_path / ".posthog_dashboard_state.test.json"
    state_file.write_text(
        json.dumps(
            {
                "cohorts": {"testers": "id-existing"},
                "insights": {},
                "dashboards": {},
                "dashboard_tiles": {},
            }
        )
    )

    posts: list = []
    patches: list = []

    def post(url, **kwargs):
        posts.append(url)
        m = MagicMock()
        m.status_code = 201
        m.raise_for_status = lambda: None
        m.json = lambda: {"id": "new-id"}
        return m

    def patch_(url, **kwargs):
        patches.append(url)
        m = MagicMock()
        m.status_code = 200
        m.raise_for_status = lambda: None
        m.json = lambda: {"id": "id-existing"}
        return m

    fake = MagicMock()
    fake.__enter__ = lambda self: fake
    fake.__exit__ = lambda *a: None
    fake.post.side_effect = post
    fake.patch.side_effect = patch_
    fake.get.side_effect = lambda url, **kw: MagicMock(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"results": []},
    )
    monkeypatch.setattr(setup.httpx, "Client", lambda *a, **kw: fake)

    setup.run(env="test", adopt=False)

    # The pre-existing cohort was PATCHed, not POSTed.
    assert any("id-existing" in url for url in patches), f"PATCH not issued for known cohort: {patches}"


def test_with_env_filter_adds_env_predicate_and_date_from():
    """_with_env_filter must inject env property filter + date_from under `filters:`."""
    base = {"name": "DAU by tool", "kind": "trends", "events": [{"id": "tool_opened"}]}
    out = setup._with_env_filter(base)
    # Original root keys preserved
    assert out["name"] == "DAU by tool"
    assert out["kind"] == "trends"
    # Additions live under filters: per PostHog's documented insight shape.
    filters = out["filters"]
    assert filters["date_from"] == setup.POSTHOG_DATA_CUTOFF
    props = filters["properties"]
    assert props["type"] == "AND"
    env_values = [v for v in props["values"] if v.get("key") == "environment"]
    assert len(env_values) == 1
    assert env_values[0]["value"] == ["dev", "prod"]
    assert env_values[0]["operator"] == "exact"
    assert env_values[0]["type"] == "event"


def test_with_env_filter_is_idempotent():
    """Applying twice does not duplicate the environment predicate."""
    base = {"name": "x", "kind": "trends", "events": [{"id": "tool_opened"}]}
    once = setup._with_env_filter(base)
    twice = setup._with_env_filter(once)
    env_values = [v for v in twice["filters"]["properties"]["values"] if v.get("key") == "environment"]
    assert len(env_values) == 1, "duplicate environment filter on re-apply"


def test_with_env_filter_preserves_prior_filters():
    """Pre-existing filters.properties survive; we add alongside, not replace."""
    base = {
        "name": "x",
        "kind": "trends",
        "filters": {
            "date_from": "-30d",
            "properties": {
                "type": "AND",
                "values": [{"key": "tool", "value": ["oneclick"], "operator": "exact", "type": "event"}],
            },
        },
    }
    out = setup._with_env_filter(base)
    # date_from gets overridden to the cutoff (intentional)
    assert out["filters"]["date_from"] == setup.POSTHOG_DATA_CUTOFF
    # Pre-existing tool predicate survives
    tool_values = [v for v in out["filters"]["properties"]["values"] if v.get("key") == "tool"]
    assert len(tool_values) == 1
    assert tool_values[0]["value"] == ["oneclick"]


def test_all_insights_use_env_filter_on_run(monkeypatch, tmp_path):
    """End-to-end: every insight POST payload contains filters.env + filters.date_from.

    Assumes empty starting state (tmp_path) so every insight goes through POST.
    If a populated state file were present, insights would PATCH instead and
    posted_insight_bodies would be empty — the assertion would pass vacuously.
    """
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "key")
    monkeypatch.setenv("POSTHOG_PROJECT_ID", "1")
    monkeypatch.setattr(setup, "STATE_DIR", tmp_path)

    posted_insight_bodies: list[dict] = []
    next_id = {"n": 1}

    def post(url, **kwargs):
        if "/insights/" in url:
            posted_insight_bodies.append(kwargs.get("json", {}))
        rid = f"id-{next_id['n']}"
        next_id["n"] += 1
        m = MagicMock()
        m.status_code = 201
        m.raise_for_status = lambda: None
        m.json = lambda: {"id": rid}
        return m

    fake = MagicMock()
    fake.__enter__ = lambda self: fake
    fake.__exit__ = lambda *a: None
    fake.post.side_effect = post
    fake.patch.side_effect = lambda *a, **kw: MagicMock(
        status_code=200, raise_for_status=lambda: None, json=lambda: {"id": "x"}
    )
    fake.get.side_effect = lambda *a, **kw: MagicMock(
        status_code=200, raise_for_status=lambda: None, json=lambda: {"results": []}
    )
    monkeypatch.setattr(setup.httpx, "Client", lambda *a, **kw: fake)

    setup.run(env="test", adopt=False)

    assert len(posted_insight_bodies) == len(setup.INSIGHTS)
    for body in posted_insight_bodies:
        filters = body.get("filters", {})
        assert filters.get("date_from") == setup.POSTHOG_DATA_CUTOFF
        env_values = [v for v in filters["properties"]["values"] if v.get("key") == "environment"]
        assert len(env_values) == 1
