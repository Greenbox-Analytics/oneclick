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
