"""Tests for SplitSheet step event instrumentation."""

import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from auth import get_current_user_id
from main import app

TEST_USER_ID = "33333333-3333-3333-3333-333333333333"


def _patches(monkeypatch, captured):
    monkeypatch.setattr(
        "splitsheet.router.analytics_capture",
        lambda uid, event, props=None: captured.append((event, dict(props or {}))),
    )
    monkeypatch.setattr("splitsheet.router.gated_split_sheet", lambda *a, **kw: None)
    monkeypatch.setattr(
        "splitsheet.router._get_entitlements_service",
        lambda: MagicMock(increment_usage=lambda *a, **kw: None),
    )


def test_generate_split_sheet_fires_generated_event(monkeypatch):
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    captured = []
    _patches(monkeypatch, captured)
    monkeypatch.setattr(
        "splitsheet.router.generate_split_sheet_pdf",
        lambda **kw: io.BytesIO(b"fakepdf"),
    )

    payload = {
        "work_title": "Test Song",
        "work_type": "single",
        "split_type": "both",
        "date": "2026-05-17",
        "format": "pdf",
        "contributors": [
            {"name": "A", "role": "writer"},
            {"name": "B", "role": "writer"},
        ],
    }
    try:
        client = TestClient(app)
        resp = client.post("/splitsheet/generate", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    generated = [c for c in captured if c[0] == "splitsheet_generated"]
    assert len(generated) == 1
    assert generated[0][1]["format"] == "pdf"
    assert generated[0][1]["collaborator_count"] == 2
    # Existing tool_used capture preserved
    tool_used = [c for c in captured if c[0] == "tool_used"]
    assert len(tool_used) == 1


def test_generate_split_sheet_fires_failed_on_exception(monkeypatch):
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    captured = []
    _patches(monkeypatch, captured)

    def explode(**kw):
        raise RuntimeError("pdf gen broken")

    monkeypatch.setattr("splitsheet.router.generate_split_sheet_pdf", explode)

    payload = {
        "work_title": "X",
        "work_type": "single",
        "split_type": "both",
        "date": "2026-05-17",
        "format": "pdf",
        "contributors": [{"name": "A", "role": "writer"}],
    }
    try:
        client = TestClient(app)
        resp = client.post("/splitsheet/generate", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 500
    failed = [c for c in captured if c[0] == "splitsheet_generation_failed"]
    assert len(failed) == 1
    assert failed[0][1]["error_code"] == "RuntimeError"
