"""Tests for integration analytics events.

Verifies:
- OAuth callbacks fire `integration_connected` on success
- OAuth callbacks fire `integration_connect_failed` on exception (and re-raise)
- Use-sites fire `integration_used` for each provider
"""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import main

# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------


class TestGoogleDriveCallbackAnalytics:
    def test_callback_fires_connected_on_success(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            "integrations.google_drive.router.analytics_capture",
            lambda uid, event, props=None: captured.append((event, dict(props or {}))),
        )
        monkeypatch.setattr(
            "integrations.google_drive.router.verify_oauth_state",
            lambda state: {"user_id": "u-1"},
        )
        monkeypatch.setattr(
            "integrations.google_drive.router.exchange_code_for_tokens",
            AsyncMock(return_value={"access_token": "x", "refresh_token": "y", "expires_in": 3600}),
        )
        monkeypatch.setattr(
            "integrations.google_drive.router.store_connection",
            AsyncMock(return_value=None),
        )

        client = TestClient(main.app)
        client.get(
            "/integrations/google-drive/callback?code=xyz&state=abc",
            follow_redirects=False,
        )

        connected = [c for c in captured if c[0] == "integration_connected"]
        assert len(connected) == 1, f"expected integration_connected, got {captured}"
        assert connected[0][1]["tool"] == "drive"

    def test_callback_fires_failed_on_token_exchange_exception(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            "integrations.google_drive.router.analytics_capture",
            lambda uid, event, props=None: captured.append((event, dict(props or {}))),
        )
        monkeypatch.setattr(
            "integrations.google_drive.router.verify_oauth_state",
            lambda state: {"user_id": "u-1"},
        )
        monkeypatch.setattr(
            "integrations.google_drive.router.exchange_code_for_tokens",
            AsyncMock(side_effect=RuntimeError("token exchange failed")),
        )

        client = TestClient(main.app)
        resp = client.get(
            "/integrations/google-drive/callback?code=xyz&state=abc",
            follow_redirects=False,
        )

        # The exception is mapped to a 400 HTTPException — must still re-raise
        # (i.e. the user sees the error, not a silent redirect).
        assert resp.status_code == 400

        failed = [c for c in captured if c[0] == "integration_connect_failed"]
        assert len(failed) == 1, f"expected integration_connect_failed, got {captured}"
        assert failed[0][1]["tool"] == "drive"
        # error_code is the original exception class name
        assert failed[0][1]["error_code"] == "RuntimeError"


class TestGoogleDriveImportAnalytics:
    def test_import_fires_integration_used(self, client, monkeypatch):
        captured = []
        monkeypatch.setattr(
            "integrations.google_drive.router.analytics_capture",
            lambda uid, event, props=None: captured.append((event, dict(props or {}))),
        )
        monkeypatch.setattr(
            "integrations.google_drive.router.get_valid_token",
            AsyncMock(return_value="ya29.fake-token"),
        )
        monkeypatch.setattr(
            "integrations.google_drive.service.import_drive_file",
            AsyncMock(return_value={"id": "f1", "name": "x.pdf"}),
        )
        # Caller must be a project member for the import membership guard to pass.
        monkeypatch.setattr("projects.service.get_user_role", AsyncMock(return_value="editor"))

        resp = client.post(
            "/integrations/google-drive/import",
            json={
                "drive_file_id": "drive-file-id",
                "project_id": "00000000-0000-0000-0000-000000000010",
                "file_type": "contract",
            },
        )

        assert resp.status_code == 200, resp.text

        used = [c for c in captured if c[0] == "integration_used"]
        assert len(used) == 1, f"expected integration_used, got {captured}"
        assert used[0][1]["tool"] == "drive"
        assert used[0][1]["action"] == "file_imported"
