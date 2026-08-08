"""Tests for integration endpoints (connections, Google Drive, OneClick share).

Acceptance criteria:
1. GET /integrations/connections - list connections (no secrets)
2. Google Drive auth/disconnect
3. OneClick share validation
"""

from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CONNECTION_ID = "conn-0000-0000-0000-0000-000000000001"

SAMPLE_CONNECTION = {
    "id": CONNECTION_ID,
    "user_id": TEST_USER_ID,
    "provider": "google_drive",
    "status": "active",
    "provider_user_id": "google-user-123",
    "provider_workspace_id": None,
    "scopes": ["https://www.googleapis.com/auth/drive.file"],
    "created_at": "2026-04-10T00:00:00+00:00",
    "updated_at": "2026-04-10T00:00:00+00:00",
}


# ============================================================
# GET /integrations/connections
# ============================================================


class TestListConnections:
    def test_returns_connections_key(self, client, mock_supabase):
        """GET /integrations/connections returns {"connections": [...]}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_CONNECTION])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/integrations/connections")

        assert response.status_code == 200
        body = response.json()
        assert "connections" in body
        assert isinstance(body["connections"], list)

    def test_returns_empty_when_no_connections(self, client, mock_supabase):
        """GET /integrations/connections returns empty list for new user."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/integrations/connections")

        assert response.status_code == 200
        assert response.json()["connections"] == []

    def test_returns_connection_fields(self, client, mock_supabase):
        """GET /integrations/connections returns provider, status, timestamps."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_CONNECTION])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/integrations/connections")

        assert response.status_code == 200
        conn = response.json()["connections"][0]
        assert conn["provider"] == "google_drive"
        assert conn["status"] == "active"
        assert "created_at" in conn
        # Encrypted tokens should NOT be present
        assert "access_token_encrypted" not in conn
        assert "refresh_token_encrypted" not in conn


# ============================================================
# Google Drive endpoints
# ============================================================


class TestGoogleDriveDisconnect:
    def test_disconnect_returns_success(self, client, mock_supabase):
        """DELETE /integrations/google-drive/disconnect returns {"success": true}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.delete("/integrations/google-drive/disconnect")

        assert response.status_code == 200
        assert response.json() == {"success": True}


class TestGoogleDriveAuth:
    @patch("integrations.google_drive.router.build_auth_url")
    def test_auth_returns_url(self, mock_build, client):
        """GET /integrations/google-drive/auth returns {"auth_url": ...}."""
        mock_build.return_value = "https://accounts.google.com/o/oauth2/v2/auth?test=1"

        response = client.get("/integrations/google-drive/auth")

        assert response.status_code == 200
        body = response.json()
        assert "auth_url" in body
        assert body["auth_url"].startswith("https://")


class TestGoogleDriveBrowse:
    @patch("integrations.google_drive.router.get_valid_token", new_callable=AsyncMock)
    def test_browse_returns_401_when_not_connected(self, mock_token, client):
        """GET /integrations/google-drive/browse returns 401 when Drive not connected."""
        mock_token.return_value = None

        response = client.get("/integrations/google-drive/browse")

        assert response.status_code == 401


# OneClick Share
# ============================================================


class TestOneClickShare:
    @patch("oneclick.share.get_valid_token", new_callable=AsyncMock)
    def test_share_to_drive_returns_401_when_not_connected(self, mock_token, client):
        """POST /oneclick/share returns 401 when Drive not connected."""
        mock_token.return_value = None

        response = client.post(
            "/oneclick/share",
            json={
                "target": "drive",
                "artist_name": "Test Artist",
                "payments": [],
                "total_payments": 0,
            },
        )

        assert response.status_code == 401

    def test_share_invalid_target_returns_422(self, client):
        """POST /oneclick/share with missing required fields returns 422."""
        response = client.post(
            "/oneclick/share",
            json={"target": "invalid"},
        )

        assert response.status_code == 422

    @patch("oneclick.share.get_valid_token", new_callable=AsyncMock)
    @patch("integrations.google_drive.service.export_pdf_to_drive", new_callable=AsyncMock)
    def test_share_to_drive_success(self, mock_export, mock_token, client):
        """POST /oneclick/share target=drive returns success when connected."""
        mock_token.return_value = "ya29.fake-token"
        mock_export.return_value = {"id": "drive-file-id", "name": "report.pdf"}

        response = client.post(
            "/oneclick/share",
            json={
                "target": "drive",
                "artist_name": "Test Artist",
                "payments": [
                    {
                        "song_title": "Hit Song",
                        "party_name": "Producer",
                        "role": "producer",
                        "royalty_type": "master",
                        "percentage": 25.0,
                        "amount_to_pay": 2500.00,
                    }
                ],
                "total_payments": 2500.00,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["target"] == "drive"


# ============================================================
# SP3 Integration OAuth gating
# ============================================================


class TestIntegrationGated:
    """SP3: Drive OAuth start is ungated."""

    def _denied_service(self, name: str):
        from subscriptions.models import CheckResult

        svc = MagicMock()
        svc.can.return_value = CheckResult(
            allowed=False,
            reason=f"{name.capitalize()} integration is a Pro feature.",
            upgrade_required=True,
        )
        return svc

    @patch("integrations.google_drive.router.build_auth_url")
    def test_drive_oauth_start_free_succeeds(self, mock_build, client, monkeypatch):
        """Drive OAuth start is NOT gated — should succeed regardless of tier."""
        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        # Simulate a Free user where USE_INTEGRATION is denied —
        # Drive should never call gated_feature so this should NOT cause a 402.
        svc = MagicMock()
        svc.can.return_value = CheckResult(allowed=False, reason="Pro feature.", upgrade_required=True)
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        mock_build.return_value = "https://accounts.google.com/o/oauth2/v2/auth?test=1"

        resp = client.get("/integrations/google-drive/auth")
        assert resp.status_code != 402, (
            "Drive OAuth start should not be gated; got 402. "
            "Check that google_drive/router.py does NOT call gated_feature."
        )
        assert resp.status_code == 200
