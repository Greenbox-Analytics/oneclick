"""Tests for integration endpoints (connections, Google Drive, Slack, OneClick share).

Acceptance criteria:
1. GET /integrations/connections - list connections (no secrets)
2. Google Drive auth/disconnect
3. Slack channels, settings, webhook
4. OneClick share validation
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

SAMPLE_SLACK_CONNECTION = {
    **SAMPLE_CONNECTION,
    "id": "conn-0000-0000-0000-0000-000000000002",
    "provider": "slack",
    "provider_workspace_id": "T12345",
    "scopes": ["channels:read", "chat:write"],
}

SAMPLE_CHANNEL = {"id": "C12345", "name": "general", "is_private": False}

SAMPLE_NOTIFICATION_SETTING = {
    "id": "ns-0000-0000-0000-0000-000000000001",
    "user_id": TEST_USER_ID,
    "provider": "slack",
    "event_type": "task_created",
    "enabled": True,
    "channel_id": "C12345",
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

    def test_returns_multiple_providers(self, client, mock_supabase):
        """GET /integrations/connections returns connections for multiple providers."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_CONNECTION, SAMPLE_SLACK_CONNECTION])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/integrations/connections")

        assert response.status_code == 200
        providers = {c["provider"] for c in response.json()["connections"]}
        assert providers == {"google_drive", "slack"}


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


# ============================================================
# Slack endpoints
# ============================================================


class TestSlackDisconnect:
    def test_disconnect_returns_success(self, client, mock_supabase):
        """DELETE /integrations/slack/disconnect returns {"success": true}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.delete("/integrations/slack/disconnect")

        assert response.status_code == 200
        assert response.json() == {"success": True}


class TestSlackSettings:
    def test_get_settings_returns_settings_key(self, client, mock_supabase):
        """GET /integrations/slack/settings returns {"settings": [...]}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_NOTIFICATION_SETTING])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/integrations/slack/settings")

        assert response.status_code == 200
        body = response.json()
        assert "settings" in body
        assert isinstance(body["settings"], list)

    def test_get_settings_empty(self, client, mock_supabase):
        """GET /integrations/slack/settings returns empty list when none configured."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/integrations/slack/settings")

        assert response.status_code == 200
        assert response.json()["settings"] == []

    def test_update_settings_returns_success(self, client, mock_supabase):
        """PUT /integrations/slack/settings returns {"success": true}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.put(
            "/integrations/slack/settings",
            json={
                "event_type": "task_created",
                "enabled": True,
                "channel_id": "C12345",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_update_settings_disable_event(self, client, mock_supabase):
        """PUT /integrations/slack/settings with enabled=false."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_NOTIFICATION_SETTING])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.put(
            "/integrations/slack/settings",
            json={
                "event_type": "task_created",
                "enabled": False,
            },
        )

        assert response.status_code == 200
        assert response.json() == {"success": True}


class TestSlackChannels:
    @patch("integrations.slack.router.get_valid_token", new_callable=AsyncMock)
    def test_channels_returns_401_when_not_connected(self, mock_token, client):
        """GET /integrations/slack/channels returns 401 when Slack not connected."""
        mock_token.return_value = None

        response = client.get("/integrations/slack/channels")

        assert response.status_code == 401


class TestSlackWebhook:
    def test_url_verification_challenge(self, client):
        """POST /integrations/slack/webhook handles URL verification."""
        response = client.post(
            "/integrations/slack/webhook",
            json={
                "type": "url_verification",
                "challenge": "test-challenge-string",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"challenge": "test-challenge-string"}

    def test_event_callback_returns_ok(self, client, mock_supabase):
        """POST /integrations/slack/webhook returns ok for event_callback."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.post(
            "/integrations/slack/webhook",
            json={
                "type": "event_callback",
                "event": {
                    "type": "app_mention",
                    "channel": "C12345",
                    "user": "U12345",
                    "text": "Hey <@BOT> check this out",
                    "ts": "1234567890.123456",
                },
            },
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_unknown_event_type_returns_ok(self, client):
        """POST /integrations/slack/webhook returns ok for unknown event types."""
        response = client.post(
            "/integrations/slack/webhook",
            json={"type": "some_other_type"},
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}


# ============================================================
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

    @patch("oneclick.share.get_valid_token", new_callable=AsyncMock)
    def test_share_to_slack_returns_401_when_not_connected(self, mock_token, client):
        """POST /oneclick/share returns 401 when Slack not connected."""
        mock_token.return_value = None

        response = client.post(
            "/oneclick/share",
            json={
                "target": "slack",
                "artist_name": "Test Artist",
                "payments": [],
                "total_payments": 0,
                "channel_id": "C12345",
            },
        )

        assert response.status_code == 401

    @patch("oneclick.share.get_valid_token", new_callable=AsyncMock)
    def test_share_to_slack_requires_channel_id(self, mock_token, client):
        """POST /oneclick/share returns 400 when no channel_id for Slack."""
        mock_token.return_value = "xoxb-fake-token"

        response = client.post(
            "/oneclick/share",
            json={
                "target": "slack",
                "artist_name": "Test Artist",
                "payments": [],
                "total_payments": 0,
            },
        )

        assert response.status_code == 400

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

    @patch("oneclick.share.get_valid_token", new_callable=AsyncMock)
    @patch("integrations.slack.service.upload_file_to_channel", new_callable=AsyncMock)
    def test_share_to_slack_success(self, mock_upload, mock_token, client):
        """POST /oneclick/share target=slack returns success when connected."""
        mock_token.return_value = "xoxb-fake-token"
        mock_upload.return_value = {"ok": True, "file": {"id": "F12345"}}

        response = client.post(
            "/oneclick/share",
            json={
                "target": "slack",
                "artist_name": "Test Artist",
                "payments": [],
                "total_payments": 0,
                "channel_id": "C12345",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["target"] == "slack"
