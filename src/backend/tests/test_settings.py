"""Tests for workspace settings endpoints.

Acceptance criteria:
1. GET /settings returns settings (creating defaults if none exist)
2. PUT /settings updates settings and returns updated record
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SETTINGS_RECORD = {
    "user_id": TEST_USER_ID,
    "board_period": "weekly",
    "custom_period_days": None,
    "calendar_view": "month",
    "timezone": "UTC",
    "use_24h_time": False,
    "created_at": "2025-01-01T00:00:00+00:00",
    "updated_at": "2025-01-01T00:00:00+00:00",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _builder(data: list):
    """Return a MockQueryBuilder pre-loaded with the given data list."""
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=data, count=len(data))
    return b


# ---------------------------------------------------------------------------
# GET /settings
# ---------------------------------------------------------------------------


class TestGetSettings:
    """GET /settings returns workspace settings for the authenticated user."""

    def test_returns_existing_settings(self, client, mock_supabase):
        """Returns the settings row when one already exists."""
        mock_supabase.table.side_effect = lambda name: _builder([SETTINGS_RECORD])

        response = client.get("/settings")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == TEST_USER_ID
        assert data["board_period"] == "weekly"
        assert data["calendar_view"] == "month"

    def test_creates_defaults_when_no_settings_exist(self, client, mock_supabase):
        """Creates and returns default settings when none exist for the user."""
        default_record = {"user_id": TEST_USER_ID}

        call_idx = [0]

        def _side_effect(name):
            # First call: SELECT returns empty (no existing settings)
            # Second call: INSERT returns the default record
            b = MockQueryBuilder()
            if call_idx[0] == 0:
                b.execute.return_value = MagicMock(data=[], count=0)
            else:
                b.execute.return_value = MagicMock(data=[default_record], count=1)
            call_idx[0] += 1
            return b

        mock_supabase.table.side_effect = _side_effect

        response = client.get("/settings")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == TEST_USER_ID

    def test_returns_200_status_code(self, client, mock_supabase):
        """GET /settings always returns HTTP 200."""
        mock_supabase.table.side_effect = lambda name: _builder([SETTINGS_RECORD])

        response = client.get("/settings")

        assert response.status_code == 200

    def test_returns_all_settings_fields(self, client, mock_supabase):
        """Response includes all expected settings fields."""
        mock_supabase.table.side_effect = lambda name: _builder([SETTINGS_RECORD])

        response = client.get("/settings")

        data = response.json()
        assert "user_id" in data
        assert "board_period" in data
        assert "calendar_view" in data
        assert "timezone" in data
        assert "use_24h_time" in data

    def test_returns_settings_with_24h_enabled(self, client, mock_supabase):
        """Returns settings correctly when use_24h_time is True."""
        settings_24h = {**SETTINGS_RECORD, "use_24h_time": True, "timezone": "America/New_York"}
        mock_supabase.table.side_effect = lambda name: _builder([settings_24h])

        response = client.get("/settings")

        assert response.status_code == 200
        data = response.json()
        assert data["use_24h_time"] is True
        assert data["timezone"] == "America/New_York"


# ---------------------------------------------------------------------------
# PUT /settings
# ---------------------------------------------------------------------------


class TestUpdateSettings:
    """PUT /settings updates workspace settings for the authenticated user."""

    def test_updates_board_period(self, client, mock_supabase):
        """PUT /settings updates board_period and returns updated record."""
        updated = {**SETTINGS_RECORD, "board_period": "monthly"}
        mock_supabase.table.side_effect = lambda name: _builder([updated])

        response = client.put("/settings", json={"board_period": "monthly"})

        assert response.status_code == 200
        data = response.json()
        assert data["board_period"] == "monthly"

    def test_updates_calendar_view(self, client, mock_supabase):
        """PUT /settings updates calendar_view field."""
        updated = {**SETTINGS_RECORD, "calendar_view": "week"}
        mock_supabase.table.side_effect = lambda name: _builder([updated])

        response = client.put("/settings", json={"calendar_view": "week"})

        assert response.status_code == 200
        data = response.json()
        assert data["calendar_view"] == "week"

    def test_updates_timezone(self, client, mock_supabase):
        """PUT /settings updates timezone field."""
        updated = {**SETTINGS_RECORD, "timezone": "America/Los_Angeles"}
        mock_supabase.table.side_effect = lambda name: _builder([updated])

        response = client.put("/settings", json={"timezone": "America/Los_Angeles"})

        assert response.status_code == 200
        data = response.json()
        assert data["timezone"] == "America/Los_Angeles"

    def test_updates_use_24h_time(self, client, mock_supabase):
        """PUT /settings updates use_24h_time boolean field."""
        updated = {**SETTINGS_RECORD, "use_24h_time": True}
        mock_supabase.table.side_effect = lambda name: _builder([updated])

        response = client.put("/settings", json={"use_24h_time": True})

        assert response.status_code == 200
        data = response.json()
        assert data["use_24h_time"] is True

    def test_updates_custom_period_days(self, client, mock_supabase):
        """PUT /settings updates custom_period_days integer field."""
        updated = {**SETTINGS_RECORD, "board_period": "custom", "custom_period_days": 14}
        mock_supabase.table.side_effect = lambda name: _builder([updated])

        response = client.put(
            "/settings",
            json={"board_period": "custom", "custom_period_days": 14},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["custom_period_days"] == 14

    def test_updates_multiple_fields_at_once(self, client, mock_supabase):
        """PUT /settings can update multiple fields in a single request."""
        updated = {
            **SETTINGS_RECORD,
            "board_period": "daily",
            "calendar_view": "day",
            "use_24h_time": True,
        }
        mock_supabase.table.side_effect = lambda name: _builder([updated])

        response = client.put(
            "/settings",
            json={
                "board_period": "daily",
                "calendar_view": "day",
                "use_24h_time": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["board_period"] == "daily"
        assert data["calendar_view"] == "day"
        assert data["use_24h_time"] is True

    def test_upserts_when_no_row_exists(self, client, mock_supabase):
        """PUT /settings inserts a new row when update returns empty (no existing row)."""
        upserted = {**SETTINGS_RECORD, "board_period": "weekly"}

        call_idx = [0]

        def _side_effect(name):
            b = MockQueryBuilder()
            # First call: UPDATE returns empty (no row to update)
            # Second call: INSERT (upsert) returns the new record
            if call_idx[0] == 0:
                b.execute.return_value = MagicMock(data=[], count=0)
            else:
                b.execute.return_value = MagicMock(data=[upserted], count=1)
            call_idx[0] += 1
            return b

        mock_supabase.table.side_effect = _side_effect

        response = client.put("/settings", json={"board_period": "weekly"})

        assert response.status_code == 200
        data = response.json()
        assert data["board_period"] == "weekly"

    def test_empty_body_returns_current_settings(self, client, mock_supabase):
        """PUT /settings with all-None fields falls back to GET (returns current settings)."""
        # When all fields are None, service calls get_settings instead of updating
        mock_supabase.table.side_effect = lambda name: _builder([SETTINGS_RECORD])

        # Send an empty JSON body — all fields are optional and None by default
        response = client.put("/settings", json={})

        assert response.status_code == 200

    def test_returns_422_for_invalid_body_type(self, client, mock_supabase):
        """PUT /settings returns 422 when body fields are wrong type."""
        # use_24h_time must be a bool, not a string
        response = client.put("/settings", json={"use_24h_time": "not-a-bool"})

        assert response.status_code == 422
