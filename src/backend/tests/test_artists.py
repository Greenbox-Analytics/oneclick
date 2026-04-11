"""Tests for artist endpoints.

Covers:
- GET /artists             returns a list (unpaginated) or PaginatedResponse
- GET /artists/{artist_id} returns artist or 403/404
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

ARTIST_RECORD = {
    "id": "artist-001",
    "user_id": TEST_USER_ID,
    "name": "Test Artist",
    "created_at": "2025-01-01T00:00:00+00:00",
    "updated_at": "2025-01-01T00:00:00+00:00",
}


# ---------------------------------------------------------------------------
# GET /artists
# ---------------------------------------------------------------------------


class TestGetArtists:
    """GET /artists returns artist list for the authenticated user."""

    def test_returns_list_without_page_param(self, client, mock_supabase):
        """Without ?page, returns raw list (backward-compat)."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[ARTIST_RECORD], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/artists")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == ARTIST_RECORD["id"]

    def test_returns_empty_list_when_no_artists(self, client, mock_supabase):
        """Returns empty list when user has no artists."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/artists")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_paginated_response_with_page_param(self, client, mock_supabase):
        """With ?page=1, returns PaginatedResponse envelope."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[ARTIST_RECORD], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/artists?page=1&page_size=10")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert isinstance(data["data"], list)
        assert data["data"][0]["id"] == ARTIST_RECORD["id"]

    def test_multiple_artists_returned(self, client, mock_supabase):
        """Returns multiple artists when they exist."""
        artist_2 = {**ARTIST_RECORD, "id": "artist-002", "name": "Second Artist"}
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[ARTIST_RECORD, artist_2], count=2)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/artists")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# GET /artists/{artist_id}
# ---------------------------------------------------------------------------


class TestGetArtistById:
    """GET /artists/{artist_id} returns a single artist or 403."""

    def _make_table_router(self, ownership_data, artist_data):
        """Return a side_effect that routes 'artists' calls by call order.

        First call: verify_user_owns_artist (ownership check)
        Second call: fetch the actual artist record
        """
        call_count = {"n": 0}

        def _router(name):
            builder = MockQueryBuilder()
            if name == "artists":
                if call_count["n"] == 0:
                    # verify_user_owns_artist
                    builder.execute.return_value = MagicMock(data=ownership_data, count=len(ownership_data))
                else:
                    # fetch single record
                    builder.execute.return_value = MagicMock(data=artist_data, count=1 if artist_data else 0)
                call_count["n"] += 1
            return builder

        return _router

    def test_returns_artist_when_owned(self, client, mock_supabase):
        """Returns artist data when user owns it."""
        mock_supabase.table.side_effect = self._make_table_router(
            ownership_data=[{"id": ARTIST_RECORD["id"]}],
            artist_data=ARTIST_RECORD,
        )

        response = client.get(f"/artists/{ARTIST_RECORD['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == ARTIST_RECORD["id"]
        assert data["name"] == ARTIST_RECORD["name"]

    def test_returns_403_when_not_owned(self, client, mock_supabase):
        """Returns 403 when the artist does not belong to the user."""
        mock_supabase.table.side_effect = self._make_table_router(
            ownership_data=[],  # verify_user_owns_artist returns False
            artist_data=None,
        )

        response = client.get("/artists/artist-other")

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"
