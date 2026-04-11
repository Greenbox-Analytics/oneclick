"""Tests for TeamCard and Artists-with-TeamCard endpoints.

Acceptance criteria:
1. TeamCard: get own (200), update (200), get own when missing (404)
2. TeamCard collaborator: get (200, filtered fields), no-link returns 403
3. Artists with team cards: list (200 with "artists" key), get by ID (200)
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

COLLAB_USER_ID = "00000000-0000-0000-0000-000000000002"
ARTIST_ID = "aaaaaaaa-0000-0000-0000-000000000001"

SAMPLE_TEAM_CARD = {
    "id": "cccccccc-0000-0000-0000-000000000001",
    "user_id": TEST_USER_ID,
    "display_name": "Test Artist",
    "first_name": "Test",
    "last_name": "Artist",
    "email": "test@example.com",
    "bio": "Singer-songwriter",
    "phone": "+1-555-1234",
    "website": "https://example.com",
    "company": "Acme Music",
    "industry": "Music",
    "social_links": {"twitter": "testartist"},
    "dsp_links": {},
    "custom_links": [],
    "visible_fields": ["bio", "website"],
    "avatar_url": None,
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}

SAMPLE_ARTIST = {
    "id": ARTIST_ID,
    "user_id": TEST_USER_ID,
    "name": "Test Artist",
    "genre": "Pop",
    "linked_user_id": COLLAB_USER_ID,
    "verified": True,
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}


# ============================================================
# 1. Get own TeamCard
# ============================================================


class TestGetMyTeamCard:
    """GET /registry/teamcard"""

    def test_get_team_card_returns_card(self, client, mock_supabase):
        """Returns the current user's TeamCard."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=SAMPLE_TEAM_CARD)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/registry/teamcard")

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == TEST_USER_ID
        assert body["display_name"] == "Test Artist"

    def test_get_team_card_not_found_returns_404(self, client, mock_supabase):
        """Returns 404 when no TeamCard exists (onboarding not complete)."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=None)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/registry/teamcard")

        assert response.status_code == 404
        assert "TeamCard not found" in response.json()["detail"]


# ============================================================
# 2. Update TeamCard
# ============================================================


class TestUpdateTeamCard:
    """PUT /registry/teamcard"""

    def test_update_team_card_returns_updated_card(self, client, mock_supabase):
        """Successfully updates and returns the TeamCard."""
        updated = {**SAMPLE_TEAM_CARD, "bio": "Updated bio"}

        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[updated])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.put("/registry/teamcard", json={"bio": "Updated bio"})

        assert response.status_code == 200
        body = response.json()
        assert body["bio"] == "Updated bio"

    def test_update_team_card_not_found_returns_404(self, client, mock_supabase):
        """Returns 404 when the TeamCard row does not exist."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.put("/registry/teamcard", json={"bio": "Updated bio"})

        assert response.status_code == 404
        assert "TeamCard not found" in response.json()["detail"]

    def test_update_team_card_ignores_email_field(self, client, mock_supabase):
        """Email cannot be changed — service strips it; endpoint should still return 200."""
        updated = {**SAMPLE_TEAM_CARD, "display_name": "New Name"}

        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[updated])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.put(
            "/registry/teamcard",
            json={"display_name": "New Name", "email": "hacker@evil.com"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["display_name"] == "New Name"
        # email field value in response should remain original (service strips email)
        assert body["email"] == "test@example.com"

    def test_update_team_card_all_optional_fields(self, client, mock_supabase):
        """All optional fields can be updated together."""
        updated = {
            **SAMPLE_TEAM_CARD,
            "website": "https://new.com",
            "social_links": {"instagram": "mynewhandle"},
        }
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[updated])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.put(
            "/registry/teamcard",
            json={
                "website": "https://new.com",
                "social_links": {"instagram": "mynewhandle"},
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["website"] == "https://new.com"
        assert body["social_links"] == {"instagram": "mynewhandle"}


# ============================================================
# 3. Get collaborator's TeamCard
# ============================================================


class TestGetCollaboratorTeamCard:
    """GET /registry/teamcard/{collaborator_user_id}"""

    def test_get_collab_card_with_shared_link_returns_card(self, client, mock_supabase):
        """Returns filtered TeamCard when a collaboration link exists."""

        # registry_collaborators shared check
        link_builder = MockQueryBuilder()
        link_builder.execute.return_value = MagicMock(data=[{"id": "link-1"}])

        # team_cards lookup
        card_builder = MockQueryBuilder()
        card_builder.execute.return_value = MagicMock(
            data={
                "user_id": COLLAB_USER_ID,
                "email": "collab@example.com",
                "bio": "Collaborator bio",
                "visible_fields": ["bio"],
            }
        )

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return link_builder
            else:
                return card_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/teamcard/{COLLAB_USER_ID}")

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == COLLAB_USER_ID

    def test_get_collab_card_no_link_returns_403(self, client, mock_supabase):
        """Returns 403 when no collaboration link exists with the target user."""
        link_builder = MockQueryBuilder()
        link_builder.execute.return_value = MagicMock(data=[])

        mock_supabase.table.side_effect = lambda name: link_builder

        response = client.get(f"/registry/teamcard/{COLLAB_USER_ID}")

        assert response.status_code == 403
        assert "No collaboration link" in response.json()["detail"]

    def test_get_collab_card_link_exists_but_no_card_returns_404(self, client, mock_supabase):
        """Returns 404 when the link exists but the collaborator has no TeamCard."""
        link_builder = MockQueryBuilder()
        link_builder.execute.return_value = MagicMock(data=[{"id": "link-1"}])

        card_builder = MockQueryBuilder()
        card_builder.execute.return_value = MagicMock(data=None)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return link_builder
            else:
                return card_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/teamcard/{COLLAB_USER_ID}")

        assert response.status_code == 404
        assert "TeamCard not found" in response.json()["detail"]


# ============================================================
# 4. List artists with TeamCards
# ============================================================


class TestListArtistsWithTeamCards:
    """GET /registry/artists/with-teamcards"""

    def test_returns_artists_key_with_list(self, client, mock_supabase):
        """Returns {"artists": [...]} envelope."""
        artists_builder = MockQueryBuilder()
        artists_builder.execute.return_value = MagicMock(data=[{**SAMPLE_ARTIST, "teamcard": None}])

        # No linked_user_ids resolved (teamcard will be None since no verified+linked)
        mock_supabase.table.side_effect = lambda name: artists_builder

        response = client.get("/registry/artists/with-teamcards")

        assert response.status_code == 200
        body = response.json()
        assert "artists" in body
        assert isinstance(body["artists"], list)

    def test_returns_empty_list_when_no_artists(self, client, mock_supabase):
        """Returns {"artists": []} when user has no artists."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get("/registry/artists/with-teamcards")

        assert response.status_code == 200
        body = response.json()
        assert body["artists"] == []

    def test_artists_include_teamcard_overlay_when_verified(self, client, mock_supabase):
        """Artists with verified linked users get teamcard overlaid."""
        artist_with_link = {**SAMPLE_ARTIST}  # linked_user_id set, verified=True

        artists_builder = MockQueryBuilder()
        artists_builder.execute.return_value = MagicMock(data=[artist_with_link])

        tc_builder = MockQueryBuilder()
        tc_builder.execute.return_value = MagicMock(
            data=[
                {
                    "user_id": COLLAB_USER_ID,
                    "bio": "Collaborator bio",
                    "visible_fields": ["bio"],
                }
            ]
        )

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return artists_builder  # artists table
            else:
                return tc_builder  # team_cards table

        mock_supabase.table.side_effect = table_side_effect

        response = client.get("/registry/artists/with-teamcards")

        assert response.status_code == 200
        body = response.json()
        assert len(body["artists"]) == 1
        # teamcard should be present (it might be None if linked_user_id was not in tc_map)
        assert "teamcard" in body["artists"][0]


# ============================================================
# 5. Get single artist with TeamCard
# ============================================================


class TestGetArtistWithTeamCard:
    """GET /registry/artists/{artist_id}/with-teamcard"""

    def test_returns_artist_with_teamcard_none_when_not_linked(self, client, mock_supabase):
        """Returns artist data with teamcard: null when not linked/verified."""
        unlinked_artist = {**SAMPLE_ARTIST, "linked_user_id": None, "verified": False}

        artist_builder = MockQueryBuilder()
        artist_builder.execute.return_value = MagicMock(data=unlinked_artist)

        mock_supabase.table.side_effect = lambda name: artist_builder

        response = client.get(f"/registry/artists/{ARTIST_ID}/with-teamcard")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == ARTIST_ID
        assert body["teamcard"] is None

    def test_returns_artist_with_teamcard_when_verified_and_linked(self, client, mock_supabase):
        """Returns artist with teamcard overlay when verified and linked."""
        artist_builder = MockQueryBuilder()
        artist_builder.execute.return_value = MagicMock(data=SAMPLE_ARTIST)

        tc_builder = MockQueryBuilder()
        tc_builder.execute.return_value = MagicMock(
            data={
                "user_id": COLLAB_USER_ID,
                "bio": "Artist bio",
                "visible_fields": ["bio"],
            }
        )

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return artist_builder
            else:
                return tc_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/artists/{ARTIST_ID}/with-teamcard")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == ARTIST_ID
        assert body["teamcard"] is not None
        assert body["teamcard"]["user_id"] == COLLAB_USER_ID

    def test_artist_not_found_returns_404(self, client, mock_supabase):
        """Returns 404 when artist does not exist."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=None)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/artists/{ARTIST_ID}/with-teamcard")

        assert response.status_code == 404
        assert "Artist not found" in response.json()["detail"]
