"""Tests for project endpoints.

Covers:
- GET /projects              returns all projects for user's artists
- POST /projects             creates a project
- GET /projects/{artist_id}  returns projects for a specific artist
"""

from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder

ARTIST_ID = "artist-001"

PROJECT_RECORD = {
    "id": "project-001",
    "artist_id": ARTIST_ID,
    "name": "Test Album",
    "description": "A test album project",
    "created_at": "2025-01-01T00:00:00+00:00",
    "updated_at": "2025-01-01T00:00:00+00:00",
}


def _make_two_table_router(artists_data, projects_data):
    """Build a table side_effect that routes by table name.

    'artists' calls return artists_data; 'projects' calls return projects_data.
    This works for endpoints that call get_user_artist_ids (artists) then query
    projects, or verify_user_owns_artist (artists) then query projects.
    """

    def _router(name):
        builder = MockQueryBuilder()
        if name == "artists":
            builder.execute.return_value = MagicMock(
                data=artists_data,
                count=len(artists_data),
            )
        elif name == "projects":
            builder.execute.return_value = MagicMock(
                data=projects_data,
                count=len(projects_data),
            )
        return builder

    return _router


# ---------------------------------------------------------------------------
# GET /projects
# ---------------------------------------------------------------------------


class TestGetAllProjects:
    """GET /projects returns projects across all the user's artists."""

    def test_returns_list_for_user_with_artists(self, client, mock_supabase):
        """Returns projects when user has artists with projects."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[PROJECT_RECORD],
        )

        response = client.get("/projects")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == PROJECT_RECORD["id"]

    def test_returns_empty_list_when_no_artists(self, client, mock_supabase):
        """Returns empty list when user has no artists."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[],
            projects_data=[],
        )

        response = client.get("/projects")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_empty_list_when_no_projects(self, client, mock_supabase):
        """Returns empty list when user has artists but no projects."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[],
        )

        response = client.get("/projects")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_paginated_response_with_page_param(self, client, mock_supabase):
        """With ?page=1, returns PaginatedResponse envelope."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[PROJECT_RECORD],
        )

        response = client.get("/projects?page=1&page_size=25")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert data["page"] == 1
        assert data["page_size"] == 25
        assert isinstance(data["data"], list)

    def test_returns_paginated_empty_when_no_artists_with_page(self, client, mock_supabase):
        """With ?page=1 and no artists, returns paginated empty response."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[],
            projects_data=[],
        )

        response = client.get("/projects?page=1")

        assert response.status_code == 200
        data = response.json()
        # When no artists, endpoint returns PaginatedResponse with empty data
        assert "data" in data
        assert data["data"] == []
        assert data["total"] == 0

    def test_multiple_projects_returned(self, client, mock_supabase):
        """Returns multiple projects across all artists."""
        project_2 = {**PROJECT_RECORD, "id": "project-002", "name": "Second Album"}
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[PROJECT_RECORD, project_2],
        )

        response = client.get("/projects")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# POST /projects
# ---------------------------------------------------------------------------


class TestCreateProject:
    """POST /projects creates a project for an artist."""

    def test_creates_project_successfully(self, client, mock_supabase):
        """Returns the created project when artist is owned by user."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],  # verify_user_owns_artist
            projects_data=[PROJECT_RECORD],  # insert returns new record
        )

        payload = {
            "artist_id": ARTIST_ID,
            "name": "Test Album",
            "description": "A test album project",
        }
        response = client.post("/projects", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == PROJECT_RECORD["id"]
        assert data["name"] == PROJECT_RECORD["name"]
        assert data["artist_id"] == ARTIST_ID

    def test_creates_project_without_description(self, client, mock_supabase):
        """Description field is optional."""
        project_no_desc = {**PROJECT_RECORD, "description": None}
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[project_no_desc],
        )

        payload = {"artist_id": ARTIST_ID, "name": "Minimal Album"}
        response = client.post("/projects", json=payload)

        assert response.status_code == 200

    def test_returns_403_when_artist_not_owned(self, client, mock_supabase):
        """Returns 403 when user does not own the target artist."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[],  # verify_user_owns_artist returns False
            projects_data=[],
        )

        payload = {
            "artist_id": "artist-other",
            "name": "Unauthorized Album",
        }
        response = client.post("/projects", json=payload)

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    def test_returns_422_on_missing_required_fields(self, client, mock_supabase):
        """Returns 422 when required fields are missing."""
        response = client.post("/projects", json={"name": "No Artist"})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /projects/{artist_id}
# ---------------------------------------------------------------------------


class TestGetProjectsForArtist:
    """GET /projects/{artist_id} returns projects for a specific artist."""

    def test_returns_projects_for_owned_artist(self, client, mock_supabase):
        """Returns list of projects when user owns the artist."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[PROJECT_RECORD],
        )

        response = client.get(f"/projects/{ARTIST_ID}")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == PROJECT_RECORD["id"]
        assert data[0]["artist_id"] == ARTIST_ID

    def test_returns_empty_list_when_no_projects(self, client, mock_supabase):
        """Returns empty list when artist exists but has no projects."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[],
        )

        response = client.get(f"/projects/{ARTIST_ID}")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_403_when_artist_not_owned(self, client, mock_supabase):
        """Returns 403 when user does not own the artist."""
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[],  # verify_user_owns_artist returns False
            projects_data=[],
        )

        response = client.get("/projects/artist-other")

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    def test_returns_multiple_projects_for_artist(self, client, mock_supabase):
        """Returns all projects when artist has multiple."""
        project_2 = {**PROJECT_RECORD, "id": "project-002", "name": "EP Release"}
        mock_supabase.table.side_effect = _make_two_table_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[PROJECT_RECORD, project_2],
        )

        response = client.get(f"/projects/{ARTIST_ID}")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[1]["id"] == "project-002"
