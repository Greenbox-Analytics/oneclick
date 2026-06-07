"""Tests for project endpoints.

Covers:
- GET /projects              returns all projects for user's artists
- POST /projects             creates a project
- GET /projects/{artist_id}  returns projects for a specific artist
"""

from unittest.mock import MagicMock

from tests.conftest import _DEFAULT_USAGE_ROW, _PRO_SUB_ROW, _PRO_TIER_ROW, TEST_USER_ID, MockQueryBuilder

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
    Subscription tables are stubbed with Pro-unlimited rows so the subscription
    gate always passes in tests that aren't testing the gate itself.
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
        elif name == "subscriptions":
            builder.execute.return_value = MagicMock(data=[_PRO_SUB_ROW], count=1)
        elif name == "tier_entitlements":
            builder.execute.return_value = MagicMock(data=[_PRO_TIER_ROW], count=1)
        elif name == "tier_overrides":
            builder.execute.return_value = MagicMock(data=[], count=0)
        elif name == "usage_counters":
            builder.execute.return_value = MagicMock(data=[_DEFAULT_USAGE_ROW], count=1)
        return builder

    return _router


def _make_create_project_router(
    *,
    artists_data,
    project_count=0,
    duplicate_data=None,
    inserted_project=PROJECT_RECORD,
    member_insert_error=False,
):
    """Router for POST /projects.

    `projects` table is queried three times, in order: gating count, duplicate-name
    check, then insert. (A 4th `projects` call — a delete — happens only on the
    owner-insert rollback path.) `project_members` is inserted once for the owner row.
    Captures all builders on `_router.builders` so tests can assert insert/delete calls.
    """
    duplicate_data = duplicate_data or []
    projects_calls = {"n": 0}
    builders = {}

    def _router(name):
        builder = MockQueryBuilder()
        builders.setdefault(name, []).append(builder)
        if name == "artists":
            builder.execute.return_value = MagicMock(data=artists_data, count=len(artists_data))
        elif name == "projects":
            projects_calls["n"] += 1
            if projects_calls["n"] == 1:  # gating count
                builder.execute.return_value = MagicMock(data=[], count=project_count)
            elif projects_calls["n"] == 2:  # duplicate-name check — returns the artist's project name rows
                builder.execute.return_value = MagicMock(data=duplicate_data, count=len(duplicate_data))
            else:  # insert (3) — rollback delete (4) uses builder.delete, not execute
                builder.execute.return_value = MagicMock(data=[inserted_project], count=1)
        elif name == "project_members":
            if member_insert_error:
                builder.execute.side_effect = RuntimeError("member insert failed")
            else:
                builder.execute.return_value = MagicMock(
                    data=[{"project_id": inserted_project["id"], "user_id": TEST_USER_ID, "role": "owner"}],
                    count=1,
                )
        elif name == "subscriptions":
            builder.execute.return_value = MagicMock(data=[_PRO_SUB_ROW], count=1)
        elif name == "tier_entitlements":
            builder.execute.return_value = MagicMock(data=[_PRO_TIER_ROW], count=1)
        elif name == "tier_overrides":
            builder.execute.return_value = MagicMock(data=[], count=0)
        elif name == "usage_counters":
            builder.execute.return_value = MagicMock(data=[_DEFAULT_USAGE_ROW], count=1)
        return builder

    _router.builders = builders
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
        mock_supabase.table.side_effect = _make_create_project_router(
            artists_data=[{"id": ARTIST_ID}],
            inserted_project=PROJECT_RECORD,
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
        mock_supabase.table.side_effect = _make_create_project_router(
            artists_data=[{"id": ARTIST_ID}],
            inserted_project=project_no_desc,
        )

        payload = {"artist_id": ARTIST_ID, "name": "Minimal Album"}
        response = client.post("/projects", json=payload)

        assert response.status_code == 200

    def test_creates_owner_membership_row(self, client, mock_supabase):
        """On success the creator is inserted into project_members as 'owner'."""
        router = _make_create_project_router(artists_data=[{"id": ARTIST_ID}])
        mock_supabase.table.side_effect = router

        response = client.post("/projects", json={"artist_id": ARTIST_ID, "name": "Owner Test"})

        assert response.status_code == 200
        member_builders = router.builders.get("project_members", [])
        assert len(member_builders) == 1, "owner row must be inserted exactly once"
        member_builders[0].insert.assert_called_once()
        payload = member_builders[0].insert.call_args.args[0]
        assert payload["role"] == "owner"
        assert payload["user_id"] == TEST_USER_ID
        assert payload["project_id"] == PROJECT_RECORD["id"]

    def test_duplicate_name_returns_409(self, client, mock_supabase):
        """A case-insensitive duplicate name for the same artist returns 409 and does not insert."""
        # Existing project stored as "test album"; the new request uses "Test Album" —
        # the normalized (lower/stripped) comparison must still flag it as a duplicate.
        router = _make_create_project_router(
            artists_data=[{"id": ARTIST_ID}],
            duplicate_data=[{"name": "test album"}],
        )
        mock_supabase.table.side_effect = router

        response = client.post("/projects", json={"artist_id": ARTIST_ID, "name": "Test Album"})

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()
        # Insert must NOT have run: only the count + duplicate-check projects calls happened.
        assert len(router.builders.get("projects", [])) == 2
        assert "project_members" not in router.builders

    def test_owner_insert_failure_rolls_back_project(self, client, mock_supabase):
        """If the owner-membership insert fails, the project is deleted and 500 returned."""
        router = _make_create_project_router(
            artists_data=[{"id": ARTIST_ID}],
            member_insert_error=True,
        )
        mock_supabase.table.side_effect = router

        response = client.post("/projects", json={"artist_id": ARTIST_ID, "name": "Rollback Album"})

        assert response.status_code == 500
        # The orphaned project must be rolled back: a 4th projects call issues delete().
        projects_builders = router.builders.get("projects", [])
        assert len(projects_builders) == 4, "rollback delete must run on the projects table"
        projects_builders[-1].delete.assert_called_once()

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


# ---------------------------------------------------------------------------
# Subscription gate tests
# ---------------------------------------------------------------------------


class TestProjectCreateGated:
    """POST /projects with a Free user at cap returns 402."""

    def test_create_project_at_cap_full_count_path(self, client, mock_supabase):
        """Full count-path test: real EntitlementsService runs over wired supabase tables.

        Catches count-query regressions (wrong table/filter/etc.) that the monkeypatch
        test would miss. Free tier max_projects=3; user has 3 projects → 402.
        """

        FREE_TIER = {
            "tier": "free",
            "max_artists": 3,
            "max_projects": 3,
            "max_boards": -1,
            "max_tasks": 50,
            "max_storage_bytes": 1073741824,
            "max_split_sheets_per_month": 5,
            "max_oneclick_runs_per_month": 1,
            "zoe_enabled": False,
            "oneclick_enabled": True,
            "registry_enabled": False,
            "integrations_allowed": ["google_drive"],
            "updated_at": "2026-05-09T00:00:00+00:00",
        }
        FREE_SUB = {
            "id": "s1",
            "user_id": TEST_USER_ID,
            "tier": "free",
            "status": "active",
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "stripe_price_id": None,
            "current_period_start": None,
            "current_period_end": None,
            "cancel_at_period_end": False,
            "canceled_at": None,
            "created_at": "2026-05-01T00:00:00+00:00",
            "updated_at": "2026-05-01T00:00:00+00:00",
        }
        ZERO_USAGE = {
            "user_id": TEST_USER_ID,
            "total_storage_bytes": 0,
            "split_sheets_this_period": 0,
            "zoe_queries_this_period": 0,
            "oneclick_runs_this_period": 0,
            "period_start": "2026-05-09T00:00:00+00:00",
            "period_end": "2099-05-09T00:00:00+00:00",
            "updated_at": "2026-05-09T00:00:00+00:00",
        }

        # Reset the EntitlementsService singleton so it is rebuilt with free-tier data
        import subscriptions.deps as _sub_deps

        _sub_deps._entitlements_service = None

        from unittest.mock import MagicMock

        def _table(name):
            b = MockQueryBuilder()
            if name == "artists":
                # verify_user_owns_artist AND get_user_artist_ids both query artists
                b.execute.return_value = MagicMock(
                    data=[
                        {"id": "a1", "user_id": TEST_USER_ID},
                        {"id": "a2", "user_id": TEST_USER_ID},
                        {"id": "a3", "user_id": TEST_USER_ID},
                    ],
                    count=3,
                )
            elif name == "projects":
                # 3 projects at cap of 3
                b.execute.return_value = MagicMock(
                    data=[{"id": "p1"}, {"id": "p2"}, {"id": "p3"}],
                    count=3,
                )
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[FREE_SUB], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "usage_counters":
                b.execute.return_value = MagicMock(data=[ZERO_USAGE], count=1)
            return b

        mock_supabase.table.side_effect = _table

        resp = client.post(
            "/projects",
            json={
                "artist_id": "a1",
                "name": "Project 4",
                "description": "",
            },
        )
        assert resp.status_code == 402
        assert "project" in resp.json()["detail"].lower()

    def test_create_project_at_cap_returns_402(self, client, mock_supabase, monkeypatch):
        """Free user at 3/3 projects → POST /projects returns 402."""
        from unittest.mock import MagicMock

        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        # Patch enforcement._service to return a service that denies CREATE_PROJECT
        deny_result = CheckResult(
            allowed=False,
            reason="You've reached your limit of 3 projects.",
            upgrade_required=True,
        )
        svc = MagicMock()
        svc.can.return_value = deny_result
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        # artists table: user owns artist a1 (so ownership check passes)
        # projects table: returns 3 projects (count=3 at cap — used by the gate query)
        def _table(name):
            b = MockQueryBuilder()
            if name == "artists":
                b.execute.return_value = MagicMock(data=[{"id": "a1"}], count=1)
            elif name == "projects":
                b.execute.return_value = MagicMock(data=[{"id": "p1"}, {"id": "p2"}, {"id": "p3"}], count=3)
            return b

        mock_supabase.table.side_effect = _table

        resp = client.post(
            "/projects",
            json={"artist_id": "a1", "name": "Project 4", "description": ""},
        )
        assert resp.status_code == 402
        assert "project" in resp.json()["detail"].lower()
