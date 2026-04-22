"""Tests for project about, agreements, and work file/audio link endpoints.

Acceptance criteria:
1. Project about: get (200), update (200), get not found (404), get no access (403)
2. Agreements: list (200 with "agreements" key), create (200)
3. Work file links: list (200 with "files" key), link (200 with "link" key), unlink (200)
4. Work audio links: list (200 with "audio" key), link (200 with "link" key), unlink (200)
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

WORK_ID = "aaaaaaaa-0000-0000-0000-000000000001"
PROJECT_ID = "bbbbbbbb-0000-0000-0000-000000000001"
ARTIST_ID = "cccccccc-0000-0000-0000-000000000001"
FILE_ID = "dddddddd-0000-0000-0000-000000000001"
AUDIO_FILE_ID = "eeeeeeee-0000-0000-0000-000000000001"
LINK_ID = "ffffffff-0000-0000-0000-000000000001"
AGREEMENT_ID = "11111111-0000-0000-0000-000000000001"

SAMPLE_AGREEMENT = {
    "id": AGREEMENT_ID,
    "work_id": WORK_ID,
    "user_id": TEST_USER_ID,
    "agreement_type": "co-publishing",
    "title": "Co-Publishing Agreement",
    "description": "Standard co-pub deal",
    "effective_date": "2026-01-01",
    "parties": [{"name": "Alice", "role": "publisher"}],
    "file_id": None,
    "document_hash": None,
    "created_at": "2026-01-01T00:00:00+00:00",
}

SAMPLE_FILE_LINK = {
    "id": LINK_ID,
    "work_id": WORK_ID,
    "file_id": FILE_ID,
    "created_at": "2026-01-01T00:00:00+00:00",
}

SAMPLE_AUDIO_LINK = {
    "id": LINK_ID,
    "work_id": WORK_ID,
    "audio_file_id": AUDIO_FILE_ID,
    "created_at": "2026-01-01T00:00:00+00:00",
}


# ============================================================
# Helper: build project-about access scenario
# ============================================================
# The GET /registry/projects/{id}/about endpoint performs:
#   1. projects table -> get artist_id
#   2. artists table  -> get user_id / linked_user_id (ownership check)
#   3. works_registry -> get work IDs for collab access check
#   4. registry_collaborators -> check if user is a collab
# The PUT calls service which does:
#   1. projects -> get artist_id
#   2. artists  -> verify ownership
#   3. projects -> update


def _make_project_about_builders(
    artist_user_id=TEST_USER_ID,
    linked_user_id=None,
    collab_work_ids=None,
    has_collab_access=False,
    about_content=None,
):
    """Build ordered MockQueryBuilders for GET project about access path."""
    project_builder = MockQueryBuilder()
    project_builder.execute.return_value = MagicMock(data={"artist_id": ARTIST_ID}, count=1)

    artist_builder = MockQueryBuilder()
    artist_builder.execute.return_value = MagicMock(
        data={"user_id": artist_user_id, "linked_user_id": linked_user_id}, count=1
    )

    works_builder = MockQueryBuilder()
    works_builder.execute.return_value = MagicMock(data=collab_work_ids or [], count=len(collab_work_ids or []))

    collab_check_builder = MockQueryBuilder()
    collab_check_builder.execute.return_value = MagicMock(
        data=[{"id": "collab-1"}] if has_collab_access else [], count=1 if has_collab_access else 0
    )

    about_builder = MockQueryBuilder()
    about_builder.execute.return_value = MagicMock(data={"about_content": about_content or []}, count=1)

    return [project_builder, artist_builder, works_builder, collab_check_builder, about_builder]


# ============================================================
# 1. Project About: GET
# ============================================================


class TestGetProjectAbout:
    """GET /registry/projects/{project_id}/about"""

    def test_owner_can_get_project_about(self, client, mock_supabase):
        """Project owner gets about content."""
        builders = _make_project_about_builders(
            artist_user_id=TEST_USER_ID,
            about_content=[{"type": "paragraph", "text": "About this project"}],
        )
        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            return builders[min(n - 1, len(builders) - 1)]

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/projects/{PROJECT_ID}/about")

        assert response.status_code == 200
        body = response.json()
        assert "about_content" in body

    def test_get_project_not_found_returns_404(self, client, mock_supabase):
        """Returns 404 when project does not exist."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=None, count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/projects/{PROJECT_ID}/about")

        assert response.status_code == 404
        assert "Project not found" in response.json()["detail"]

    def test_non_owner_non_collab_gets_403(self, client, mock_supabase):
        """User unrelated to the project gets 403."""
        project_builder = MockQueryBuilder()
        project_builder.execute.return_value = MagicMock(data={"artist_id": ARTIST_ID}, count=1)

        artist_builder = MockQueryBuilder()
        # Different owner
        artist_builder.execute.return_value = MagicMock(
            data={"user_id": "other-user-id", "linked_user_id": None}, count=1
        )

        works_builder = MockQueryBuilder()
        works_builder.execute.return_value = MagicMock(data=[], count=0)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return project_builder
            elif n == 2:
                return artist_builder
            else:
                return works_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/projects/{PROJECT_ID}/about")

        assert response.status_code == 403
        assert "Not authorized" in response.json()["detail"]

    def test_collaborator_can_get_project_about(self, client, mock_supabase):
        """A collaborator on a work in the project can read about content."""
        builders = _make_project_about_builders(
            artist_user_id="other-owner",
            collab_work_ids=[{"id": WORK_ID}],
            has_collab_access=True,
            about_content=[],
        )
        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            return builders[min(n - 1, len(builders) - 1)]

        mock_supabase.table.side_effect = table_side_effect

        response = client.get(f"/registry/projects/{PROJECT_ID}/about")

        assert response.status_code == 200
        body = response.json()
        assert "about_content" in body


# ============================================================
# 2. Project About: PUT
# ============================================================


class TestUpdateProjectAbout:
    """PUT /registry/projects/{project_id}/about"""

    def test_owner_can_update_project_about(self, client, mock_supabase):
        """Project owner can update about content."""
        # service calls: projects (get artist_id), artists (verify owner), projects (update)
        project_get_builder = MockQueryBuilder()
        project_get_builder.execute.return_value = MagicMock(data={"id": PROJECT_ID, "artist_id": ARTIST_ID}, count=1)

        artist_builder = MockQueryBuilder()
        artist_builder.execute.return_value = MagicMock(data={"user_id": TEST_USER_ID}, count=1)

        project_update_builder = MockQueryBuilder()
        project_update_builder.execute.return_value = MagicMock(data=[{"id": PROJECT_ID, "about_content": []}], count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return project_get_builder
            elif n == 2:
                return artist_builder
            else:
                return project_update_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.put(
            f"/registry/projects/{PROJECT_ID}/about",
            json={"about_content": [{"type": "paragraph", "text": "Hello"}]},
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_update_project_not_found_returns_404(self, client, mock_supabase):
        """Returns 404 when project does not exist (service returns None)."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=None, count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.put(
            f"/registry/projects/{PROJECT_ID}/about",
            json={"about_content": []},
        )

        assert response.status_code == 404
        assert "Project not found" in response.json()["detail"]

    def test_update_sends_empty_content(self, client, mock_supabase):
        """Accepts empty about_content list."""
        project_get_builder = MockQueryBuilder()
        project_get_builder.execute.return_value = MagicMock(data={"id": PROJECT_ID, "artist_id": ARTIST_ID}, count=1)

        artist_builder = MockQueryBuilder()
        artist_builder.execute.return_value = MagicMock(data={"user_id": TEST_USER_ID}, count=1)

        project_update_builder = MockQueryBuilder()
        project_update_builder.execute.return_value = MagicMock(data=[{"id": PROJECT_ID, "about_content": []}], count=1)

        call_count = {"n": 0}

        def table_side_effect(name):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return project_get_builder
            elif n == 2:
                return artist_builder
            else:
                return project_update_builder

        mock_supabase.table.side_effect = table_side_effect

        response = client.put(
            f"/registry/projects/{PROJECT_ID}/about",
            json={"about_content": []},
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}


# ============================================================
# 3. Agreements
# ============================================================


class TestListAgreements:
    """GET /registry/agreements?work_id=..."""

    def test_list_agreements_returns_agreements_key(self, client, mock_supabase):
        """Returns {"agreements": [...]} envelope."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_AGREEMENT], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/agreements?work_id={WORK_ID}")

        assert response.status_code == 200
        body = response.json()
        assert "agreements" in body
        assert isinstance(body["agreements"], list)

    def test_list_agreements_empty(self, client, mock_supabase):
        """Returns empty list when no agreements exist for the work."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/agreements?work_id={WORK_ID}")

        assert response.status_code == 200
        assert response.json()["agreements"] == []

    def test_list_agreements_with_data(self, client, mock_supabase):
        """Returns agreements list with correct data."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_AGREEMENT], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/agreements?work_id={WORK_ID}")

        assert response.status_code == 200
        body = response.json()
        assert len(body["agreements"]) == 1
        assert body["agreements"][0]["id"] == AGREEMENT_ID
        assert body["agreements"][0]["title"] == "Co-Publishing Agreement"

    def test_list_agreements_requires_work_id(self, client, mock_supabase):
        """Returns 422 when work_id query param is missing."""
        response = client.get("/registry/agreements")

        assert response.status_code == 422


class TestCreateAgreement:
    """POST /registry/agreements"""

    def test_create_agreement_success(self, client, mock_supabase):
        """Creates and returns new agreement."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_AGREEMENT], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        payload = {
            "work_id": WORK_ID,
            "agreement_type": "co-publishing",
            "title": "Co-Publishing Agreement",
            "effective_date": "2026-01-01",
            "parties": [{"name": "Alice", "role": "publisher"}],
        }
        response = client.post("/registry/agreements", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == AGREEMENT_ID
        assert body["title"] == "Co-Publishing Agreement"

    def test_create_agreement_failure_returns_500(self, client, mock_supabase):
        """Returns 500 when insert fails."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        payload = {
            "work_id": WORK_ID,
            "agreement_type": "distribution",
            "title": "Distribution Deal",
            "effective_date": "2026-01-01",
            "parties": [{"name": "Bob", "role": "distributor"}],
        }
        response = client.post("/registry/agreements", json=payload)

        assert response.status_code == 500
        assert "Failed to create agreement" in response.json()["detail"]

    def test_create_agreement_missing_required_fields_returns_422(self, client, mock_supabase):
        """Returns 422 when required fields are missing."""
        response = client.post("/registry/agreements", json={"work_id": WORK_ID})

        assert response.status_code == 422

    def test_create_agreement_with_optional_fields(self, client, mock_supabase):
        """Creates agreement with optional description and file_id."""
        enriched = {**SAMPLE_AGREEMENT, "description": "Detailed terms", "file_id": FILE_ID}

        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[enriched], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        payload = {
            "work_id": WORK_ID,
            "agreement_type": "licensing",
            "title": "Sync License",
            "effective_date": "2026-06-01",
            "parties": [{"name": "Studio", "role": "licensee"}],
            "description": "Detailed terms",
            "file_id": FILE_ID,
        }
        response = client.post("/registry/agreements", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["description"] == "Detailed terms"


# ============================================================
# 4. Work File Links
# ============================================================


class TestWorkFileLinks:
    """GET/POST/DELETE /registry/works/{work_id}/files"""

    def test_list_work_files_returns_files_key(self, client, mock_supabase):
        """Returns {"files": [...]} envelope."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_FILE_LINK], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/works/{WORK_ID}/files")

        assert response.status_code == 200
        body = response.json()
        assert "files" in body
        assert isinstance(body["files"], list)

    def test_list_work_files_empty(self, client, mock_supabase):
        """Returns empty list when no files linked."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/works/{WORK_ID}/files")

        assert response.status_code == 200
        assert response.json()["files"] == []

    def test_list_work_files_with_data(self, client, mock_supabase):
        """Returns file link records."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_FILE_LINK], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/works/{WORK_ID}/files")

        assert response.status_code == 200
        body = response.json()
        assert len(body["files"]) == 1
        assert body["files"][0]["id"] == LINK_ID

    def test_link_file_to_work_returns_link(self, client, mock_supabase):
        """POST creates a work-file link and returns {"link": {...}}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_FILE_LINK], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.post(f"/registry/works/{WORK_ID}/files?file_id={FILE_ID}")

        assert response.status_code == 200
        body = response.json()
        assert "link" in body

    def test_link_file_missing_file_id_returns_422(self, client, mock_supabase):
        """Returns 422 when file_id query param is missing."""
        response = client.post(f"/registry/works/{WORK_ID}/files")

        assert response.status_code == 422

    def test_unlink_file_from_work_returns_deleted(self, client, mock_supabase):
        """DELETE removes link and returns {"deleted": link_id}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.delete(f"/registry/works/{WORK_ID}/files/{LINK_ID}")

        assert response.status_code == 200
        body = response.json()
        assert body == {"deleted": LINK_ID}

    def test_link_file_returns_none_when_insert_empty(self, client, mock_supabase):
        """POST returns {"link": null} when insert returns no data."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.post(f"/registry/works/{WORK_ID}/files?file_id={FILE_ID}")

        assert response.status_code == 200
        body = response.json()
        assert "link" in body
        assert body["link"] is None


# ============================================================
# 5. Work Audio Links
# ============================================================


class TestWorkAudioLinks:
    """GET/POST/DELETE /registry/works/{work_id}/audio"""

    def test_list_work_audio_returns_audio_key(self, client, mock_supabase):
        """Returns {"audio": [...]} envelope."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_AUDIO_LINK], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/works/{WORK_ID}/audio")

        assert response.status_code == 200
        body = response.json()
        assert "audio" in body
        assert isinstance(body["audio"], list)

    def test_list_work_audio_empty(self, client, mock_supabase):
        """Returns empty list when no audio linked."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/works/{WORK_ID}/audio")

        assert response.status_code == 200
        assert response.json()["audio"] == []

    def test_list_work_audio_with_data(self, client, mock_supabase):
        """Returns audio link records."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_AUDIO_LINK], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.get(f"/registry/works/{WORK_ID}/audio")

        assert response.status_code == 200
        body = response.json()
        assert len(body["audio"]) == 1
        assert body["audio"][0]["id"] == LINK_ID

    def test_link_audio_to_work_returns_link(self, client, mock_supabase):
        """POST creates a work-audio link and returns {"link": {...}}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[SAMPLE_AUDIO_LINK], count=1)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.post(f"/registry/works/{WORK_ID}/audio?audio_file_id={AUDIO_FILE_ID}")

        assert response.status_code == 200
        body = response.json()
        assert "link" in body

    def test_link_audio_missing_audio_file_id_returns_422(self, client, mock_supabase):
        """Returns 422 when audio_file_id query param is missing."""
        response = client.post(f"/registry/works/{WORK_ID}/audio")

        assert response.status_code == 422

    def test_unlink_audio_from_work_returns_deleted(self, client, mock_supabase):
        """DELETE removes link and returns {"deleted": link_id}."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.delete(f"/registry/works/{WORK_ID}/audio/{LINK_ID}")

        assert response.status_code == 200
        body = response.json()
        assert body == {"deleted": LINK_ID}

    def test_link_audio_returns_none_when_insert_empty(self, client, mock_supabase):
        """POST returns {"link": null} when insert returns no data."""
        builder = MockQueryBuilder()
        builder.execute.return_value = MagicMock(data=[], count=0)
        mock_supabase.table.side_effect = lambda name: builder

        response = client.post(f"/registry/works/{WORK_ID}/audio?audio_file_id={AUDIO_FILE_ID}")

        assert response.status_code == 200
        body = response.json()
        assert "link" in body
        assert body["link"] is None
