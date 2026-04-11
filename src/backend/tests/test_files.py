"""Tests for file listing, contract listing, and contract deletion endpoints.

Covers:
- GET /files/{project_id}                            file listing by project
- GET /files/artist/{artist_id}/category/{category}  file listing by artist + category
- GET /projects/{project_id}/contracts               contract listing for a project
- GET /projects/{project_id}/documents               document listing for a project
- DELETE /contracts/{contract_id}                    contract deletion
"""

from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder

ARTIST_ID = "artist-001"
PROJECT_ID = "project-001"
CONTRACT_ID = "contract-001"

FILE_RECORD = {
    "id": "file-001",
    "project_id": PROJECT_ID,
    "folder_category": "contract",
    "file_name": "deal.pdf",
    "file_url": "https://example.com/deal.pdf",
    "file_path": f"{ARTIST_ID}/{PROJECT_ID}/contract/deal.pdf",
    "file_size": 12345,
    "file_type": "application/pdf",
    "created_at": "2026-01-01T00:00:00+00:00",
}

CONTRACT_RECORD = {
    **FILE_RECORD,
    "id": CONTRACT_ID,
    "folder_category": "contract",
}


# ---------------------------------------------------------------------------
# Table-routing helpers
# ---------------------------------------------------------------------------


def _make_project_ownership_router(
    *,
    artists_data=None,
    projects_data=None,
    project_files_data=None,
):
    """Return a table side_effect for endpoints that call verify_user_owns_project.

    Call sequence for verify_user_owns_project:
      1. artists  (get_user_artist_ids)
      2. projects (check project belongs to those artists)
    Followed by:
      3. project_files (the actual query)

    Each table name returns a fresh builder pre-configured with the supplied data.
    """
    if artists_data is None:
        artists_data = [{"id": ARTIST_ID}]
    if projects_data is None:
        projects_data = [{"id": PROJECT_ID}]
    if project_files_data is None:
        project_files_data = []

    def _router(name):
        builder = MockQueryBuilder()
        if name == "artists":
            builder.execute.return_value = MagicMock(data=artists_data, count=len(artists_data))
        elif name == "projects":
            builder.execute.return_value = MagicMock(data=projects_data, count=len(projects_data))
        elif name == "project_files":
            builder.execute.return_value = MagicMock(data=project_files_data, count=len(project_files_data))
        return builder

    return _router


def _make_artist_ownership_router(
    *,
    artists_data=None,
    projects_data=None,
    project_files_data=None,
):
    """Return a table side_effect for endpoints that call verify_user_owns_artist.

    Call sequence:
      1. artists       (verify_user_owns_artist)
      2. projects      (get project IDs for the artist)
      3. project_files (filtered files)
    """
    if artists_data is None:
        artists_data = [{"id": ARTIST_ID}]
    if projects_data is None:
        projects_data = [{"id": PROJECT_ID}]
    if project_files_data is None:
        project_files_data = []

    def _router(name):
        builder = MockQueryBuilder()
        if name == "artists":
            builder.execute.return_value = MagicMock(data=artists_data, count=len(artists_data))
        elif name == "projects":
            builder.execute.return_value = MagicMock(data=projects_data, count=len(projects_data))
        elif name == "project_files":
            builder.execute.return_value = MagicMock(data=project_files_data, count=len(project_files_data))
        return builder

    return _router


# ---------------------------------------------------------------------------
# GET /files/{project_id}
# ---------------------------------------------------------------------------


class TestGetProjectFiles:
    """GET /files/{project_id} returns files for an owned project."""

    def test_returns_files_for_owned_project(self, client, mock_supabase):
        """Returns a list of files when user owns the project."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[FILE_RECORD],
        )

        response = client.get(f"/files/{PROJECT_ID}")

        assert response.status_code == 200
        data = response.json()
        # Without pagination param, returns raw list or paginated — check both shapes
        records = data["data"] if isinstance(data, dict) and "data" in data else data
        assert isinstance(records, list)
        assert len(records) == 1
        assert records[0]["id"] == FILE_RECORD["id"]

    def test_returns_empty_list_when_no_files(self, client, mock_supabase):
        """Returns empty list when project has no files."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[],
        )

        response = client.get(f"/files/{PROJECT_ID}")

        assert response.status_code == 200
        data = response.json()
        records = data["data"] if isinstance(data, dict) and "data" in data else data
        assert records == []

    def test_returns_403_when_project_not_owned(self, client, mock_supabase):
        """Returns 403 when user does not own the project."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            artists_data=[],  # get_user_artist_ids returns empty -> ownership fails
            projects_data=[],
        )

        response = client.get(f"/files/{PROJECT_ID}")

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    def test_returns_403_when_project_belongs_to_other_artist(self, client, mock_supabase):
        """Returns 403 when the project exists but belongs to another user's artist."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            artists_data=[{"id": ARTIST_ID}],
            projects_data=[],  # project not in user's artists -> ownership fails
        )

        response = client.get(f"/files/{PROJECT_ID}")

        assert response.status_code == 403

    def test_paginated_response_with_page_param(self, client, mock_supabase):
        """With ?page=1, returns a PaginatedResponse envelope."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[FILE_RECORD],
        )

        response = client.get(f"/files/{PROJECT_ID}?page=1&page_size=25")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert data["page"] == 1
        assert data["page_size"] == 25

    def test_multiple_files_returned(self, client, mock_supabase):
        """Returns all files when project has multiple."""
        file_2 = {**FILE_RECORD, "id": "file-002", "file_name": "royalty.pdf"}
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[FILE_RECORD, file_2],
        )

        response = client.get(f"/files/{PROJECT_ID}")

        assert response.status_code == 200
        data = response.json()
        records = data["data"] if isinstance(data, dict) and "data" in data else data
        assert len(records) == 2


# ---------------------------------------------------------------------------
# GET /files/artist/{artist_id}/category/{category}
# ---------------------------------------------------------------------------


class TestGetArtistFilesByCategory:
    """GET /files/artist/{artist_id}/category/{category} filters by category."""

    def test_returns_files_for_owned_artist_and_category(self, client, mock_supabase):
        """Returns files when user owns the artist and category matches."""
        mock_supabase.table.side_effect = _make_artist_ownership_router(
            project_files_data=[FILE_RECORD],
        )

        response = client.get(f"/files/artist/{ARTIST_ID}/category/contract")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["folder_category"] == "contract"

    def test_returns_empty_list_when_no_files_in_category(self, client, mock_supabase):
        """Returns empty list when no files match the category."""
        mock_supabase.table.side_effect = _make_artist_ownership_router(
            project_files_data=[],
        )

        response = client.get(f"/files/artist/{ARTIST_ID}/category/royalty_statement")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_empty_list_when_artist_has_no_projects(self, client, mock_supabase):
        """Returns empty list when artist exists but has no projects."""
        mock_supabase.table.side_effect = _make_artist_ownership_router(
            projects_data=[],
        )

        response = client.get(f"/files/artist/{ARTIST_ID}/category/contract")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_403_when_artist_not_owned(self, client, mock_supabase):
        """Returns 403 when user does not own the artist."""
        mock_supabase.table.side_effect = _make_artist_ownership_router(
            artists_data=[],  # verify_user_owns_artist returns False
        )

        response = client.get("/files/artist/other-artist/category/contract")

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    def test_split_sheet_category_is_accepted(self, client, mock_supabase):
        """split_Sheet category is accepted and mapped to split_sheet."""
        split_file = {**FILE_RECORD, "folder_category": "split_sheet"}
        mock_supabase.table.side_effect = _make_artist_ownership_router(
            project_files_data=[split_file],
        )

        response = client.get(f"/files/artist/{ARTIST_ID}/category/split_Sheet")

        assert response.status_code == 200

    def test_royalty_statement_category(self, client, mock_supabase):
        """royalty_statement category is accepted."""
        royalty_file = {**FILE_RECORD, "folder_category": "royalty_statement"}
        mock_supabase.table.side_effect = _make_artist_ownership_router(
            project_files_data=[royalty_file],
        )

        response = client.get(f"/files/artist/{ARTIST_ID}/category/royalty_statement")

        assert response.status_code == 200
        assert response.json()[0]["folder_category"] == "royalty_statement"


# ---------------------------------------------------------------------------
# GET /projects/{project_id}/contracts
# ---------------------------------------------------------------------------


class TestGetProjectContracts:
    """GET /projects/{project_id}/contracts returns contract files for a project."""

    def test_returns_contracts_for_owned_project(self, client, mock_supabase):
        """Returns contract records when user owns the project."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[CONTRACT_RECORD],
        )

        response = client.get(f"/projects/{PROJECT_ID}/contracts")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == CONTRACT_ID

    def test_returns_empty_list_when_no_contracts(self, client, mock_supabase):
        """Returns empty list when project has no contracts."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[],
        )

        response = client.get(f"/projects/{PROJECT_ID}/contracts")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_403_when_project_not_owned(self, client, mock_supabase):
        """Returns 403 when user does not own the project."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            artists_data=[],
            projects_data=[],
        )

        response = client.get(f"/projects/{PROJECT_ID}/contracts")

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    def test_returns_multiple_contracts(self, client, mock_supabase):
        """Returns all contracts when project has multiple."""
        contract_2 = {**CONTRACT_RECORD, "id": "contract-002", "file_name": "deal2.pdf"}
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[CONTRACT_RECORD, contract_2],
        )

        response = client.get(f"/projects/{PROJECT_ID}/contracts")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


# ---------------------------------------------------------------------------
# GET /projects/{project_id}/documents
# ---------------------------------------------------------------------------


class TestGetProjectDocuments:
    """GET /projects/{project_id}/documents returns contracts and split sheets."""

    def test_returns_documents_for_owned_project(self, client, mock_supabase):
        """Returns document records when user owns the project."""
        split_record = {**FILE_RECORD, "id": "file-ss-001", "folder_category": "split_sheet"}
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[CONTRACT_RECORD, split_record],
        )

        response = client.get(f"/projects/{PROJECT_ID}/documents")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_returns_empty_list_when_no_documents(self, client, mock_supabase):
        """Returns empty list when no contracts or split sheets exist."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            project_files_data=[],
        )

        response = client.get(f"/projects/{PROJECT_ID}/documents")

        assert response.status_code == 200
        assert response.json() == []

    def test_returns_403_when_project_not_owned(self, client, mock_supabase):
        """Returns 403 when user does not own the project."""
        mock_supabase.table.side_effect = _make_project_ownership_router(
            artists_data=[],
            projects_data=[],
        )

        response = client.get(f"/projects/{PROJECT_ID}/documents")

        assert response.status_code == 403


# ---------------------------------------------------------------------------
# DELETE /contracts/{contract_id}
# ---------------------------------------------------------------------------


class TestDeleteContract:
    """DELETE /contracts/{contract_id} deletes a contract and cleans up storage."""

    def _make_delete_router(
        self,
        *,
        ownership_project_files_data=None,
        artists_data=None,
        projects_data=None,
        contract_detail_data=None,
    ):
        """Build a table router for the delete endpoint.

        Delete call sequence:
          1. project_files  — verify_user_owns_contract: get project_id for the contract
          2. artists        — get_user_artist_ids
          3. projects       — verify project belongs to artist
          4. project_files  — fetch full contract record
          5. project_files  — delete record
        """
        if ownership_project_files_data is None:
            ownership_project_files_data = [{"project_id": PROJECT_ID}]
        if artists_data is None:
            artists_data = [{"id": ARTIST_ID}]
        if projects_data is None:
            projects_data = [{"id": PROJECT_ID}]
        if contract_detail_data is None:
            contract_detail_data = [CONTRACT_RECORD]

        # Track how many times project_files was called so we can serve different
        # responses for the ownership check vs the detail fetch.
        state = {"project_files_call": 0}

        def _router(name):
            builder = MockQueryBuilder()
            if name == "artists":
                builder.execute.return_value = MagicMock(data=artists_data, count=len(artists_data))
            elif name == "projects":
                builder.execute.return_value = MagicMock(data=projects_data, count=len(projects_data))
            elif name == "project_files":
                call_num = state["project_files_call"]
                state["project_files_call"] += 1
                if call_num == 0:
                    # First call: ownership check — return project_id only
                    builder.execute.return_value = MagicMock(
                        data=ownership_project_files_data,
                        count=len(ownership_project_files_data),
                    )
                else:
                    # Subsequent calls: full contract record (or delete)
                    builder.execute.return_value = MagicMock(
                        data=contract_detail_data,
                        count=len(contract_detail_data),
                    )
            return builder

        return _router

    def test_deletes_contract_successfully(self, client, mock_supabase):
        """Returns 200 success response when contract is deleted."""
        mock_supabase.table.side_effect = self._make_delete_router()

        response = client.delete(f"/contracts/{CONTRACT_ID}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["contract_id"] == CONTRACT_ID

    def test_response_contains_success_message(self, client, mock_supabase):
        """Success response includes a human-readable message."""
        mock_supabase.table.side_effect = self._make_delete_router()

        response = client.delete(f"/contracts/{CONTRACT_ID}")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert len(data["message"]) > 0

    def test_returns_403_when_contract_not_owned(self, client, mock_supabase):
        """Returns 403 when user does not own the contract's project."""
        mock_supabase.table.side_effect = self._make_delete_router(
            ownership_project_files_data=[{"project_id": PROJECT_ID}],
            artists_data=[],  # get_user_artist_ids returns empty -> ownership fails
            projects_data=[],
        )

        response = client.delete(f"/contracts/{CONTRACT_ID}")

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    def test_returns_403_when_contract_project_not_found(self, client, mock_supabase):
        """Returns 403 when contract's project_id is not found in ownership check."""
        mock_supabase.table.side_effect = self._make_delete_router(
            ownership_project_files_data=[],  # contract not found -> ownership returns False
        )

        response = client.delete(f"/contracts/{CONTRACT_ID}")

        assert response.status_code == 403

    def test_returns_404_when_contract_record_missing(self, client, mock_supabase):
        """Returns 404 when ownership passes but contract record not found in DB."""
        mock_supabase.table.side_effect = self._make_delete_router(
            contract_detail_data=[],  # second project_files call returns empty
        )

        response = client.delete(f"/contracts/{CONTRACT_ID}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_storage_remove_called_when_file_path_present(self, client, mock_supabase):
        """Storage remove is attempted when the contract has a file_path."""
        mock_supabase.table.side_effect = self._make_delete_router()

        client.delete(f"/contracts/{CONTRACT_ID}")

        mock_supabase.storage.from_.assert_called()
