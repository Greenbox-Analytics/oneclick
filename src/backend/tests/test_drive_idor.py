"""IDOR remediation tests for Google Drive cross-tenant write endpoints.

Verifies that POST /integrations/google-drive/import and
POST /integrations/google-drive/sync/setup enforce project membership
before allowing any Drive write operation.
"""

from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder


def test_import_denied_for_non_member(client, mock_supabase):
    """Non-member cannot import a Drive file into a project they don't belong to."""

    def _router(name):
        b = MockQueryBuilder()
        if name == "project_members":
            b.execute.return_value = MagicMock(data=None)  # get_user_role -> None
        return b

    mock_supabase.table.side_effect = _router
    resp = client.post(
        "/integrations/google-drive/import",
        json={"project_id": "victim-proj", "drive_file_id": "f1"},
    )
    assert resp.status_code == 403


def test_sync_setup_denied_for_non_member(client, mock_supabase):
    """Non-member cannot configure Drive sync for a project they don't belong to."""

    def _router(name):
        b = MockQueryBuilder()
        if name == "project_members":
            b.execute.return_value = MagicMock(data=None)  # get_user_role -> None
        return b

    mock_supabase.table.side_effect = _router
    resp = client.post(
        "/integrations/google-drive/sync/setup",
        json={"project_id": "victim-proj", "drive_folder_id": "fold1"},
    )
    assert resp.status_code == 403
