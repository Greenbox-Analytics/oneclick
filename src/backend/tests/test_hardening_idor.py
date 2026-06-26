"""Tests for lower-severity IDOR hardening (Task 9).

Covers:
- POST /credentials rejects artist not owned by the caller (Item B)
- POST /upload rejects a project_id not owned by the caller (Item A)
"""

from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder


def test_create_credential_rejects_unowned_artist(client, mock_supabase):
    """POST /credentials with an artist the caller does not own -> 403."""

    def _router(name):
        b = MockQueryBuilder()
        if name == "artists":
            # verify_user_owns_artist returns empty -> not owned
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    mock_supabase.table.side_effect = _router

    # Required fields from CredentialCreate: artist_id, platform_name, login_identifier, password
    resp = client.post(
        "/credentials",
        json={
            "artist_id": "victim-artist",
            "platform_name": "distrokid",
            "login_identifier": "u",
            "password": "p",
        },
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Access denied"


def test_upload_rejects_unowned_project(client, mock_supabase):
    """POST /upload with a project_id the caller does not own -> 403.

    The artist IS owned (so the artist check passes) but the project belongs
    to a different user's artist.
    """
    import io

    def _router(name):
        b = MockQueryBuilder()
        if name == "artists":
            # verify_user_owns_artist passes (artist IS owned)
            b.execute.return_value = MagicMock(data=[{"id": "my-artist"}], count=1)
        elif name == "projects":
            # verify_user_owns_project: get_user_artist_ids returns the artist,
            # but the project lookup returns empty (project belongs to another user)
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    mock_supabase.table.side_effect = _router

    files = {"file": ("test.txt", io.BytesIO(b"content"), "text/plain")}
    data = {
        "artist_id": "my-artist",
        "category": "contract",
        "project_id": "victim-project",
    }
    resp = client.post("/upload", files=files, data=data)
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Access denied"
