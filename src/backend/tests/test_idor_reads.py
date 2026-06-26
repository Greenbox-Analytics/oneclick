"""IDOR remediation tests: Task 6 — roster reads and artist-overlay reads.

Verifies that:
A. GET /projects/{project_id}/members      → 403 for non-members
B. GET /projects/{project_id}/pending-invites → 403 for non-members
C. GET /registry/artists/{artist_id}/with-teamcard → 403 for non-owners
"""

from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder


def test_members_denied_for_non_member(client, mock_supabase):
    def _router(name):
        b = MockQueryBuilder()
        if name == "project_members":
            b.execute.return_value = MagicMock(data=None)  # get_user_role -> None
        return b

    mock_supabase.table.side_effect = _router
    resp = client.get("/projects/victim-proj/members")
    assert resp.status_code == 403


def test_pending_invites_denied_for_non_member(client, mock_supabase):
    def _router(name):
        b = MockQueryBuilder()
        if name == "project_members":
            b.execute.return_value = MagicMock(data=None)
        return b

    mock_supabase.table.side_effect = _router
    resp = client.get("/projects/victim-proj/pending-invites")
    assert resp.status_code == 403


def test_with_teamcard_denied_for_unowned_artist(client, mock_supabase):
    def _router(name):
        b = MockQueryBuilder()
        if name == "artists":
            b.execute.return_value = MagicMock(data=[], count=0)  # not owned
        return b

    mock_supabase.table.side_effect = _router
    resp = client.get("/registry/artists/victim-artist/with-teamcard")
    assert resp.status_code == 403
