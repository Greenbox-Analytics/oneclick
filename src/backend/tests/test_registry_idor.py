"""IDOR regression tests for registry work-scoped endpoints (Task 5).

These verify that the work-scoped link/invite/derive endpoints deny callers
who have no access to the target work. The access resolver itself is mocked to
an empty WorkAccess() (deny-all) so we exercise the gate, not the resolver.

Patch targets must match WHERE get_work_access is actually called:
- work_links_service for the file/audio link endpoints
- registry.router for the invite + derive-from-contracts checks
- registry.service for get_works_by_project (via get_user_role)
"""

from unittest.mock import AsyncMock, MagicMock, patch

from registry.access import WorkAccess
from tests.conftest import MockQueryBuilder, _default_table_side_effect

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _sub_wrap(fn):
    """Route subscription tables through the Pro-tier default so gating passes."""

    def _wrapped(name):
        if name in _SUBSCRIPTION_TABLES:
            return _default_table_side_effect(name)
        return fn(name)

    return _wrapped


WORK_ID = "aaaaaaaa-0000-0000-0000-00000000dead"
LINK_ID = "ffffffff-0000-0000-0000-00000000dead"
FILE_ID = "dddddddd-0000-0000-0000-00000000dead"
AUDIO_FILE_ID = "eeeeeeee-0000-0000-0000-00000000dead"
PROJECT_ID = "bbbbbbbb-0000-0000-0000-00000000dead"
CONTRACT_ID = "cccccccc-0000-0000-0000-00000000dead"


# ============================================================
# Item A — work file/audio link endpoints deny without access
# ============================================================


def test_list_work_files_denied_without_access(client):
    with patch("registry.work_links_service.get_work_access", AsyncMock(return_value=WorkAccess())):
        resp = client.get(f"/registry/works/{WORK_ID}/files")
    assert resp.status_code == 403


def test_link_file_denied_without_edit(client):
    with patch("registry.work_links_service.get_work_access", AsyncMock(return_value=WorkAccess())):
        resp = client.post(f"/registry/works/{WORK_ID}/files?file_id={FILE_ID}")
    assert resp.status_code == 403


def test_unlink_file_denied_without_edit(client):
    with patch("registry.work_links_service.get_work_access", AsyncMock(return_value=WorkAccess())):
        resp = client.delete(f"/registry/works/{WORK_ID}/files/{LINK_ID}")
    assert resp.status_code == 403


def test_list_work_audio_denied_without_access(client):
    with patch("registry.work_links_service.get_work_access", AsyncMock(return_value=WorkAccess())):
        resp = client.get(f"/registry/works/{WORK_ID}/audio")
    assert resp.status_code == 403


def test_link_audio_denied_without_edit(client):
    with patch("registry.work_links_service.get_work_access", AsyncMock(return_value=WorkAccess())):
        resp = client.post(f"/registry/works/{WORK_ID}/audio?audio_file_id={AUDIO_FILE_ID}")
    assert resp.status_code == 403


def test_unlink_audio_denied_without_edit(client):
    with patch("registry.work_links_service.get_work_access", AsyncMock(return_value=WorkAccess())):
        resp = client.delete(f"/registry/works/{WORK_ID}/audio/{LINK_ID}")
    assert resp.status_code == 403


# A viewer (can_view but not can_edit) may read but not mutate links.
def _viewer():
    return WorkAccess(work_role="viewer")


def test_link_file_denied_for_viewer(client):
    with patch("registry.work_links_service.get_work_access", AsyncMock(return_value=_viewer())):
        resp = client.post(f"/registry/works/{WORK_ID}/files?file_id={FILE_ID}")
    assert resp.status_code == 403


# ============================================================
# Item B — get_works_by_project denies non-members
# ============================================================


def test_list_works_by_project_denied_for_non_member(client):
    # get_user_role returns None => PermissionError => 403. get_works_by_project
    # imports it lazily from projects.service, so patch it at the source module.
    with patch("projects.service.get_user_role", AsyncMock(return_value=None)):
        resp = client.get(f"/registry/works/by-project/{PROJECT_ID}")
    assert resp.status_code == 403


def test_list_works_by_project_allowed_for_member(client, mock_supabase):
    works_builder = MockQueryBuilder()
    works_builder.execute.return_value = MagicMock(data=[{"id": "w1", "project_id": PROJECT_ID}], count=1)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: works_builder)
    with patch("projects.service.get_user_role", AsyncMock(return_value="editor")):
        resp = client.get(f"/registry/works/by-project/{PROJECT_ID}")
    assert resp.status_code == 200
    assert "works" in resp.json()


# ============================================================
# Item C — get_note scopes to the owner (404 for other users' notes)
# ============================================================


def test_get_note_returns_404_for_other_users_note(client, mock_supabase):
    # maybe_single() scoped by user_id returns no row for someone else's note.
    note_builder = MockQueryBuilder()
    note_builder.execute.return_value = MagicMock(data=None, count=0)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: note_builder)
    resp = client.get("/registry/notes/some-other-users-note")
    assert resp.status_code == 404


# ============================================================
# Item D — plain invite endpoint requires can_manage
# ============================================================


def test_invite_denied_without_manage(client):
    with patch("registry.router.get_work_access", AsyncMock(return_value=WorkAccess())):
        resp = client.post(
            "/registry/collaborators/invite",
            json={"work_id": WORK_ID, "email": "x@example.com", "name": "X", "role": "writer"},
        )
    assert resp.status_code == 403


def test_invite_denied_for_viewer(client):
    with patch("registry.router.get_work_access", AsyncMock(return_value=_viewer())):
        resp = client.post(
            "/registry/collaborators/invite",
            json={"work_id": WORK_ID, "email": "x@example.com", "name": "X", "role": "writer"},
        )
    assert resp.status_code == 403


# ============================================================
# Item E — derive-from-contracts validates contract linkage
# ============================================================


def test_derive_denied_when_contract_not_linked_to_work(client, mock_supabase):
    """can_manage passes, but a supplied contract id not linked to the work => 403."""
    # work_files linkage lookup returns no rows for the supplied contract id.
    work_files_builder = MockQueryBuilder()
    work_files_builder.execute.return_value = MagicMock(data=[], count=0)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: work_files_builder)

    owner = WorkAccess(work_role="owner")
    with patch("registry.router.get_work_access", AsyncMock(return_value=owner)):
        resp = client.post(
            "/registry/collaborators/derive-from-contracts",
            json={
                "work_id": WORK_ID,
                "name": "Marcus",
                "email": "m@example.com",
                "contract_file_ids": [CONTRACT_ID],
            },
        )
    assert resp.status_code == 403
