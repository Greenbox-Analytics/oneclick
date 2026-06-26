"""Per-contract (resource-level) access tests.

Covers the access-model enhancement that lets a work-only collaborator who was
GRANTED a specific contract/document run Zoe and OneClick on THAT file, even when
they are not a project member or work owner.

The security boundary is the file itself: `user_can_access_file` is True when the
caller is a member of the file's project OR the file is linked (via work_files) to
a work where the caller is elevated / was granted this specific file.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from registry.access import WorkAccess
from tests.conftest import MockQueryBuilder, _default_table_side_effect

# --- helpers ---------------------------------------------------------------


def _granted_work_access(file_id: str) -> WorkAccess:
    """A work-only collaborator who was granted exactly `file_id` (no elevation)."""
    wa = WorkAccess(work_role="viewer")
    wa._all_visible = False
    wa.visible_file_ids = {file_id}
    return wa


def _empty_work_access() -> WorkAccess:
    """A confirmed collaborator with NO grant on the file (not elevated, no file)."""
    wa = WorkAccess(work_role="viewer")
    wa._all_visible = False
    wa.visible_file_ids = set()
    return wa


def _not_member_db(file_id: str, project_id: str, work_ids: list[str], work_project_id: str = None) -> MagicMock:
    """Mock db where the caller is NOT a project member / artist owner, but the file
    is linked to the given works. `work_project_id` is the project the linked work(s)
    belong to; it defaults to the file's own project (the legitimate same-project case).
    Set it to a different value to simulate a cross-project work_files link.
    Subscription tables fall through to Pro defaults."""
    if work_project_id is None:
        work_project_id = project_id
    db = MagicMock()

    def _router(name):
        if name == "project_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"project_id": project_id}], count=1)
            return b
        if name == "project_members":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=None)  # get_user_role -> None
            return b
        if name == "artists":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)  # owns no artists
            return b
        if name == "projects":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)
            return b
        if name == "work_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"work_id": w} for w in work_ids], count=len(work_ids))
            return b
        if name == "works_registry":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"project_id": work_project_id}], count=1)
            return b
        return _default_table_side_effect(name)

    db.table.side_effect = _router
    return db


# --- 1. user_can_access_file: granted collaborator -> True -----------------


def _run_user_can_access_file(db, user_id, file_id, work_access):
    """Run user_can_access_file directly, patching both get_work_access and the
    global get_supabase_client (used by the sync project-ownership fallback) to db."""
    import main

    with (
        patch("registry.access.get_work_access", AsyncMock(return_value=work_access)),
        patch("main.get_supabase_client", return_value=db),
    ):
        return asyncio.run(main.user_can_access_file(db, user_id, file_id))


def test_user_can_access_file_true_for_granted_collaborator():
    db = _not_member_db("file-1", "victim-proj", ["w1"])
    result = _run_user_can_access_file(db, "collab-user", "file-1", _granted_work_access("file-1"))
    assert result is True


def test_user_can_access_file_true_for_elevated_work_access():
    # Elevated on a work in the file's OWN project -> access granted.
    db = _not_member_db("file-1", "victim-proj", ["w1"])  # work_project_id defaults to same project
    wa = WorkAccess(work_role="admin")
    wa._all_visible = True
    result = _run_user_can_access_file(db, "collab-user", "file-1", wa)
    assert result is True


def test_user_can_access_file_false_for_elevated_cross_project_link():
    # Elevated on a work whose project DIFFERS from the file's project (a cross-project
    # work_files link). all_visible must NOT widen access to another project's file.
    db = _not_member_db("file-1", "victim-proj", ["w1"], work_project_id="attacker-proj")
    wa = WorkAccess(work_role="admin")
    wa._all_visible = True
    result = _run_user_can_access_file(db, "collab-user", "file-1", wa)
    assert result is False


# --- 2. user_can_access_file: NOT granted -> False -------------------------


def test_user_can_access_file_false_when_not_granted():
    db = _not_member_db("file-1", "victim-proj", ["w1"])
    result = _run_user_can_access_file(db, "collab-user", "file-1", _empty_work_access())
    assert result is False


def test_user_can_access_file_false_when_no_work_links():
    # File exists but is linked to no works and caller is not a member.
    db = _not_member_db("file-1", "victim-proj", [])
    result = _run_user_can_access_file(db, "collab-user", "file-1", _empty_work_access())
    assert result is False


# --- 3. Zoe ask-stream: granted contract passes; unowned artist dropped -----


def test_zoe_ask_stream_collaborator_granted_contract_not_403(client, mock_supabase):
    """Collaborator with a granted contract AND an unowned artist_id:
    - the contract passes the access gate (not 403)
    - the unowned artist_id is dropped to context, NOT a 403.
    """

    def _router(name):
        if name == "project_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"project_id": "victim-proj"}], count=1)
            return b
        if name == "project_members":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=None)  # not a member
            return b
        if name == "artists":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)  # owns no artists
            return b
        if name == "projects":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)
            return b
        if name == "work_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"work_id": "w1"}], count=1)
            return b
        return _default_table_side_effect(name)

    mock_supabase.table.side_effect = _router

    # Stub the chatbot so the stream terminates immediately and asserts only on the gate.
    fake_chatbot = MagicMock()
    fake_chatbot.ask_stream.return_value = iter([])

    with (
        patch("registry.access.get_work_access", AsyncMock(return_value=_granted_work_access("c1"))),
        patch("main.get_zoe_chatbot", return_value=fake_chatbot),
    ):
        resp = client.post(
            "/zoe/ask-stream",
            json={"query": "hi", "artist_id": "victim-artist", "contract_ids": ["c1"]},
        )
    assert resp.status_code != 403


def test_zoe_ask_stream_rejects_ungranted_contract(client, mock_supabase):
    """Deny-by-default still holds: a contract the caller wasn't granted -> 403."""

    def _router(name):
        if name == "project_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"project_id": "victim-proj"}], count=1)
            return b
        if name == "project_members":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=None)
            return b
        if name == "artists":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)
            return b
        if name == "projects":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)
            return b
        if name == "work_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"work_id": "w1"}], count=1)
            return b
        return _default_table_side_effect(name)

    mock_supabase.table.side_effect = _router

    with patch("registry.access.get_work_access", AsyncMock(return_value=_empty_work_access())):
        resp = client.post(
            "/zoe/ask-stream",
            json={"query": "hi", "contract_ids": ["c1"]},
        )
    assert resp.status_code == 403


# --- 4. OneClick calculate-royalties: granted statement+contract -----------


def _oneclick_router(grant_file_ids: set[str], work_ids: list[str]):
    """Build a side_effect for OneClick tests. Caller is NOT a member; files are
    linked to `work_ids`. `get_work_access` (patched separately) decides per-file."""

    def _router(name):
        if name == "project_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"project_id": "victim-proj"}], count=1)
            return b
        if name == "project_members":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=None)
            return b
        if name == "artists":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)
            return b
        if name == "projects":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)
            return b
        if name == "work_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"work_id": w} for w in work_ids], count=len(work_ids))
            return b
        return _default_table_side_effect(name)

    return _router


def test_oneclick_calculate_collaborator_granted_inputs_not_403(client, mock_supabase):
    """Collaborator granted both the statement and the contract -> NOT 403."""
    mock_supabase.table.side_effect = _oneclick_router({"stmt-1", "c1"}, ["w1"])

    # get_work_access grants whichever file is asked about. We can't vary by file
    # via a single AsyncMock easily, so return all_visible False but grant both files.
    wa = WorkAccess(work_role="viewer")
    wa._all_visible = False
    wa.visible_file_ids = {"stmt-1", "c1"}

    # Stop downstream calc work by making the calculator raise after the gate; we only
    # assert the gate did NOT 403. A non-403 status proves the access check passed.
    with patch("registry.access.get_work_access", AsyncMock(return_value=wa)):
        resp = client.post(
            "/oneclick/calculate-royalties",
            json={"project_id": "victim-proj", "royalty_statement_file_id": "stmt-1", "contract_ids": ["c1"]},
        )
    assert resp.status_code != 403


def test_oneclick_calculate_missing_grant_on_contract_403(client, mock_supabase):
    """Statement granted but a contract is NOT granted -> 403."""
    mock_supabase.table.side_effect = _oneclick_router({"stmt-1"}, ["w1"])

    # Grant only the statement, not the contract.
    wa = WorkAccess(work_role="viewer")
    wa._all_visible = False
    wa.visible_file_ids = {"stmt-1"}

    with patch("registry.access.get_work_access", AsyncMock(return_value=wa)):
        resp = client.post(
            "/oneclick/calculate-royalties",
            json={"project_id": "victim-proj", "royalty_statement_file_id": "stmt-1", "contract_ids": ["c1"]},
        )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Access denied"
