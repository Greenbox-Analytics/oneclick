import asyncio
import json
from unittest.mock import MagicMock

from registry import access, router
from registry.access import WorkAccess


def _db_for(*, work_owner, project_role_row=None, collab_row=None, stakes=None, grants=None):
    """MagicMock db whose .table(name) returns a fake chainable query for that table."""
    db = MagicMock()

    def table(name):
        q = MagicMock()
        rows = {
            "works_registry": [{"id": "w1", "user_id": work_owner, "project_id": "p1"}],
            "project_members": [project_role_row] if project_role_row else [],
            "registry_collaborators": [collab_row] if collab_row else [],
            "ownership_stakes": stakes or [],
            "registry_access_grants": grants or [],
        }[name]
        q.select.return_value = q
        q.eq.return_value = q
        q.in_.return_value = q
        q.neq.return_value = q
        exec_res = MagicMock()
        exec_res.data = rows
        q.execute.return_value = exec_res
        single = MagicMock()
        single.execute.return_value = MagicMock(data=(rows[0] if rows else None))
        q.single.return_value = single
        q.maybe_single.return_value = single
        return q

    db.table.side_effect = table
    return db


def test_owner_sees_everything():
    db = _db_for(work_owner="u1")
    wa = asyncio.run(access.get_work_access(db, "u1", "w1"))
    assert wa.work_role == "owner"
    assert wa.can_edit and wa.can_manage and wa.can_delete and wa.can_see_full_ownership


def test_project_admin_is_elevated_but_not_work_owner():
    db = _db_for(work_owner="someone_else", project_role_row={"user_id": "u2", "role": "admin"})
    wa = asyncio.run(access.get_work_access(db, "u2", "w1"))
    assert wa.work_role == "none"
    assert wa.project_role == "admin"
    assert wa.elevated and wa.can_manage and wa.can_edit


def test_project_editor_can_view_not_edit():
    db = _db_for(work_owner="x", project_role_row={"user_id": "u3", "role": "editor"})
    wa = asyncio.run(access.get_work_access(db, "u3", "w1"))
    assert wa.can_view and not wa.can_edit and not wa.can_manage


def test_viewer_sees_only_own_and_owner_stakes():
    collab = {"id": "c1", "collaborator_user_id": "u4", "access_level": "viewer", "status": "confirmed"}
    stakes = [
        {"id": "s_owner", "is_owner_stake": True, "collaborator_id": None},
        {"id": "s_me", "is_owner_stake": False, "collaborator_id": "c1"},
        {"id": "s_other", "is_owner_stake": False, "collaborator_id": "c2"},
    ]
    db = _db_for(work_owner="x", collab_row=collab, stakes=stakes)
    wa = asyncio.run(access.get_work_access(db, "u4", "w1"))
    assert wa.work_role == "viewer"
    assert wa.visible_stake_ids == {"s_owner", "s_me"}
    assert not wa.can_see_full_ownership


def test_viewer_with_ownership_breakdown_sees_all_stakes():
    collab = {"id": "c1", "collaborator_user_id": "u4", "access_level": "viewer", "status": "confirmed"}
    stakes = [
        {"id": "s_owner", "is_owner_stake": True, "collaborator_id": None},
        {"id": "s_other", "is_owner_stake": False, "collaborator_id": "c2"},
    ]
    grants = [{"resource_type": "ownership_breakdown", "resource_id": None, "collaborator_id": "c1"}]
    db = _db_for(work_owner="x", collab_row=collab, stakes=stakes, grants=grants)
    wa = asyncio.run(access.get_work_access(db, "u4", "w1"))
    assert wa.can_see_full_ownership
    assert wa.visible_stake_ids == {"s_owner", "s_other"}


def test_non_member_gets_nothing():
    db = _db_for(work_owner="x")
    wa = asyncio.run(access.get_work_access(db, "stranger", "w1"))
    assert wa.work_role == "none" and wa.project_role == "none"
    assert not wa.can_view


def test_work_access_endpoint_serializes_sets_to_lists(monkeypatch):
    """The /works/{id}/access endpoint must convert the WorkAccess sets to JSON-able
    lists. The resolver itself is tested above; here we only verify serialization."""
    wa = WorkAccess(
        work_role="viewer",
        project_role="none",
        my_collaborator_id="c1",
        is_project_member=False,
        can_see_full_ownership=False,
        visible_stake_ids={"s1", "s2"},
        visible_file_ids={"f1"},
        visible_audio_ids=set(),
        visible_license_ids={"l1"},
        visible_agreement_ids=set(),
    )

    async def _fake_get_work_access(_db, _user_id, _work_id):
        return wa

    monkeypatch.setattr(router, "_get_supabase", lambda: MagicMock())
    monkeypatch.setattr(router, "get_work_access", _fake_get_work_access)

    result = asyncio.run(router.work_access("w1", user_id="u1"))

    # Sets must become lists so the response is JSON-serializable.
    for key in (
        "visible_stake_ids",
        "visible_file_ids",
        "visible_audio_ids",
        "visible_license_ids",
        "visible_agreement_ids",
    ):
        assert isinstance(result[key], list), f"{key} should be a list, got {type(result[key])}"
    assert isinstance(result["visible_file_ids"], list)
    assert sorted(result["visible_stake_ids"]) == ["s1", "s2"]

    # Properties surfaced correctly and the whole dict round-trips through json.
    assert result["can_view"] is True
    assert result["can_edit"] is False
    assert result["my_collaborator_id"] == "c1"
    assert result["all_visible"] is False
    json.dumps(result)  # raises TypeError if any value (e.g. a set) is unserializable
