"""Tests for access-filtered registry reads (Task 4).

These exercise the *_filtered service helpers directly, patching
``registry.service.get_work_access`` so each test pins a specific WorkAccess
shape, and patching the underlying unfiltered getters to return known rows.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from registry import service
from registry.access import WorkAccess


def _run(coro):
    return asyncio.run(coro)


def _patch_access(wa):
    """Patch registry.service.get_work_access to return *wa* (awaitable)."""
    return patch.object(service, "get_work_access", AsyncMock(return_value=wa))


def _owner_db(owner_id="owner-1", full_name="Owner Name"):
    """MagicMock db so service._owner_identity resolves a real owner row."""
    db = MagicMock()

    def table(name):
        q = MagicMock()
        q.select.return_value = q
        q.eq.return_value = q
        single = MagicMock()
        if name == "works_registry":
            single.execute.return_value = MagicMock(data={"user_id": owner_id})
        elif name == "profiles":
            single.execute.return_value = MagicMock(data={"full_name": full_name})
        else:
            single.execute.return_value = MagicMock(data=None)
        q.maybe_single.return_value = single
        return q

    db.table.side_effect = table
    return db


def test_viewer_licenses_filtered_to_visible_ids():
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", visible_license_ids={"L1"})
    db = MagicMock()
    with _patch_access(wa), patch.object(service, "get_licenses", AsyncMock(return_value=[{"id": "L1"}, {"id": "L2"}])):
        result = _run(service.get_licenses_filtered(db, "u1", "w1"))
    assert result == [{"id": "L1"}]


def test_viewer_stakes_filtered_to_visible_ids():
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", visible_stake_ids={"s_owner", "s_me"})
    db = MagicMock()
    rows = [{"id": "s_owner"}, {"id": "s_me"}, {"id": "s_other"}]
    with _patch_access(wa), patch.object(service, "get_stakes", AsyncMock(return_value=rows)):
        result = _run(service.get_stakes_filtered(db, "u1", "w1"))
    assert [r["id"] for r in result] == ["s_owner", "s_me"]


def test_no_access_returns_none():
    wa = WorkAccess()  # work_role/project_role = none -> can_view False
    db = MagicMock()
    with _patch_access(wa):
        assert _run(service.get_licenses_filtered(db, "u1", "w1")) is None
        assert _run(service.get_stakes_filtered(db, "u1", "w1")) is None
        assert _run(service.get_agreements_filtered(db, "u1", "w1")) is None
        assert _run(service.get_collaborators_filtered(db, "u1", "w1")) is None
        assert _run(service.get_work_filtered(db, "u1", "w1")) is None


def test_no_breakdown_viewer_sees_owner_and_self_only():
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", can_see_full_ownership=False)
    db = _owner_db(owner_id="owner-1")
    rows = [
        {"id": "c1", "name": "Me", "role": "Writer", "status": "confirmed", "email": "me@x.com"},
        {"id": "c2", "name": "Other", "role": "Producer", "status": "confirmed", "email": "other@x.com"},
    ]
    with _patch_access(wa), patch.object(service, "get_collaborators", AsyncMock(return_value=rows)):
        result = _run(service.get_collaborators_filtered(db, "u1", "w1"))
    ids = {r["id"] for r in result}
    assert ids == {"owner:owner-1", "c1"}
    assert "c2" not in ids


def test_full_ownership_viewer_sees_others_but_stripped():
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", can_see_full_ownership=True)
    db = _owner_db(owner_id="owner-1")
    rows = [
        {"id": "c1", "name": "Me", "role": "Writer", "status": "confirmed", "email": "me@x.com"},
        {
            "id": "c2",
            "name": "Other",
            "role": "Producer",
            "status": "confirmed",
            "email": "other@x.com",
            "terms": "secret terms",
            "access_level": "admin",
            "collaborator_user_id": "u2",
        },
    ]
    with _patch_access(wa), patch.object(service, "get_collaborators", AsyncMock(return_value=rows)):
        result = _run(service.get_collaborators_filtered(db, "u1", "w1"))

    by_id = {r["id"]: r for r in result}
    # Owner row present, self (c1) present in full, c2 present but stripped.
    assert "owner:owner-1" in by_id
    assert by_id["c1"]["email"] == "me@x.com"  # self keeps its full row
    c2 = by_id["c2"]
    assert "email" not in c2
    assert "terms" not in c2
    assert "access_level" not in c2
    # Stripped row keeps display-safe fields.
    assert c2["name"] == "Other"
    assert c2["role"] == "Producer"
    assert c2["collaborator_user_id"] == "u2"


def test_all_visible_returns_full_unfiltered_rows():
    # Project member / elevated -> all_visible() True -> no filtering applied.
    wa = WorkAccess(work_role="owner")
    wa._all_visible = True
    db = MagicMock()
    collab_rows = [
        {"id": "c1", "name": "Me", "email": "me@x.com", "terms": "t", "access_level": "admin"},
        {"id": "c2", "name": "Other", "email": "other@x.com", "terms": "t2", "access_level": "viewer"},
    ]
    license_rows = [{"id": "L1"}, {"id": "L2"}]
    with (
        _patch_access(wa),
        patch.object(service, "get_collaborators", AsyncMock(return_value=collab_rows)),
        patch.object(service, "get_licenses", AsyncMock(return_value=license_rows)),
    ):
        collabs = _run(service.get_collaborators_filtered(db, "u1", "w1"))
        licenses = _run(service.get_licenses_filtered(db, "u1", "w1"))
    assert collabs == collab_rows  # full rows, including email/terms/access_level
    assert licenses == license_rows
