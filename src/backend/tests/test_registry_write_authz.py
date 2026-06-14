"""Tests for registry write authorization (Task 5).

Writes/management are now allowed for anyone with `wa.can_edit` (work owner/admin
OR project owner/admin) rather than the work creator only. Created stakes/licenses/
agreements stamp `user_id = work owner` (NOT the caller) so the
"user_id = work owner" invariant holds when a non-owner admin creates them. The
owner's own stake is flagged `is_owner_stake=true` via a holder-email safety-net.

These tests exercise the SERVICE layer directly with mock_supabase, patching
``registry.service.get_work_access`` (async) to control the authorization result.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry import service
from registry.access import WorkAccess
from tests.conftest import MockQueryBuilder

WORK_ID = "w1"
STAKE_ID = "s1"


def _run(coro):
    return asyncio.run(coro)


# ============================================================
# update_stake — can_edit gate
# ============================================================


def test_update_stake_raises_when_not_allowed():
    """update_stake raises PermissionError when wa.can_edit is False (viewer)."""
    db = MagicMock()
    stake_row = MockQueryBuilder()
    stake_row.execute.return_value = MagicMock(data={"work_id": WORK_ID})
    db.table.side_effect = lambda name: stake_row

    with (
        patch.object(service, "get_work_access", AsyncMock(return_value=WorkAccess(work_role="viewer"))),
        pytest.raises(PermissionError),
    ):
        _run(service.update_stake(db, "viewer-user", STAKE_ID, {"percentage": 10.0}))


def test_update_stake_succeeds_when_allowed():
    """update_stake succeeds when wa.can_edit is True (owner)."""
    db = MagicMock()
    lookup_builder = MockQueryBuilder()
    lookup_builder.execute.return_value = MagicMock(data={"work_id": WORK_ID})

    update_builder = MockQueryBuilder()
    update_builder.execute.return_value = MagicMock(data=[{"id": STAKE_ID, "percentage": 10.0}])

    call_count = [0]

    def table_side_effect(name):
        call_count[0] += 1
        return lookup_builder if call_count[0] == 1 else update_builder

    db.table.side_effect = table_side_effect

    with patch.object(service, "get_work_access", AsyncMock(return_value=WorkAccess(work_role="owner"))):
        result = _run(service.update_stake(db, "owner-user", STAKE_ID, {"percentage": 10.0}))

    assert result == {"id": STAKE_ID, "percentage": 10.0}


def test_update_stake_returns_none_when_stake_missing():
    """update_stake returns None (not PermissionError) when the stake row doesn't exist."""
    db = MagicMock()
    lookup_builder = MockQueryBuilder()
    lookup_builder.execute.return_value = MagicMock(data=None)
    db.table.side_effect = lambda name: lookup_builder

    # get_work_access should not even be reached; patch it to blow up if it is.
    with patch.object(service, "get_work_access", AsyncMock(side_effect=AssertionError("should not be called"))):
        result = _run(service.update_stake(db, "owner-user", STAKE_ID, {"percentage": 10.0}))

    assert result is None


def test_delete_stake_raises_when_not_allowed():
    """delete_stake raises PermissionError when wa.can_edit is False."""
    db = MagicMock()
    stake_row = MockQueryBuilder()
    stake_row.execute.return_value = MagicMock(data={"work_id": WORK_ID})
    db.table.side_effect = lambda name: stake_row

    with (
        patch.object(service, "get_work_access", AsyncMock(return_value=WorkAccess(work_role="viewer"))),
        pytest.raises(PermissionError),
    ):
        _run(service.delete_stake(db, "viewer-user", STAKE_ID))


# ============================================================
# create_stake — user_id stamping + is_owner_stake safety-net
# ============================================================


def _capture_insert_db(work_owner_id):
    """Build a mock db whose ownership_stakes.insert captures the inserted row.

    works_registry lookup returns {"user_id": work_owner_id}.
    Returns (db, captured) where captured["data"] is set on insert.
    """
    db = MagicMock()
    captured = {}

    work_builder = MockQueryBuilder()
    work_builder.execute.return_value = MagicMock(data={"user_id": work_owner_id})

    stakes_builder = MockQueryBuilder()

    def _insert(payload):
        captured["data"] = payload
        return stakes_builder

    stakes_builder.insert = MagicMock(side_effect=_insert)
    stakes_builder.execute.return_value = MagicMock(data=[{"id": STAKE_ID}])

    def table_side_effect(name):
        if name == "works_registry":
            return work_builder
        return stakes_builder

    db.table.side_effect = table_side_effect
    return db, captured


def test_create_stake_stamps_user_id_to_work_owner():
    """create_stake stamps data['user_id'] to the WORK OWNER even when the caller
    is a non-owner admin."""
    db, captured = _capture_insert_db(work_owner_id="realowner")

    data = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "Some Producer",
        "holder_role": "producer",
        "percentage": 50.0,
        "holder_email": "producer@x.com",
    }

    with (
        patch.object(service, "get_work_access", AsyncMock(return_value=WorkAccess(work_role="admin"))),
        # _resolve_auth_email returns a non-matching owner email so is_owner_stake stays off
        patch.object(service, "_resolve_auth_email", return_value="owner@x.com"),
    ):
        _run(service.create_stake(db, "adminuser", data))

    assert captured["data"]["user_id"] == "realowner"
    assert not captured["data"].get("is_owner_stake")


def test_create_stake_raises_when_not_allowed():
    """create_stake raises PermissionError when wa.can_edit is False."""
    db = MagicMock()
    db.table.side_effect = lambda name: MockQueryBuilder()

    data = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "X",
        "holder_role": "producer",
        "percentage": 50.0,
    }

    with (
        patch.object(service, "get_work_access", AsyncMock(return_value=WorkAccess(work_role="viewer"))),
        pytest.raises(PermissionError),
    ):
        _run(service.create_stake(db, "viewer-user", data))


def test_create_stake_sets_is_owner_stake_when_email_matches():
    """create_stake sets is_owner_stake=True when holder_email matches owner's auth
    email and there's no collaborator link."""
    db, captured = _capture_insert_db(work_owner_id="realowner")

    data = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "The Owner",
        "holder_role": "artist",
        "percentage": 100.0,
        "holder_email": "owner@x.com",
        # no collaborator_id
    }

    with (
        patch.object(service, "get_work_access", AsyncMock(return_value=WorkAccess(work_role="owner"))),
        patch.object(service, "_resolve_auth_email", return_value="owner@x.com"),
    ):
        _run(service.create_stake(db, "realowner", data))

    assert captured["data"]["is_owner_stake"] is True
    assert captured["data"]["user_id"] == "realowner"


def test_create_stake_does_not_set_is_owner_stake_when_collaborator_linked():
    """create_stake does NOT set is_owner_stake when collaborator_id is present, even
    if the holder email matches the owner email."""
    db, captured = _capture_insert_db(work_owner_id="realowner")

    data = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "Collab",
        "holder_role": "producer",
        "percentage": 30.0,
        "holder_email": "owner@x.com",  # matches owner email...
        "collaborator_id": "collab-1",  # ...but is collaborator-linked
    }

    with (
        patch.object(service, "get_work_access", AsyncMock(return_value=WorkAccess(work_role="owner"))),
        # _resolve_auth_email should not even be consulted; assert if it is.
        patch.object(service, "_resolve_auth_email", side_effect=AssertionError("should not be called")),
    ):
        _run(service.create_stake(db, "realowner", data))

    assert not captured["data"].get("is_owner_stake")


# ============================================================
# validate_stake_percentage — work-scoped, not user-scoped
# ============================================================


def test_validate_stake_percentage_sums_across_work_returns_false_when_over():
    """validate_stake_percentage sums ALL stakes of the type on the work (regardless
    of user_id); 80 existing + 30 new > 100 -> False."""
    db = MagicMock()
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[{"id": "other", "percentage": 80.0}])
    db.table.side_effect = lambda name: builder

    valid = _run(service.validate_stake_percentage(db, "anyuser", WORK_ID, "master", 30.0))
    assert valid is False


def test_validate_stake_percentage_sums_across_work_returns_true_when_under():
    """validate_stake_percentage: 80 existing + 20 new == 100 -> True."""
    db = MagicMock()
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[{"id": "other", "percentage": 80.0}])
    db.table.side_effect = lambda name: builder

    valid = _run(service.validate_stake_percentage(db, "anyuser", WORK_ID, "master", 20.0))
    assert valid is True
