"""Boards on Teams — org-side hooks: roster endpoint, offboard cleanup, dissolve revert."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from orgs import service
from tests.conftest import MockQueryBuilder

ORG, ME, OTHER = "org-1", "u-me", "u-other"


def _db(tbl: dict, *, member=True):
    builders: dict[str, MockQueryBuilder] = {}

    def _side(name):
        # ONE builder per table name, memoized: `db.table("x")` in a test must
        # return the same object the service used, or `.delete.assert_called()`
        # inspects a fresh mock that has seen nothing.
        if name not in builders:
            b = MockQueryBuilder()
            if name in tbl:
                b.execute.return_value = tbl[name]
            builders[name] = b
        return builders[name]

    db = MagicMock()
    db.table.side_effect = _side
    db.rpc.return_value.execute.return_value = MagicMock(data=member)  # is_org_member / is_org_admin
    return db


async def test_roster_returns_active_seats_with_profiles():
    db = _db(
        {
            "org_members": MagicMock(
                data=[{"user_id": ME, "role": "admin", "org_id": ORG}, {"user_id": OTHER, "role": "member"}]
            ),
            "organizations": MagicMock(data=[{"id": ORG}]),  # live_org_ids: not archived, not lapsed
            "profiles": MagicMock(data=[{"id": ME, "full_name": "Me", "avatar_url": None}]),
        }
    )
    out = await service.list_org_roster(db, ME, ORG)
    assert out == [
        {"user_id": ME, "role": "admin", "full_name": "Me", "avatar_url": None},
        {"user_id": OTHER, "role": "member", "full_name": None, "avatar_url": None},
    ]


async def test_roster_denied_in_a_lapsed_or_archived_org():
    """A member seat is not enough — the org must be LIVE, like every other
    predicate this feature added. A lapsed org's boards and artists are inert;
    its roster must not still hand out names and avatars."""
    db = _db(
        {
            "org_members": MagicMock(data=[{"user_id": ME, "role": "admin", "org_id": ORG}]),
            "organizations": MagicMock(data=[]),  # archived or lapsed -> not live
        }
    )
    with pytest.raises(HTTPException) as e:
        await service.list_org_roster(db, ME, ORG)
    assert e.value.status_code == 404


async def test_roster_non_member_404():
    with pytest.raises(HTTPException) as e:
        await service.list_org_roster(_db({}, member=False), ME, ORG)
    assert e.value.status_code == 404


def test_remove_purges_board_membership_but_not_assignments(monkeypatch):
    monkeypatch.setattr("orgs.projects.revoke_org_granted_memberships", lambda *a, **k: None)
    db = _db({"boards": MagicMock(data=[{"id": "b1"}])})
    service._revoke_offboarded_member_access(db, ORG, OTHER, "removed")
    tables = [c.args[0] for c in db.table.call_args_list]
    assert "board_members" in tables
    # SCOPED to this ONE user. Without asserting the filter, dropping
    # .eq("user_id", ...) — which would purge EVERY member's rows on the org's
    # boards — keeps the suite green, because MockQueryBuilder.eq() discards
    # its arguments and only the call record survives.
    db.table("board_members").delete.return_value.eq.assert_called_with("user_id", OTHER)
    # Assignments are HISTORY, not access — access is already denied by the
    # inactive seat, so nothing deletes them.
    assert "board_task_assignees" not in tables


def test_suspend_destroys_nothing(monkeypatch):
    """Suspend is reversible (reactivate_member); a deleted board_members row is not."""
    monkeypatch.setattr("orgs.projects.revoke_org_granted_memberships", lambda *a, **k: None)
    db = _db({"boards": MagicMock(data=[{"id": "b1"}])})
    service._revoke_offboarded_member_access(db, ORG, OTHER, "suspended")
    assert "board_members" not in [c.args[0] for c in db.table.call_args_list]


def test_offboard_cleanup_never_raises(monkeypatch):
    monkeypatch.setattr("orgs.projects.revoke_org_granted_memberships", lambda *a, **k: None)
    db = MagicMock()
    db.table.side_effect = RuntimeError("boom")
    service._revoke_offboarded_member_access(db, ORG, OTHER, "removed")  # no exception


def test_revert_org_boards_uses_creator_when_still_seated():
    """Assert the PAYLOAD, not just which tables were touched: replacing the
    whole recipient rule with `recipient = fallback_user` leaves a
    tables-touched assertion green, and every dissolved team's boards would
    silently land on the dissolving admin instead of their creators."""
    written = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "boards":
            b.execute.return_value = MagicMock(data=[{"id": "b1", "owner_id": ME}])

            def _capture_update(payload, **kwargs):
                written["boards"] = payload
                return b

            b.update = _capture_update
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"user_id": ME}])  # ME still active
        return b

    db = MagicMock()
    db.table.side_effect = _side
    service._revert_org_boards(db, ORG, fallback_user=OTHER)
    assert written["boards"]["owner_id"] == ME  # the CREATOR, not the fallback
    assert written["boards"]["team_id"] is None and written["boards"]["restricted"] is False
    # artist_id cleared so the detach cannot collide with uq_boards_personal_artist
    assert written["boards"]["artist_id"] is None


def test_revert_org_boards_falls_back_when_creator_offboarded():
    """A board created by someone since removed must land on the dissolving
    admin — otherwise nobody can reach it once the org is archived."""
    written = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "boards":
            b.execute.return_value = MagicMock(data=[{"id": "b1", "owner_id": "u-gone"}])

            # A real function, NOT `lambda ...: written.setdefault(...) or b` —
            # setdefault returns the (truthy) payload, so `or b` short-circuits
            # and the chain's next .eq() would hit a dict.
            def _capture_update(payload, **kwargs):
                written["boards"] = payload
                return b

            b.update = _capture_update
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"user_id": ME}])  # u-gone is NOT active
        return b

    db = MagicMock()
    db.table.side_effect = _side
    service._revert_org_boards(db, ORG, fallback_user=OTHER)
    assert written["boards"]["owner_id"] == OTHER
    assert written["boards"]["team_id"] is None and written["boards"]["restricted"] is False
    assert written["boards"]["artist_id"] is None
