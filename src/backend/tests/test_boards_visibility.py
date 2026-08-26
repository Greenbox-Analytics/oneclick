"""Boards on Teams — visibility narrowing + update_board (spec 2026-08-16 §1/§3)."""

from unittest.mock import MagicMock

import pytest

from boards import service
from tests.conftest import MockQueryBuilder

ADMIN, MEMBER = "u-admin", "u-member"
ORG = "org-1"
B_OPEN = {"id": "b-open", "team_id": ORG, "owner_id": ADMIN, "archived": False, "restricted": False}
B_RESTR = {"id": "b-restr", "team_id": ORG, "owner_id": ADMIN, "archived": False, "restricted": True}


# One org_members mock serves EVERY read of that table in a call: live_org_ids wants
# `org_id`, the admin narrowing wants `role`/`status`, update_board's validation wants
# `user_id`. A plain seat (no role) is therefore a live, NON-admin member.
SEAT = MagicMock(data=[{"org_id": ORG, "user_id": MEMBER}])
ADMIN_SEAT = MagicMock(data=[{"org_id": ORG, "user_id": MEMBER, "role": "admin", "status": "active"}])


def _db(tbl: dict, *, admin=False):
    """tbl: table name -> MagicMock(data=...) returned for EVERY read of that table
    (the service reads org_members / boards more than once per call, so each mock
    must be valid for all of them — see SEAT)."""

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
    db.rpc.return_value.execute.return_value = MagicMock(data=admin)
    return db


async def test_list_boards_hides_restricted_from_unlisted_member():
    db = _db(
        {
            "org_members": SEAT,  # live_org_ids: seat (and NOT an admin seat)
            "organizations": MagicMock(data=[{"id": ORG}]),  # live_org_ids: live
            "boards": MagicMock(data=[B_OPEN, B_RESTR]),
            "board_members": MagicMock(data=[]),  # caller listed on nothing / member ids
        }
    )
    rows = await service.list_boards(db, MEMBER, team_id=ORG)
    assert [b["id"] for b in rows] == ["b-open"]
    assert rows[0]["restricted"] is False and rows[0]["member_user_ids"] == []


async def test_list_boards_shows_restricted_to_listed_member_and_admin():
    listed = _db(
        {
            "org_members": SEAT,
            "organizations": MagicMock(data=[{"id": ORG}]),
            "boards": MagicMock(data=[B_RESTR]),
            "board_members": MagicMock(data=[{"board_id": "b-restr", "user_id": MEMBER}]),
        }
    )
    assert [b["id"] for b in await service.list_boards(listed, MEMBER, team_id=ORG)] == ["b-restr"]
    admin = _db(
        {
            "org_members": ADMIN_SEAT,  # active ADMIN seat → sees every board of the org
            "organizations": MagicMock(data=[{"id": ORG}]),
            "boards": MagicMock(data=[B_RESTR]),
            "board_members": MagicMock(data=[]),
        }
    )
    assert [b["id"] for b in await service.list_boards(admin, MEMBER, team_id=ORG)] == ["b-restr"]


async def test_list_boards_non_member_denied():
    db = _db({"org_members": MagicMock(data=[]), "organizations": MagicMock(data=[])})
    with pytest.raises(PermissionError):
        await service.list_boards(db, MEMBER, team_id=ORG)


async def test_update_board_visibility_needs_admin_or_owner():
    db = _db(
        {
            "boards": MagicMock(data=[B_OPEN]),
            "org_members": SEAT,
            "organizations": MagicMock(data=[{"id": ORG}]),
            "board_members": MagicMock(data=[]),
        }
    )
    with pytest.raises(PermissionError):
        await service.update_board(db, MEMBER, "b-open", {"restricted": True})


async def test_update_board_rejects_foreign_member_ids():
    db = _db(
        {
            "boards": MagicMock(data=[B_OPEN]),
            "org_members": SEAT,  # active ids = {MEMBER}, so "u-stranger" is foreign
            "organizations": MagicMock(data=[{"id": ORG}]),
            "board_members": MagicMock(data=[]),
        }
    )
    with pytest.raises(service.InvalidBoardMembersError):
        await service.update_board(db, ADMIN, "b-open", {"restricted": True, "member_user_ids": [MEMBER, "u-stranger"]})


async def test_update_board_keeps_an_already_listed_member_whose_seat_went_inactive():
    """Suspend does not purge board_members (reversible), and the roster is
    ACTIVE-only — so re-saving a board that still lists a suspended person must
    NOT 422, or a pure rename becomes impossible until they are dropped."""
    db = _db(
        {
            "boards": MagicMock(data=[B_OPEN]),
            # ADMIN's own seat only: the caller is live, MEMBER is NOT an active seat.
            "org_members": MagicMock(data=[{"org_id": ORG, "user_id": ADMIN}]),
            "organizations": MagicMock(data=[{"id": ORG}]),
            "board_members": MagicMock(data=[{"user_id": MEMBER}]),  # ...but is already on the board
        }
    )
    out = await service.update_board(db, ADMIN, "b-open", {"restricted": True, "member_user_ids": [MEMBER]})
    assert out["member_user_ids"] == [MEMBER]


async def test_update_board_still_rejects_a_stranger_not_already_listed():
    db = _db(
        {
            "boards": MagicMock(data=[B_OPEN]),
            "org_members": MagicMock(data=[{"org_id": ORG, "user_id": ADMIN}]),
            "organizations": MagicMock(data=[{"id": ORG}]),
            "board_members": MagicMock(data=[]),  # not already on the board either
        }
    )
    with pytest.raises(service.InvalidBoardMembersError):
        await service.update_board(db, ADMIN, "b-open", {"restricted": True, "member_user_ids": ["u-stranger"]})


async def test_update_board_replaces_member_set():
    db = _db(
        {
            "boards": MagicMock(data=[B_OPEN]),
            "org_members": SEAT,
            "organizations": MagicMock(data=[{"id": ORG}]),
            "board_members": MagicMock(data=[{"user_id": MEMBER}]),
        }
    )
    out = await service.update_board(db, ADMIN, "b-open", {"restricted": True, "member_user_ids": [MEMBER]})
    assert out["member_user_ids"] == [MEMBER]
    # REPLACE, not append: assert on the delete CALL, not on a call count.
    # A count assertion stays green if the delete is removed entirely (insert +
    # final read = 2 calls), turning replace-set into append-set — a removed
    # person would keep their access. And because MockQueryBuilder.eq() discards
    # its args, the scoping has to be asserted explicitly or dropping
    # .eq("board_id", ...) — a global wipe of EVERY board's membership — is
    # also green. conftest's docstring recommends exactly this pattern.
    builder = db.table("board_members")
    builder.delete.assert_called()
    builder.delete.return_value.eq.assert_called_with("board_id", "b-open")


async def test_update_board_personal_cannot_be_restricted():
    personal = {"id": "b-p", "team_id": None, "owner_id": MEMBER, "archived": False, "restricted": False}
    db = _db({"boards": MagicMock(data=[personal])})
    with pytest.raises(service.InvalidBoardMembersError):
        await service.update_board(db, MEMBER, "b-p", {"restricted": True})
