"""Board hard-delete / restore / archived-listing (spec 2026-07-03)."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from boards import service
from tests.conftest import MockQueryBuilder

USER = "00000000-0000-0000-0000-000000000001"
BOARD = "b0000000-0000-0000-0000-000000000001"


def _db(tbl):
    def _side(name):
        b = MockQueryBuilder()
        if name in tbl:
            b.execute.return_value = tbl[name]
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


async def test_delete_board_ok_when_name_matches(monkeypatch):
    # NOTE: do NOT monkeypatch authz.get_board — delete_board reads the name via a DIRECT
    # `boards` select, so the mock's `name` must flow through the real path (guards the
    # blocker where authz.get_board omits `name`).
    monkeypatch.setattr(service, "_can_archive_board", lambda db, u, board: True)
    db = _db(
        {
            "boards": MagicMock(data=[{"id": BOARD, "name": "My Board", "team_id": None, "owner_id": USER}]),
            "board_tasks": MagicMock(data=[], count=3),
        }
    )
    result = await service.delete_board(db, USER, BOARD, "My Board")
    assert result == {"deleted": BOARD, "tasks": 3}


async def test_delete_board_rejects_wrong_name(monkeypatch):
    monkeypatch.setattr(service, "_can_archive_board", lambda db, u, board: True)
    db = _db({"boards": MagicMock(data=[{"id": BOARD, "name": "My Board", "team_id": None, "owner_id": USER}])})
    with pytest.raises(service.ConfirmationError):
        await service.delete_board(db, USER, BOARD, "wrong name")


async def test_delete_board_trims_and_normalizes(monkeypatch):
    monkeypatch.setattr(service, "_can_archive_board", lambda db, u, board: True)
    db = _db(
        {
            "boards": MagicMock(data=[{"id": BOARD, "name": "My Board", "team_id": None, "owner_id": USER}]),
            "board_tasks": MagicMock(data=[], count=0),
        }
    )
    result = await service.delete_board(db, USER, BOARD, "  My Board  ")  # surrounding whitespace tolerated
    assert result["deleted"] == BOARD


async def test_delete_board_denied_for_non_admin(monkeypatch):
    monkeypatch.setattr(service, "_can_archive_board", lambda db, u, board: False)
    db = _db({"boards": MagicMock(data=[{"id": BOARD, "name": "My Board", "team_id": "t1", "owner_id": "someone"}])})
    with pytest.raises(PermissionError):
        await service.delete_board(db, USER, BOARD, "My Board")


async def test_restore_board_unarchives(monkeypatch):
    # restore_board uses authz.get_board (name not needed) + _can_archive_board.
    monkeypatch.setattr(service, "_can_archive_board", lambda db, u, board: True)
    monkeypatch.setattr(service.authz, "get_board", lambda db, b: {"id": BOARD, "team_id": None, "owner_id": USER})
    db = _db({"boards": MagicMock(data=[{"id": BOARD, "archived": False}])})
    result = await service.restore_board(db, USER, BOARD)
    assert result == {"restored": BOARD}


# --- Real _can_archive_board gate (NOT monkeypatched) ---------------------------


async def test_delete_board_denied_when_personal_board_owned_by_other():
    # Real _can_archive_board: personal board → owner must match caller.
    db = _db(
        {"boards": MagicMock(data=[{"id": BOARD, "name": "My Board", "team_id": None, "owner_id": "someone-else"}])}
    )
    with pytest.raises(PermissionError):
        await service.delete_board(db, USER, BOARD, "My Board")


def _org_db(tbl: dict, *, admin: bool):
    """_db plus a live seat in org 't1' and an is_org_admin rpc answer — what
    authz.is_board_admin consults for a team board."""
    db = _db({"org_members": MagicMock(data=[{"org_id": "t1", "user_id": USER}]), **tbl})
    db.rpc.return_value.execute.return_value = MagicMock(data=admin)
    return db


async def test_delete_board_denied_for_team_board_when_not_admin():
    # Real _can_archive_board: team board → live seat but not an org admin.
    db = _org_db(
        {
            "organizations": MagicMock(data=[{"id": "t1"}]),
            "boards": MagicMock(data=[{"id": BOARD, "name": "My Board", "team_id": "t1", "owner_id": "x"}]),
        },
        admin=False,
    )
    with pytest.raises(PermissionError):
        await service.delete_board(db, USER, BOARD, "My Board")


async def test_delete_board_ok_for_team_board_when_admin():
    # Real _can_archive_board: team board → live org admin → delete proceeds.
    db = _org_db(
        {
            "organizations": MagicMock(data=[{"id": "t1"}]),
            "boards": MagicMock(data=[{"id": BOARD, "name": "My Board", "team_id": "t1", "owner_id": "x"}]),
            "board_tasks": MagicMock(data=[], count=7),
        },
        admin=True,
    )
    result = await service.delete_board(db, USER, BOARD, "My Board")
    assert result == {"deleted": BOARD, "tasks": 7}


# --- list_archived_boards --------------------------------------------------------


async def test_list_archived_boards_denied_for_non_admin_team():
    """authz.require_org_admin raises HTTPException(403) rather than PermissionError —
    the router lets it through unchanged, so the caller still sees 403."""
    db = _org_db({"organizations": MagicMock(data=[{"id": "t1"}])}, admin=False)
    with pytest.raises(HTTPException) as exc:
        await service.list_archived_boards(db, USER, team_id="t1")
    assert exc.value.status_code == 403


async def test_list_archived_boards_team_admin_attaches_task_count():
    db = _org_db(
        {
            "organizations": MagicMock(data=[{"id": "t1"}]),
            "boards": MagicMock(
                data=[{"id": "ab1", "name": "Arch", "archived": True, "team_id": "t1", "owner_id": USER}]
            ),
            "board_members": MagicMock(data=[]),
            "board_tasks": MagicMock(data=[], count=4),
        },
        admin=True,
    )
    result = await service.list_archived_boards(db, USER, team_id="t1")
    assert len(result) == 1
    assert result[0]["id"] == "ab1"
    assert result[0]["task_count"] == 4


async def test_list_archived_boards_personal_branch_scoped_to_caller():
    # team_id=None → personal branch (query filters owner_id == user_id + team_id IS NULL).
    db = _db(
        {
            "boards": MagicMock(data=[{"id": "pb1", "name": "Personal Arch", "archived": True, "owner_id": USER}]),
            "board_tasks": MagicMock(data=[], count=2),
        }
    )
    result = await service.list_archived_boards(db, USER, team_id=None)
    assert len(result) == 1
    assert result[0]["id"] == "pb1"
    assert result[0]["task_count"] == 2


# --- Not-found paths (locks router 404 mapping) ---------------------------------


async def test_delete_board_not_found(monkeypatch):
    monkeypatch.setattr(service, "_can_archive_board", lambda db, u, board: True)
    db = _db({"boards": MagicMock(data=[])})
    with pytest.raises(ValueError):
        await service.delete_board(db, USER, BOARD, "My Board")


async def test_restore_board_not_found(monkeypatch):
    monkeypatch.setattr(service.authz, "get_board", lambda db, b: None)
    db = _db({})
    with pytest.raises(ValueError):
        await service.restore_board(db, USER, BOARD)
