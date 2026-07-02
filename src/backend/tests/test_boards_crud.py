from unittest.mock import MagicMock

import pytest

from boards import service
from tests.conftest import MockQueryBuilder

USER = "00000000-0000-0000-0000-000000000001"
TEAM = "10000000-0000-0000-0000-000000000001"
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


async def test_create_team_board_requires_membership(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_member", lambda *a: False)
    with pytest.raises(PermissionError):
        await service.create_board(MagicMock(), USER, name="T", team_id=TEAM)


async def test_list_personal_boards_active_only():
    db = _db({"boards": MagicMock(data=[{"id": BOARD, "team_id": None, "archived": False}], count=1)})
    assert (await service.list_boards(db, USER, team_id=None))[0]["id"] == BOARD


async def test_archive_team_board_requires_admin(monkeypatch):
    board = {"id": BOARD, "team_id": TEAM, "owner_id": USER}
    monkeypatch.setattr(service.authz, "get_board", lambda db, b: board)
    monkeypatch.setattr(service.authz, "is_team_admin", lambda *a: False)
    with pytest.raises(PermissionError):
        await service.archive_board(MagicMock(), USER, BOARD)
