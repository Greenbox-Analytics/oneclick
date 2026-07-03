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


async def test_subtask_auto_column_is_board_scoped(monkeypatch):
    """A subtask created with an explicit board_id (e.g. a team board) and no column_id must
    pick its auto-column FROM THAT BOARD — not from the caller's personal columns."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    eq_calls = []

    class _RecordingBuilder(MockQueryBuilder):
        def eq(self, *args, **kwargs):
            eq_calls.append(args)
            return self

    col_builder = _RecordingBuilder()
    col_builder.execute.return_value = MagicMock(data=[{"id": "col-team-1"}])
    task_builder = MockQueryBuilder()
    task_builder.execute.return_value = MagicMock(data=[{"id": "task-1", "board_id": BOARD, "column_id": "col-team-1"}])

    def _side(name):
        if name == "board_columns":
            return col_builder
        if name == "board_tasks":
            return task_builder
        return MockQueryBuilder()

    db = MagicMock()
    db.table.side_effect = _side

    task = await service.create_task(db, USER, {"title": "sub", "parent_task_id": "parent-1", "board_id": BOARD})

    assert ("board_id", BOARD) in eq_calls, "column pick must filter by the task's board"
    assert all(c[0] != "user_id" for c in eq_calls), "must not fall back to the user-scoped pick"
    assert task["column_id"] == "col-team-1"
