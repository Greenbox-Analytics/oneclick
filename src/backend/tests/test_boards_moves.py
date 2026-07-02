"""Tests for cross-board / subtree move rules on drag-and-drop reorder (spec §7.3).

batch_reorder gates require_board_edit on both the source and destination board when a
reorder's target column lives on a different board, scopes the write by task id (not
user_id), rejects cross-team (and cross-owner personal↔personal) moves, and rejects
moving a lone subtask to another board on its own — a subtask's board always follows
its parent (moved via _apply_cross_board_move, defined in Task 4 and reused here).
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from boards import service

USER = "00000000-0000-0000-0000-000000000001"
TASK = "task-0000-0000-0000-0000-000000000001"
SUBTASK = "task-0000-0000-0000-0000-000000000002"
PARENT = "prnt-0000-0000-0000-0000-000000000001"
COLUMN = "col-00000000-0000-0000-0000-000000000001"
BOARD_A = "b0000000-0000-0000-0000-00000000000a"
BOARD_B = "b0000000-0000-0000-0000-00000000000b"
TEAM_1 = "team0000-0000-0000-0000-000000000001"
TEAM_2 = "team0000-0000-0000-0000-000000000002"


def _mv_db(task_row: dict, col_board_id: str, boards: list[dict]) -> MagicMock:
    """Mock supabase client for batch_reorder / _apply_cross_board_move:

    - board_tasks: reads return [task_row]; the trailing update/eq/execute chain on the
      same builder absorbs both the final task update and the subtree (children) update.
    - board_columns: reads return [{"board_id": col_board_id}].
    - boards: reads return `boards` (id, team_id, owner_id rows), used by
      _apply_cross_board_move's cross-team/owner check.
    """

    def _side(name):
        b = MagicMock()
        if name == "board_tasks":
            b.select.return_value = b
            b.update.return_value = b
            b.eq.return_value = b
            b.limit.return_value = b
            b.execute.return_value = MagicMock(data=[task_row])
        elif name == "board_columns":
            b.select.return_value = b
            b.eq.return_value = b
            b.limit.return_value = b
            b.execute.return_value = MagicMock(data=[{"board_id": col_board_id}])
        elif name == "boards":
            b.select.return_value = b
            b.in_.return_value = b
            b.execute.return_value = MagicMock(data=boards)
        else:
            b.execute.return_value = MagicMock(data=[])
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


async def test_move_across_teams_rejected(monkeypatch):
    """Task lives on a team-A board; the drop target column lives on a team-B board -> ValueError."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    task_row = {"id": TASK, "board_id": BOARD_A, "parent_task_id": None}
    boards = [
        {"id": BOARD_A, "team_id": TEAM_1, "owner_id": None},
        {"id": BOARD_B, "team_id": TEAM_2, "owner_id": None},
    ]
    db = _mv_db(task_row, BOARD_B, boards)
    reorders = [{"task_id": TASK, "target_column_id": COLUMN, "position": 0}]

    with pytest.raises(ValueError):
        await service.batch_reorder(db, USER, reorders)


async def test_same_board_move_ok(monkeypatch):
    """Target column is on the same board -> no cross-board validation, just a gated update."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    task_row = {"id": TASK, "board_id": BOARD_A, "parent_task_id": None}
    boards = [{"id": BOARD_A, "team_id": None, "owner_id": USER}]
    db = _mv_db(task_row, BOARD_A, boards)
    reorders = [{"task_id": TASK, "target_column_id": COLUMN, "position": 2}]

    result = await service.batch_reorder(db, USER, reorders)

    assert result is True


async def test_non_member_rejected(monkeypatch):
    """require_board_edit rejects the caller (not a board member) -> HTTPException(404) propagates."""

    def _deny(db, u, b):
        raise HTTPException(status_code=404, detail="Board not found")

    monkeypatch.setattr(service.authz, "require_board_edit", _deny)
    task_row = {"id": TASK, "board_id": BOARD_A, "parent_task_id": None}
    boards = [{"id": BOARD_A, "team_id": None, "owner_id": "someone-else"}]
    db = _mv_db(task_row, BOARD_A, boards)
    reorders = [{"task_id": TASK, "target_column_id": COLUMN, "position": 0}]

    with pytest.raises(HTTPException) as exc_info:
        await service.batch_reorder(db, USER, reorders)
    assert exc_info.value.status_code == 404


async def test_lone_subtask_cross_board_rejected(monkeypatch):
    """A subtask (parent_task_id set) can't be moved to another board on its own -> ValueError."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    task_row = {"id": SUBTASK, "board_id": BOARD_A, "parent_task_id": PARENT}
    boards = [
        {"id": BOARD_A, "team_id": TEAM_1, "owner_id": None},
        {"id": BOARD_B, "team_id": TEAM_1, "owner_id": None},
    ]
    db = _mv_db(task_row, BOARD_B, boards)
    reorders = [{"task_id": SUBTASK, "target_column_id": COLUMN, "position": 0}]

    with pytest.raises(ValueError):
        await service.batch_reorder(db, USER, reorders)
