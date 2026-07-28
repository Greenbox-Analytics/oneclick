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


async def test_get_columns_gates_explicit_board(monkeypatch):
    monkeypatch.setattr(
        service.authz,
        "require_board_access",
        lambda db, u, b: (_ for _ in ()).throw(HTTPException(status_code=404)),
    )
    with pytest.raises(HTTPException):
        await service.get_columns(_db({}), USER, board_id=BOARD)


async def test_get_task_detail_gates_on_task_board(monkeypatch):
    monkeypatch.setattr(
        service.authz,
        "require_board_access",
        lambda db, u, b: (_ for _ in ()).throw(HTTPException(status_code=404)),
    )
    db = _db({"board_tasks": MagicMock(data=[{"id": "t1", "board_id": BOARD, "is_parent": False}], count=1)})
    with pytest.raises(HTTPException):
        await service.get_task_detail(db, USER, "t1")


def _queue_db(tbl: dict):
    """Like _db, but each table serves a QUEUE of responses — one per call, in order."""

    def _side(name):
        b = MockQueryBuilder()
        queue = tbl.get(name)
        if queue:
            b.execute.return_value = MagicMock(data=queue.pop(0), count=None)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


async def test_calendar_spans_team_boards_and_stamps_team():
    """The unfiltered calendar is NOT personal-only: it also pulls the boards of every
    team the caller belongs to, and stamps each task with its team so the UI can colour
    and legend by team."""
    team_board = "b0000000-0000-0000-0000-0000000000t1"
    db = _queue_db(
        {
            "boards": [
                [{"id": BOARD}],  # _personal_board_ids
                [{"id": team_board}],  # team boards
                [{"id": team_board, "team_id": "team-1"}],  # _stamp_team_context
            ],
            "team_members": [[{"team_id": "team-1"}]],
            "teams": [
                [{"id": "team-1"}],  # non-archived filter
                [{"id": "team-1", "name": "Team One"}],  # name lookup
            ],
            "board_tasks": [[{"id": "t1", "board_id": team_board, "user_id": USER}]],
        }
    )

    tasks = await service.get_tasks_by_date_range(db, USER, "2026-07-01", "2026-07-31")

    assert [t["id"] for t in tasks] == ["t1"]
    assert tasks[0]["team_id"] == "team-1"
    assert tasks[0]["team_name"] == "Team One"


async def test_calendar_stamps_personal_tasks_with_no_team():
    """A personal-board task carries team_id/team_name = None (legend shows 'Personal')."""
    db = _queue_db(
        {
            "boards": [
                [{"id": BOARD}],  # _personal_board_ids
                [{"id": BOARD, "team_id": None}],  # _stamp_team_context
            ],
            "team_members": [[]],  # no team memberships
            "board_tasks": [[{"id": "t1", "board_id": BOARD, "user_id": USER}]],
        }
    )

    tasks = await service.get_tasks_by_date_range(db, USER, "2026-07-01", "2026-07-31")

    assert tasks[0]["team_id"] is None
    assert tasks[0]["team_name"] is None
