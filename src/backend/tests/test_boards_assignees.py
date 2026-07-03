"""Tests for board task assignees (spec §5.6).

add_assignee/remove_assignee resolve the task's board_id, gate via
authz.require_board_edit, and (for add) validate the target via
authz.can_assign_user — personal board -> owner only, team board -> member only.
"""

from unittest.mock import MagicMock

import pytest

from boards import service
from tests.conftest import MockQueryBuilder

USER = "00000000-0000-0000-0000-000000000001"
TARGET = "00000000-0000-0000-0000-000000000002"
BOARD = "b0000000-0000-0000-0000-000000000001"
TASK = "task-0000-0000-0000-0000-000000000001"


def _db(tbl):
    def _side(name):
        b = MockQueryBuilder()
        if name in tbl:
            b.execute.return_value = tbl[name]
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


async def test_add_assignee_rejected_when_not_assignable(monkeypatch):
    """§5.6: require_board_edit passes but can_assign_user rejects the target -> PermissionError."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    monkeypatch.setattr(service.authz, "can_assign_user", lambda db, target, b: False)
    db = _db({"board_tasks": MagicMock(data=[{"board_id": BOARD}], count=1)})

    with pytest.raises(PermissionError):
        await service.add_assignee(db, USER, TASK, TARGET)


async def test_add_assignee_ok(monkeypatch):
    """When both authz checks pass, the assignment succeeds and echoes task/user ids."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    monkeypatch.setattr(service.authz, "can_assign_user", lambda db, target, b: True)
    db = _db({"board_tasks": MagicMock(data=[{"board_id": BOARD}], count=1)})

    result = await service.add_assignee(db, USER, TASK, TARGET)

    assert result == {"task_id": TASK, "user_id": TARGET}


async def test_add_assignee_task_not_found_raises_value_error(monkeypatch):
    """No task row (no board_id resolvable) -> ValueError('Task not found')."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    monkeypatch.setattr(service.authz, "can_assign_user", lambda db, target, b: True)
    db = _db({"board_tasks": MagicMock(data=[], count=0)})

    with pytest.raises(ValueError):
        await service.add_assignee(db, USER, TASK, TARGET)


async def test_remove_assignee_ok(monkeypatch):
    """Happy path: require_board_edit passes, the assignee row is removed."""
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, b: None)
    db = _db({"board_tasks": MagicMock(data=[{"board_id": BOARD}], count=1)})

    result = await service.remove_assignee(db, USER, TASK, TARGET)

    assert result == {"unassigned": TARGET}


def test_enrich_tasks_attaches_assignees_batched():
    """_enrich_tasks attaches assignees to list-path tasks (one board_task_assignees
    query for all task_ids, one profiles query for all users) — no N+1, [] when none."""
    t1 = "task-0000-0000-0000-0000-000000000001"
    t2 = "task-0000-0000-0000-0000-000000000002"
    db = _db(
        {
            "board_task_artists": MagicMock(data=[]),
            "board_task_projects": MagicMock(data=[]),
            "board_task_contracts": MagicMock(data=[]),
            "board_task_assignees": MagicMock(data=[{"task_id": t1, "user_id": TARGET}]),
            "profiles": MagicMock(data=[{"id": TARGET, "full_name": "Jane", "avatar_url": None}]),
        }
    )

    out = service._enrich_tasks(db, [{"id": t1}, {"id": t2}])

    by_id = {t["id"]: t for t in out}
    assert by_id[t1]["assignees"] == [{"user_id": TARGET, "full_name": "Jane", "avatar_url": None}]
    assert by_id[t2]["assignees"] == []
