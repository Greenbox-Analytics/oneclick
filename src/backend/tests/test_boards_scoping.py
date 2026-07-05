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
