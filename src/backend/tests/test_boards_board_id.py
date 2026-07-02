"""ensure_personal_board + gated board_id resolution on insert paths."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from boards import service
from tests.conftest import MockQueryBuilder

USER = "00000000-0000-0000-0000-000000000001"
ARTIST = "a0000000-0000-0000-0000-000000000001"
BOARD = "b0000000-0000-0000-0000-000000000001"


def _db(seqs):
    counters = {k: 0 for k in seqs}

    def _side(name):
        b = MockQueryBuilder()
        if name in seqs:
            i = min(counters[name], len(seqs[name]) - 1)
            counters[name] += 1
            b.execute.return_value = seqs[name][i]
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


def test_ensure_personal_board_returns_existing():
    db = _db({"boards": [MagicMock(data=[{"id": BOARD}], count=1)]})
    assert service.ensure_personal_board(db, USER, None) == BOARD


def test_ensure_personal_board_creates_when_missing():
    db = _db(
        {
            "boards": [MagicMock(data=[], count=0), MagicMock(data=[{"id": BOARD}], count=1)],
            "artists": [MagicMock(data=[{"name": "Jane"}], count=1)],
        }
    )
    assert service.ensure_personal_board(db, USER, ARTIST) == BOARD


async def test_create_task_gates_on_resolved_column_board(monkeypatch):
    # a foreign column resolves to a board the caller can't edit → 404 from require_board_edit
    monkeypatch.setattr(
        service.authz, "require_board_edit", lambda db, u, bid: (_ for _ in ()).throw(HTTPException(status_code=404))
    )
    db = _db({"board_columns": [MagicMock(data=[{"board_id": BOARD}], count=1)]})
    with pytest.raises(HTTPException):
        await service.create_task(db, USER, {"title": "x", "column_id": "col1"})


async def test_create_column_sets_and_gates_board(monkeypatch):
    monkeypatch.setattr(service, "ensure_personal_board", lambda db, u, a: BOARD)
    gated = {}
    monkeypatch.setattr(service.authz, "require_board_edit", lambda db, u, bid: gated.update(board=bid))
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "board_columns":

            def _insert(p):
                captured.update(p)
                return b

            b.insert.side_effect = _insert
            b.execute.return_value = MagicMock(data=[{"id": "c1", **captured}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    await service.create_column(db, USER, {"title": "To Do", "artist_id": ARTIST})
    assert captured["board_id"] == BOARD and gated["board"] == BOARD
