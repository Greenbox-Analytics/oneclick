"""Board LISTS narrowed to the workspace — the half test_boards_workspace.py
doesn't cover (that file pins _personal_board_ids / ensure_personal_board /
create_board; this one pins the calendar composition and the two list_boards
paths).

The workspace view of boards is a union of two different edges:

    boards.org_id   filing label — which workspace a PERSONAL board shows in
    boards.team_id  sharing grant — an org-owned board, one org's seats see it

An org workspace shows personally-filed boards under that org PLUS that org's
shared boards; Personal shows personally-filed boards only; unscoped is the
pre-scoping union of everything.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from artist_access import Scope
from boards import service
from tests.conftest import TEST_USER_ID, MockQueryBuilder

ORG_A = "20000000-0000-0000-0000-00000000000a"
ORG_B = "20000000-0000-0000-0000-00000000000b"
MY_BOARD = "bbbbbbbb-0000-0000-0000-000000000001"
ORG_BOARD = "bbbbbbbb-0000-0000-0000-000000000002"


class _Recorder:
    """Routes by (table, select-columns); captures the filters built on
    `boards` so the tests pin query SHAPE, as everywhere else in this suite."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.requested: list = []
        self.board_filters: list = []

    def __call__(self, name):
        rec = self

        class _B(MockQueryBuilder):
            def select(self, *args, **kwargs):
                sig = args[0] if args else "*"
                rec.requested.append((name, sig))
                rows = rec.mapping.get((name, sig), [])
                self.execute.return_value = MagicMock(data=rows, count=len(rows))
                return self

        b = _B()
        if name == "boards":
            for method in ("eq", "in_", "is_"):
                original = getattr(b, method)

                def _capture(*args, _o=original, _m=method, **kwargs):
                    rec.board_filters.append((_m, args))
                    return _o(*args, **kwargs)

                setattr(b, method, _capture)
        return b


def _db(rec):
    db = MagicMock()
    db.table.side_effect = rec
    return db


def _seats_and_orgs(org_ids):
    return {
        ("org_members", "org_id"): [{"org_id": o} for o in org_ids],
        ("organizations", "id"): [{"id": o} for o in org_ids],
    }


def _org_board(team_id):
    return {"id": ORG_BOARD, "owner_id": "someone-else", "team_id": team_id, "restricted": False}


# ---------------------------------------------------------------------------
# _calendar_board_ids
# ---------------------------------------------------------------------------


def test_calendar_unscoped_is_the_full_union():
    rec = _Recorder(
        {
            ("boards", "id"): [{"id": MY_BOARD}],
            ("boards", "*"): [_org_board(ORG_A)],
            ("board_members", "board_id, user_id"): [],
            ("org_members", "org_id, role, status"): [],
            **_seats_and_orgs([ORG_A]),
        }
    )
    ids = service._calendar_board_ids(_db(rec), TEST_USER_ID)

    assert set(ids) == {MY_BOARD, ORG_BOARD}


def test_calendar_personal_scope_is_personally_filed_only():
    """No team boards at all — and no org lookups, so the narrowing can't
    accidentally re-widen through _team_board_ids."""
    rec = _Recorder(
        {
            ("boards", "id"): [{"id": MY_BOARD}],
            ("boards", "*"): [_org_board(ORG_A)],  # must never be reached
            **_seats_and_orgs([ORG_A]),
        }
    )
    ids = service._calendar_board_ids(_db(rec), TEST_USER_ID, Scope.personal())

    assert ids == [MY_BOARD]
    assert ("is_", ("org_id", "null")) in rec.board_filters
    assert ("boards", "*") not in rec.requested


def test_calendar_org_scope_is_that_orgs_boards_only():
    """Filed-under-org + THAT org's shared boards. ORG_B's boards stay out even
    though the caller holds a seat there — the whole point of the workspace."""
    rec = _Recorder(
        {
            ("boards", "id"): [{"id": MY_BOARD}],
            ("boards", "*"): [_org_board(ORG_A)],
            ("board_members", "board_id, user_id"): [],
            ("org_members", "org_id, role, status"): [],
            **_seats_and_orgs([ORG_A, ORG_B]),
        }
    )
    ids = service._calendar_board_ids(_db(rec), TEST_USER_ID, Scope.org(ORG_A))

    assert set(ids) == {MY_BOARD, ORG_BOARD}
    assert ("eq", ("org_id", ORG_A)) in rec.board_filters
    # The shared-board half was narrowed to the ONE org, not every seat.
    assert ("in_", ("team_id", [ORG_A])) in rec.board_filters


def test_calendar_org_scope_falls_closed_for_a_dead_org():
    """An archived/lapsed org contributes no live id, so its shared boards are
    never queried — same fall-closed posture as scoped_artists."""
    rec = _Recorder(
        {
            ("boards", "id"): [{"id": MY_BOARD}],
            ("boards", "*"): [_org_board(ORG_A)],  # must never be reached
            **_seats_and_orgs([]),  # seat gone / org dead
        }
    )
    ids = service._calendar_board_ids(_db(rec), TEST_USER_ID, Scope.org(ORG_A))

    assert ids == [MY_BOARD]
    assert ("boards", "*") not in rec.requested


# ---------------------------------------------------------------------------
# list_boards / list_archived_boards — the personal branch
# ---------------------------------------------------------------------------


def _run_list(rec, **kwargs):
    return asyncio.run(service.list_boards(_db(rec), TEST_USER_ID, **kwargs))


def test_list_boards_personal_scope_filters_the_filing_label():
    rec = _Recorder({("boards", "*"): [{"id": MY_BOARD}]})
    _run_list(rec, scope=Scope.personal())

    assert ("is_", ("org_id", "null")) in rec.board_filters


def test_list_boards_org_scope_filters_to_that_org():
    rec = _Recorder({("boards", "*"): [{"id": MY_BOARD}]})
    _run_list(rec, scope=Scope.org(ORG_A))

    assert ("eq", ("org_id", ORG_A)) in rec.board_filters


def test_list_boards_unscoped_does_not_filter():
    rec = _Recorder({("boards", "*"): [{"id": MY_BOARD}]})
    _run_list(rec, scope=Scope.unscoped())

    assert not [f for f in rec.board_filters if f[1] and f[1][0] == "org_id"]


def test_list_boards_explicit_team_id_ignores_the_scope():
    """An explicit team listing is an authorization-checked request for ONE
    org's shared boards — the filing label plays no part in it."""
    rec = _Recorder(
        {
            ("boards", "*"): [_org_board(ORG_A)],
            ("board_members", "board_id, user_id"): [],
            ("org_members", "org_id, role, status"): [],
            **_seats_and_orgs([ORG_A]),
        }
    )
    boards = _run_list(rec, team_id=ORG_A, scope=Scope.personal())

    assert [b["id"] for b in boards] == [ORG_BOARD]
    assert not [f for f in rec.board_filters if f[1] and f[1][0] == "org_id"]


def test_list_boards_foreign_team_id_still_denied():
    rec = _Recorder(_seats_and_orgs([ORG_A]))
    with pytest.raises(PermissionError):
        _run_list(rec, team_id=ORG_B, scope=Scope.org(ORG_B))


def test_list_archived_boards_personal_scope_filters_the_filing_label():
    rec = _Recorder({("boards", "*"): [], ("board_tasks", "id"): []})
    asyncio.run(service.list_archived_boards(_db(rec), TEST_USER_ID, scope=Scope.personal()))

    assert ("is_", ("org_id", "null")) in rec.board_filters
    assert ("eq", ("archived", True)) in rec.board_filters
