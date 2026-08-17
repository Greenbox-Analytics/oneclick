"""boards/authz — access predicates on ORG membership (spec 2026-08-16 §1).

WHAT THIS FILE CANNOT PROVE. `MockQueryBuilder.eq()/.in_()/.neq()` return
`self` and DISCARD their arguments (tests/conftest.py:66-72), so no assertion
here can show that a filter is actually on a query. Specifically uncovered,
and covered instead by `supabase/qa/gates_boards_on_teams.sql` against real
Postgres:
  - `org_members.status = 'active'` (suspended / removed seats),
  - `organizations.archived_at IS NULL` vs `status <> 'lapsed'` as DISTINCT
    denials — both collapse into one `live_org=False` flag here,
  - `_listed`'s `.eq("user_id", ...)` — dropping it would open a restricted
    board to the whole org and every test below would still pass.
Read a green run as "the branch logic is right", never as "the filters are".
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from boards import authz
from tests.conftest import MockQueryBuilder

OWNER, MEMBER, OUTSIDER = "u-owner", "u-member", "u-out"
ORG, BOARD = "org-1", "b-1"


def _db(*, board, live_org=True, seat=True, admin=False, listed=False):
    """boards row + the three reads can_access_board makes: org_members (seat),
    organizations (liveness), board_members (listing). is_org_admin is an RPC."""

    def _side(name):
        b = MockQueryBuilder()
        if name == "boards":
            b.execute.return_value = MagicMock(data=[board] if board else [])
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"org_id": ORG}] if seat else [])
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG}] if live_org else [])
        elif name == "board_members":
            b.execute.return_value = MagicMock(data=[{"id": "bm"}] if listed else [])
        return b

    db = MagicMock()
    db.table.side_effect = _side

    # Keyed on the FUNCTION NAME, not a blanket return value: with a blanket
    # mock, replacing is_org_admin with is_org_member in the implementation
    # keeps every assertion below green (verified by mutation). This makes the
    # admin-only assertions actually mean "admin".
    def _rpc(fn, params=None):
        return MagicMock(execute=MagicMock(return_value=MagicMock(data=(admin if fn == "is_org_admin" else True))))

    db.rpc.side_effect = _rpc
    return db


PERSONAL = {"id": BOARD, "team_id": None, "owner_id": OWNER, "archived": False, "restricted": False}
# A board owned by an org the caller is NOT in. _db()'s live_org_ids mock
# answers for ORG only, so this is the cross-tenant probe.
FOREIGN = {"id": BOARD, "team_id": "org-2", "owner_id": OWNER, "archived": False, "restricted": False}
OPEN = {"id": BOARD, "team_id": ORG, "owner_id": OWNER, "archived": False, "restricted": False}
RESTRICTED = {**OPEN, "restricted": True}


def test_personal_owner_only():
    assert authz.can_access_board(_db(board=PERSONAL), OWNER, BOARD)
    assert not authz.can_access_board(_db(board=PERSONAL), MEMBER, BOARD)


def test_open_board_needs_live_seat():
    assert authz.can_access_board(_db(board=OPEN), MEMBER, BOARD)
    assert not authz.can_access_board(_db(board=OPEN, seat=False), OUTSIDER, BOARD)
    assert not authz.can_access_board(_db(board=OPEN, live_org=False), MEMBER, BOARD)


def test_restricted_board_owner_admin_listed_only():
    assert authz.can_access_board(_db(board=RESTRICTED), OWNER, BOARD)
    assert authz.can_access_board(_db(board=RESTRICTED, admin=True), MEMBER, BOARD)
    assert authz.can_access_board(_db(board=RESTRICTED, listed=True), MEMBER, BOARD)
    assert not authz.can_access_board(_db(board=RESTRICTED), MEMBER, BOARD)


def test_restricted_admin_still_needs_live_org():
    assert not authz.can_access_board(_db(board=RESTRICTED, admin=True, live_org=False), MEMBER, BOARD)


def test_board_in_a_different_org_is_denied():
    """Cross-tenant probe. `_db` grants a live seat in ORG only; this board
    belongs to org-2. An implementation that asks "is this user in ANY live
    org?" instead of "in THIS org?" passes every other test in this file."""
    assert not authz.can_access_board(_db(board=FOREIGN), MEMBER, BOARD)
    assert not authz.can_assign_user(_db(board=FOREIGN), MEMBER, BOARD)
    assert not authz.can_manage_board(_db(board=FOREIGN), MEMBER, FOREIGN)
    assert not authz.is_board_admin(_db(board=FOREIGN, admin=True), MEMBER, FOREIGN)


def test_missing_board_is_false():
    assert not authz.can_access_board(_db(board=None), OWNER, BOARD)


def test_can_assign_user_mirrors_access():
    assert authz.can_assign_user(_db(board=PERSONAL), OWNER, BOARD)
    assert not authz.can_assign_user(_db(board=PERSONAL), MEMBER, BOARD)
    assert authz.can_assign_user(_db(board=OPEN), MEMBER, BOARD)
    assert not authz.can_assign_user(_db(board=RESTRICTED), MEMBER, BOARD)


def test_can_manage_board_admin_or_owner():
    assert authz.can_manage_board(_db(board=OPEN), OWNER, OPEN)
    assert authz.can_manage_board(_db(board=OPEN, admin=True), MEMBER, OPEN)
    assert not authz.can_manage_board(_db(board=OPEN), MEMBER, OPEN)
    assert authz.can_manage_board(_db(board=PERSONAL), OWNER, PERSONAL)


def test_is_board_admin_is_org_admin_only():
    assert not authz.is_board_admin(_db(board=OPEN), OWNER, OPEN)  # creator but plain member
    assert authz.is_board_admin(_db(board=OPEN, admin=True), MEMBER, OPEN)
    assert not authz.is_board_admin(_db(board=OPEN, admin=True, live_org=False), MEMBER, OPEN)
    assert authz.is_board_admin(_db(board=PERSONAL), OWNER, PERSONAL)


def test_require_board_access_404():
    with pytest.raises(HTTPException) as e:
        authz.require_board_access(_db(board=OPEN, seat=False), OUTSIDER, BOARD)
    assert e.value.status_code == 404


def test_require_org_admin_403():
    with pytest.raises(HTTPException) as e:
        authz.require_org_admin(_db(board=OPEN), MEMBER, ORG)
    assert e.value.status_code == 403
