"""Unit tests for teams.authz access-control helpers (mock-based, no DB)."""

import pytest
from fastapi import HTTPException

from teams import authz
from tests.conftest import MockQueryBuilder

OWNER = "00000000-0000-0000-0000-000000000001"
OTHER = "00000000-0000-0000-0000-000000000002"
TEAM = "10000000-0000-0000-0000-000000000001"
PERSONAL_BOARD = "20000000-0000-0000-0000-000000000001"
TEAM_BOARD = "20000000-0000-0000-0000-000000000002"


def _client(*, board=None, members=None):
    """Build a mock supabase whose table() returns rows for boards/team_members.

    `board` is the single boards row (or None). `members` is the list of
    team_members rows the membership queries should see.
    """
    from unittest.mock import MagicMock

    def _side(name):
        b = MockQueryBuilder()
        if name == "boards":
            b.execute.return_value = MagicMock(data=([board] if board else []), count=0)
        elif name == "team_members":
            b.execute.return_value = MagicMock(data=(members or []), count=len(members or []))
        else:
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    client = MagicMock()
    client.table.side_effect = _side
    return client


def test_is_team_member_true_when_row_present():
    client = _client(members=[{"id": "m1"}])
    assert authz.is_team_member(client, OWNER, TEAM) is True


def test_is_team_member_false_when_empty():
    client = _client(members=[])
    assert authz.is_team_member(client, OTHER, TEAM) is False


def test_is_team_admin_true_when_admin_row_present():
    client = _client(members=[{"id": "m1"}])
    assert authz.is_team_admin(client, OWNER, TEAM) is True


def test_can_access_personal_board_owner_true():
    board = {"id": PERSONAL_BOARD, "team_id": None, "owner_id": OWNER, "archived": False}
    client = _client(board=board)
    assert authz.can_access_board(client, OWNER, PERSONAL_BOARD) is True


def test_can_access_personal_board_other_false():
    board = {"id": PERSONAL_BOARD, "team_id": None, "owner_id": OWNER, "archived": False}
    client = _client(board=board)
    assert authz.can_access_board(client, OTHER, PERSONAL_BOARD) is False


def test_can_access_team_board_member_true():
    board = {"id": TEAM_BOARD, "team_id": TEAM, "owner_id": OWNER, "archived": False}
    client = _client(board=board, members=[{"id": "m1"}])
    assert authz.can_access_board(client, OTHER, TEAM_BOARD) is True


def test_can_access_missing_board_false():
    client = _client(board=None)
    assert authz.can_access_board(client, OWNER, "deadbeef") is False


def test_can_edit_board_matches_access_for_team_member():
    board = {"id": TEAM_BOARD, "team_id": TEAM, "owner_id": OWNER, "archived": False}
    client = _client(board=board, members=[{"id": "m1"}])
    assert authz.can_edit_board(client, OTHER, TEAM_BOARD) is True


def test_can_assign_personal_board_only_self():
    board = {"id": PERSONAL_BOARD, "team_id": None, "owner_id": OWNER, "archived": False}
    client = _client(board=board)
    assert authz.can_assign_user(client, OWNER, PERSONAL_BOARD) is True
    # a non-owner cannot be assigned on a personal board
    client2 = _client(board=board)
    assert authz.can_assign_user(client2, OTHER, PERSONAL_BOARD) is False


def test_can_assign_team_board_requires_membership():
    board = {"id": TEAM_BOARD, "team_id": TEAM, "owner_id": OWNER, "archived": False}
    member_client = _client(board=board, members=[{"id": "m1"}])
    assert authz.can_assign_user(member_client, OTHER, TEAM_BOARD) is True
    non_member_client = _client(board=board, members=[])
    assert authz.can_assign_user(non_member_client, OTHER, TEAM_BOARD) is False


def test_require_board_access_raises_404_when_denied():
    board = {"id": PERSONAL_BOARD, "team_id": None, "owner_id": OWNER, "archived": False}
    client = _client(board=board)
    with pytest.raises(HTTPException) as exc:
        authz.require_board_access(client, OTHER, PERSONAL_BOARD)
    assert exc.value.status_code == 404


def test_require_board_edit_raises_404_when_denied():
    board = {"id": PERSONAL_BOARD, "team_id": None, "owner_id": OWNER, "archived": False}
    client = _client(board=board)
    with pytest.raises(HTTPException) as exc:
        authz.require_board_edit(client, OTHER, PERSONAL_BOARD)
    assert exc.value.status_code == 404


def test_require_team_admin_raises_403_when_not_admin():
    client = _client(members=[])
    with pytest.raises(HTTPException) as exc:
        authz.require_team_admin(client, OTHER, TEAM)
    assert exc.value.status_code == 403
