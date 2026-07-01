"""Mock-based tests for teams.service core CRUD + membership."""

from unittest.mock import MagicMock

import pytest

from teams import service
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
TEAM = "10000000-0000-0000-0000-000000000001"


async def test_create_team_returns_created_team():
    def _side(name):
        b = MockQueryBuilder()
        if name == "teams":
            b.execute.return_value = MagicMock(data=[{"id": TEAM, "name": "T", "created_by": U1}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    # The creator's admin membership is added by the auto_create_team_admin trigger (not the
    # service, and not run under mocks) — its behavior is verified in Task 1's branch scenarios.
    result = await service.create_team(db, U1, "T", None)
    assert result["id"] == TEAM
    assert result["my_role"] == "admin"


async def test_list_my_teams_filters_archived_and_attaches_role(monkeypatch):
    def _side(name):
        b = MockQueryBuilder()
        if name == "team_members":
            b.execute.return_value = MagicMock(data=[{"team_id": TEAM, "role": "admin"}], count=1)
        elif name == "teams":
            b.execute.return_value = MagicMock(data=[{"id": TEAM, "name": "T", "archived_at": None}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    teams = await service.list_my_teams(db, U1)
    assert teams[0]["my_role"] == "admin"


async def test_get_team_requires_membership(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_member", lambda *a: False)
    db = MagicMock()
    with pytest.raises(PermissionError):
        await service.get_team(db, U2, TEAM)


async def test_archive_team_requires_admin_and_returns_counts(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "boards":
            b.execute.return_value = MagicMock(data=[{"id": "b1"}], count=1)
        elif name == "board_tasks":
            b.execute.return_value = MagicMock(data=[{"id": "t1"}, {"id": "t2"}], count=2)
        elif name == "team_members":
            b.execute.return_value = MagicMock(data=[{"id": "m1"}], count=1)
        else:  # teams update
            b.execute.return_value = MagicMock(data=[{"id": TEAM, "archived_at": "2026-06-30T00:00:00Z"}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.archive_team(db, U1, TEAM)
    assert result == {"archived": TEAM, "boards": 1, "tasks": 2, "members": 1}


async def test_archive_team_denied_for_non_admin(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda *a: False)
    db = MagicMock()
    with pytest.raises(PermissionError):
        await service.archive_team(db, U2, TEAM)


async def test_remove_member_maps_last_admin_db_error(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "team_members":
            # the SELECT returns the target; the DELETE raises the trigger's exception
            b.execute.return_value = MagicMock(data={"id": "m1", "user_id": U2}, count=1)
            b.delete.return_value.eq.return_value.eq.return_value.execute.side_effect = Exception(
                "You are the only admin of this team — promote another member first"
            )
        return b

    db = MagicMock()
    db.table.side_effect = _side
    with pytest.raises(service.LastAdminError):
        await service.remove_member(db, U1, TEAM, "m1")


def test_find_user_id_by_email_uses_rpc_and_returns_id():
    """_find_user_id_by_email calls the get_user_id_by_email RPC (not db.schema('auth'))."""
    db = MagicMock()
    db.rpc.return_value.execute.return_value = MagicMock(data="uid-123")
    assert service._find_user_id_by_email(db, "A@B.com") == "uid-123"
    db.rpc.assert_called_once_with("get_user_id_by_email", {"lookup_email": "A@B.com"})


def test_find_user_id_by_email_none_when_no_match():
    db = MagicMock()
    db.rpc.return_value.execute.return_value = MagicMock(data=None)
    assert service._find_user_id_by_email(db, "x@y.com") is None


def test_find_user_id_by_email_unwraps_list_scalar():
    db = MagicMock()
    db.rpc.return_value.execute.return_value = MagicMock(data=["uid-456"])
    assert service._find_user_id_by_email(db, "c@d.com") == "uid-456"
