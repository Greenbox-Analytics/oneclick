"""Team hard-delete (archive-first) / restore / archived-listing (spec 2026-07-03)."""

from unittest.mock import MagicMock

import pytest

from teams import service
from tests.conftest import MockQueryBuilder

USER = "00000000-0000-0000-0000-000000000001"
TEAM = "10000000-0000-0000-0000-000000000001"


def _db(tbl):
    def _side(name):
        b = MockQueryBuilder()
        if name in tbl:
            b.execute.return_value = tbl[name]
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


async def test_delete_team_requires_archived_first(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda db, u, t: True)
    db = _db({"teams": MagicMock(data=[{"id": TEAM, "name": "My Team", "archived_at": None}])})
    with pytest.raises(service.NotArchivedError):
        await service.delete_team(db, USER, TEAM, "My Team")


async def test_delete_team_requires_admin(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda db, u, t: False)
    db = _db({"teams": MagicMock(data=[{"id": TEAM, "name": "My Team", "archived_at": "2026-01-01T00:00:00Z"}])})
    with pytest.raises(PermissionError):
        await service.delete_team(db, USER, TEAM, "My Team")


async def test_delete_team_rejects_wrong_name(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda db, u, t: True)
    db = _db({"teams": MagicMock(data=[{"id": TEAM, "name": "My Team", "archived_at": "2026-01-01T00:00:00Z"}])})
    with pytest.raises(service.ConfirmationError):
        await service.delete_team(db, USER, TEAM, "nope")


async def test_delete_team_ok_when_archived_and_name_matches(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda db, u, t: True)
    db = _db(
        {
            "teams": MagicMock(data=[{"id": TEAM, "name": "My Team", "archived_at": "2026-01-01T00:00:00Z"}]),
            # boards.data MUST be non-empty with ids: delete_team gates the task-count query on
            # `if board_ids:`, so an empty list would skip it and yield tasks=0 (not 5).
            "boards": MagicMock(data=[{"id": "board-1"}, {"id": "board-2"}], count=2),
            "board_tasks": MagicMock(data=[], count=5),
            "team_members": MagicMock(data=[], count=3),
        }
    )
    result = await service.delete_team(db, USER, TEAM, "  My Team ")  # trimmed
    assert result == {"deleted": TEAM, "boards": 2, "tasks": 5, "members": 3}


async def test_restore_team(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda db, u, t: True)
    db = _db({"teams": MagicMock(data=[{"id": TEAM, "archived_at": None}])})
    result = await service.restore_team(db, USER, TEAM)
    assert result == {"restored": TEAM}


# --- delete/restore edge cases (M1) ---


async def test_delete_team_not_found(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda db, u, t: True)
    db = _db({"teams": MagicMock(data=[])})
    with pytest.raises(ValueError):
        await service.delete_team(db, USER, TEAM, "x")


async def test_restore_team_denies_non_admin(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda db, u, t: False)
    db = _db({"teams": MagicMock(data=[{"id": TEAM, "archived_at": "2026-01-01T00:00:00Z"}])})
    with pytest.raises(PermissionError):
        await service.restore_team(db, USER, TEAM)


# --- real is_team_admin gate wiring (I2) ---


async def test_delete_team_uses_real_admin_gate_denies():
    # Do NOT monkeypatch is_team_admin — exercise the real authz gate. With no admin
    # team_members row, is_team_admin returns False → delete_team raises PermissionError.
    db = _db({"team_members": MagicMock(data=[])})
    with pytest.raises(PermissionError):
        await service.delete_team(db, USER, TEAM, "x")


# --- list_archived_teams (I1): admin scoping + counts ---


async def test_list_archived_teams_denies_non_admin_caller():
    # A member-only caller: the memberships query filters role == "admin", so it returns no
    # rows → no admin team ids → empty list. This is the load-bearing authz scoping assertion.
    db = _db({"team_members": MagicMock(data=[])})
    assert await service.list_archived_teams(db, USER) == []


async def test_list_archived_teams_empty_when_no_memberships():
    # No admin memberships → early return before the teams table is ever queried.
    db = _db({"team_members": MagicMock(data=[])})
    assert await service.list_archived_teams(db, USER) == []


async def test_list_archived_teams_admin_returns_teams_with_counts():
    db = _db(
        {
            # reused for BOTH the admin-memberships query (.data) and the per-team members count (.count)
            "team_members": MagicMock(data=[{"team_id": TEAM}], count=3),
            "teams": MagicMock(data=[{"id": TEAM, "name": "Arch", "archived_at": "2026-01-01T00:00:00Z"}]),
            "boards": MagicMock(data=[{"id": "board-1"}], count=1),
            "board_tasks": MagicMock(data=[], count=7),
        }
    )
    result = await service.list_archived_teams(db, USER)
    assert len(result) == 1
    t = result[0]
    assert t["id"] == TEAM
    assert (t["boards"], t["tasks"], t["members"]) == (1, 7, 3)
