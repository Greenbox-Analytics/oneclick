"""Mock-based tests for teams.service invite flow.

Async tests are collected automatically (asyncio_mode = auto, pyproject.toml). On a single
table, select/insert/update chains all resolve to builder.execute.return_value (see
conftest's MockQueryBuilder), so a table that's queried twice in one function needs its
results fed in CALL ORDER — that's what _db_seq does. (delete uses an independent chain and
isn't exercised here.)
"""

from unittest.mock import MagicMock

import pytest

from teams import service
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
EXISTING = "00000000-0000-0000-0000-000000000099"
TEAM = "10000000-0000-0000-0000-000000000001"
TOKEN = "30000000-0000-0000-0000-000000000001"


def _db_seq(seqs):
    """seqs: dict table_name -> list of execute() return values, consumed in call order."""
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


async def test_invite_existing_member_raises_duplicate(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda *a: True)
    monkeypatch.setattr(service.authz, "is_team_member", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: EXISTING)
    with pytest.raises(service.DuplicateInviteError):
        await service.invite_member(_db_seq({}), U1, TEAM, "x@example.com", "member")


async def test_invite_fresh_email_inserts_pending(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    db = _db_seq(
        {
            "pending_team_invites": [
                MagicMock(data=None, count=0),  # 1st call: existing-invite lookup -> none
                MagicMock(  # 2nd call: insert -> new row
                    data=[{"id": "i1", "token": TOKEN, "email": "new@example.com", "role": "member"}], count=1
                ),
            ]
        }
    )
    result = await service.invite_member(db, U1, TEAM, "new@example.com", "member")
    assert result["type"] == "invited"
    assert result["notify_user_id"] is None
    assert result["invite"]["token"] == TOKEN


async def test_invite_existing_invite_updates_row(monkeypatch):
    monkeypatch.setattr(service.authz, "is_team_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    db = _db_seq(
        {
            "pending_team_invites": [
                MagicMock(data={"id": "i1"}, count=1),  # 1st call: existing-invite lookup -> found
                MagicMock(  # 2nd call: update -> updated row
                    data=[{"id": "i1", "token": TOKEN, "status": "pending", "role": "admin"}], count=1
                ),
            ]
        }
    )
    result = await service.invite_member(db, U1, TEAM, "back@example.com", "admin")
    assert result["invite"]["status"] == "pending"
    assert result["invite"]["role"] == "admin"


async def test_accept_invite_rejects_email_mismatch():
    db = _db_seq(
        {
            "pending_team_invites": [
                MagicMock(
                    data={
                        "id": "i1",
                        "team_id": TEAM,
                        "email": "owner@example.com",
                        "role": "member",
                        "status": "pending",
                        "expires_at": "2999-01-01T00:00:00+00:00",
                        "invited_by": U1,
                    },
                    count=1,
                )
            ]
        }
    )
    with pytest.raises(PermissionError):
        await service.accept_invite(db, "someone", "intruder@example.com", TOKEN)


async def test_accept_invite_expired_raises():
    db = _db_seq(
        {
            "pending_team_invites": [
                MagicMock(
                    data={
                        "id": "i1",
                        "team_id": TEAM,
                        "email": "u@example.com",
                        "role": "member",
                        "status": "pending",
                        "expires_at": "2000-01-01T00:00:00+00:00",
                        "invited_by": U1,
                    },
                    count=1,
                )
            ]
        }
    )
    with pytest.raises(service.InviteInvalidError):
        await service.accept_invite(db, "u", "u@example.com", TOKEN)


async def test_create_team_invite_notification_inserts_row():
    captured = {}
    b = MockQueryBuilder()

    def _insert(payload):
        captured.update(payload)
        return b

    b.insert.side_effect = _insert
    db = MagicMock()
    db.table.return_value = b
    service.create_team_invite_notification(db, EXISTING, TEAM, "Acme", "Alice", TOKEN)
    assert captured["type"] == "team_invite"
    assert captured["entity_type"] == "team"
    assert captured["entity_id"] == TEAM
    assert captured["metadata"]["token"] == TOKEN
