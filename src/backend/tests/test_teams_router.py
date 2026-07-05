"""Endpoint contract tests for the teams router (uses the shared `client` fixture)."""

from unittest.mock import AsyncMock, patch


def test_list_teams_ok(client):
    with patch("teams.router.service.list_my_teams", new=AsyncMock(return_value=[{"id": "t1", "my_role": "admin"}])):
        resp = client.get("/teams")
    assert resp.status_code == 200
    assert resp.json() == {"teams": [{"id": "t1", "my_role": "admin"}]}


def test_create_team_ok(client):
    with patch("teams.router.service.create_team", new=AsyncMock(return_value={"id": "t1", "name": "T"})):
        resp = client.post("/teams", json={"name": "T"})
    assert resp.status_code == 200
    assert resp.json()["id"] == "t1"


def test_get_team_not_member_403(client):
    with patch("teams.router.service.get_team", new=AsyncMock(side_effect=PermissionError())):
        resp = client.get("/teams/t1")
    assert resp.status_code == 403


def test_invite_duplicate_409(client):
    from teams.service import DuplicateInviteError

    with patch("teams.router.service.invite_member", new=AsyncMock(side_effect=DuplicateInviteError("dup"))):
        resp = client.post("/teams/t1/invites", json={"email": "a@b.com", "role": "member"})
    assert resp.status_code == 409


def test_remove_member_last_admin_409(client):
    from teams.service import LastAdminError

    with patch("teams.router.service.remove_member", new=AsyncMock(side_effect=LastAdminError("only admin"))):
        resp = client.delete("/teams/t1/members/m1")
    assert resp.status_code == 409


def test_accept_invite_expired_410(client):
    from teams.service import InviteInvalidError

    with patch("teams.router.service.accept_invite", new=AsyncMock(side_effect=InviteInvalidError("expired"))):
        resp = client.post("/teams/invites/tok/accept")
    assert resp.status_code == 410
