"""Dual-channel team invites + ungated notification mark-read (spec 2026-07-03)."""

from unittest.mock import AsyncMock, patch

TEAM = "10000000-0000-0000-0000-000000000001"


def test_existing_user_invite_sends_notification_and_email(client):
    """An existing-user invite fires BOTH the in-app notification and the (existing_user) email."""
    with (
        patch(
            "teams.router.service.invite_member",
            new=AsyncMock(return_value={"type": "invited", "invite": {"token": "tok"}, "notify_user_id": "u1"}),
        ),
        patch("teams.router.service.create_team_invite_notification") as notif,
        patch("teams.router._schedule_invite_email") as email,
    ):
        r = client.post(f"/teams/{TEAM}/invites", json={"email": "a@b.com", "role": "member"})
    assert r.status_code == 200
    notif.assert_called_once()
    email.assert_called_once()
    assert email.call_args.kwargs.get("existing_user") is True


def test_new_user_invite_sends_email_only(client):
    """A new-user invite emails the signup variant and creates no notification."""
    with (
        patch(
            "teams.router.service.invite_member",
            new=AsyncMock(return_value={"type": "invited", "invite": {"token": "tok"}, "notify_user_id": None}),
        ),
        patch("teams.router.service.create_team_invite_notification") as notif,
        patch("teams.router._schedule_invite_email") as email,
    ):
        r = client.post(f"/teams/{TEAM}/invites", json={"email": "new@b.com", "role": "member"})
    assert r.status_code == 200
    notif.assert_not_called()
    email.assert_called_once()
    assert email.call_args.kwargs.get("existing_user") is False


def test_mark_read_not_gated_on_registry(client):
    """mark_read must not call gated_feature (notifications are global now)."""
    import registry.router as rr

    with (
        patch.object(rr, "gated_feature") as gate,
        patch("registry.router.service.mark_notification_read", new=AsyncMock()),
    ):
        r = client.post("/registry/notifications/nid/read")
    assert r.status_code == 200
    gate.assert_not_called()


def test_mark_all_read_not_gated_on_registry(client):
    """read-all must not call gated_feature either (parity with mark_read)."""
    import registry.router as rr

    with (
        patch.object(rr, "gated_feature") as gate,
        patch("registry.router.service.mark_all_notifications_read", new=AsyncMock()),
    ):
        r = client.post("/registry/notifications/read-all")
    assert r.status_code == 200
    gate.assert_not_called()
