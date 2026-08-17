"""Ungated notification mark-read (spec 2026-07-03)."""

from unittest.mock import AsyncMock, patch


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
