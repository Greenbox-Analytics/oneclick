"""Tests for registry notifications endpoints.

Acceptance criteria:
1. GET /registry/notifications - list all notifications
2. POST /registry/notifications/{id}/read - mark single notification as read
3. POST /registry/notifications/read-all - mark all notifications as read
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder, _default_table_side_effect

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _sub_wrap(fn):
    """Wrap a table side-effect function to route subscription tables through the Pro default."""

    def _wrapped(name):
        if name in _SUBSCRIPTION_TABLES:
            return _default_table_side_effect(name)
        return fn(name)

    return _wrapped


NOTIFICATION_ID = "11111111-aaaa-0000-0000-000000000001"
WORK_ID = "aaaaaaaa-0000-0000-0000-000000000001"

SAMPLE_NOTIFICATION = {
    "id": NOTIFICATION_ID,
    "user_id": TEST_USER_ID,
    "work_id": WORK_ID,
    "type": "status_change",
    "title": "Work status changed",
    "message": "Your work has been approved.",
    "read": False,
    "metadata": {},
    "created_at": "2026-01-01T00:00:00+00:00",
}

READ_NOTIFICATION = {**SAMPLE_NOTIFICATION, "read": True}


# ============================================================
# List Notifications
# ============================================================


def test_list_notifications_returns_notifications_key(client, mock_supabase):
    """GET /registry/notifications returns {"notifications": [...]}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTIFICATION], count=1)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.get("/registry/notifications")

    assert response.status_code == 200
    body = response.json()
    assert "notifications" in body
    assert isinstance(body["notifications"], list)


def test_list_notifications_empty(client, mock_supabase):
    """GET /registry/notifications returns empty list when no notifications exist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[], count=0)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.get("/registry/notifications")

    assert response.status_code == 200
    assert response.json()["notifications"] == []


def test_list_notifications_with_items(client, mock_supabase):
    """GET /registry/notifications returns notifications from Supabase."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTIFICATION], count=1)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.get("/registry/notifications")

    assert response.status_code == 200
    body = response.json()
    assert len(body["notifications"]) == 1
    assert body["notifications"][0]["id"] == NOTIFICATION_ID
    assert body["notifications"][0]["title"] == "Work status changed"
    assert body["notifications"][0]["read"] is False


def test_list_notifications_unread_only_filter(client, mock_supabase):
    """GET /registry/notifications?unread_only=true filters to unread notifications."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTIFICATION], count=1)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.get("/registry/notifications?unread_only=true")

    assert response.status_code == 200
    body = response.json()
    assert "notifications" in body
    # All returned items should be unread (based on our mock data)
    for notif in body["notifications"]:
        assert notif["read"] is False


def test_list_notifications_unread_only_false_returns_all(client, mock_supabase):
    """GET /registry/notifications?unread_only=false returns all notifications."""
    all_notifications = [SAMPLE_NOTIFICATION, READ_NOTIFICATION]
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=all_notifications, count=2)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.get("/registry/notifications?unread_only=false")

    assert response.status_code == 200
    body = response.json()
    assert len(body["notifications"]) == 2


def test_list_notifications_multiple_types(client, mock_supabase):
    """GET /registry/notifications returns notifications of various types."""
    notif_invite = {
        **SAMPLE_NOTIFICATION,
        "id": "22222222-aaaa-0000-0000-000000000001",
        "type": "invite",
        "title": "Collaboration invite",
    }
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTIFICATION, notif_invite], count=2)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.get("/registry/notifications")

    assert response.status_code == 200
    body = response.json()
    assert len(body["notifications"]) == 2
    types = {n["type"] for n in body["notifications"]}
    assert "status_change" in types
    assert "invite" in types


# ============================================================
# Mark Single Notification Read
# ============================================================


def test_mark_notification_read_success(client, mock_supabase):
    """POST /registry/notifications/{id}/read returns {"ok": True}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[READ_NOTIFICATION])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.post(f"/registry/notifications/{NOTIFICATION_ID}/read")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_mark_notification_read_always_returns_ok(client, mock_supabase):
    """POST /registry/notifications/{id}/read returns {"ok": True} even if not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.post(f"/registry/notifications/{NOTIFICATION_ID}/read")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_mark_notification_read_different_id(client, mock_supabase):
    """POST /registry/notifications/{id}/read works for any notification ID."""
    other_id = "33333333-aaaa-0000-0000-000000000001"
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[{**READ_NOTIFICATION, "id": other_id}])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.post(f"/registry/notifications/{other_id}/read")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ============================================================
# Mark All Notifications Read
# ============================================================


def test_mark_all_notifications_read_success(client, mock_supabase):
    """POST /registry/notifications/read-all returns {"ok": True}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[READ_NOTIFICATION])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.post("/registry/notifications/read-all")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_mark_all_notifications_read_no_notifications(client, mock_supabase):
    """POST /registry/notifications/read-all returns {"ok": True} when no notifications exist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.post("/registry/notifications/read-all")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_mark_all_notifications_read_bulk(client, mock_supabase):
    """POST /registry/notifications/read-all marks multiple notifications as read."""
    many_read = [{**READ_NOTIFICATION, "id": f"notif-{i}"} for i in range(5)]
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=many_read)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    response = client.post("/registry/notifications/read-all")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ============================================================
# "Mark all read" must not consume an unanswered invitation
# ============================================================
# An invite row is a to-do, not a message: NotificationRow gates its
# Accept/Decline buttons on unread. Sweeping one up with the rest is what made
# an invite unrecoverable — the user clicks "Mark all read", the buttons vanish,
# and the invite is still pending server-side with no in-app way to accept it.


async def test_mark_all_read_skips_org_invites():
    """One UPDATE whose filter is the exact negation of NotificationRow.tsx's
    isOrgInvite (type == 'invitation' AND entity_type == 'org'), spelled with
    an explicit NULL arm so registry 'invitation' rows (entity_type NULL) are
    still marked."""
    from registry import service

    db = MagicMock()
    await service.mark_all_notifications_read(db, TEST_USER_ID)

    update_chain = db.table.return_value.update
    update_chain.assert_called_once_with({"read": True})
    eq_calls = [c.args for c in update_chain.return_value.eq.call_args_list]
    assert ("user_id", TEST_USER_ID) in eq_calls
    or_chain = update_chain.return_value.eq.return_value.eq.return_value.or_
    or_chain.assert_called_once_with("type.neq.invitation,entity_type.is.null,entity_type.neq.org")
    or_chain.return_value.execute.assert_called_once()
