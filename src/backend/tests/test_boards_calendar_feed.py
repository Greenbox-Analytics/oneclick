"""Tests for the .ics calendar feed (boards/ics.py).

The feed URL is the only credential a calendar client can present, so the token
check and the escaping/folding of user-supplied text are what matter here.
"""

import os
from unittest.mock import MagicMock, patch

from boards import ics, service

USER = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-8888-7777-6666-555555555555"


def _with_secret():
    return patch.dict(os.environ, {"INTEGRATION_OAUTH_STATE_SECRET": "test-secret"})


def test_token_roundtrips_and_rejects_other_users_and_scopes():
    with _with_secret():
        token = ics.feed_token(USER, "all")
        assert ics.verify_feed_token(USER, "all", token)
        # A valid token for one user must not unlock another user's feed...
        assert not ics.verify_feed_token(OTHER, "all", token)
        # ...nor a different scope of the same user's.
        assert not ics.verify_feed_token(USER, "personal", token)
        assert not ics.verify_feed_token(USER, "all", "deadbeef")
        assert not ics.verify_feed_token(USER, "all", "")


def test_token_requires_a_secret():
    with patch.dict(os.environ, {}, clear=True):
        assert not ics.verify_feed_token(USER, "all", "anything")


def test_feed_url_carries_user_scope_and_token():
    with _with_secret(), patch.dict(os.environ, {"VITE_BACKEND_API_URL": "https://api.example.com/"}):
        url = ics.feed_url(USER, "personal")
        token = ics.feed_token(USER, "personal")
        assert url == f"https://api.example.com/boards/calendar/{USER}/personal/{token}.ics"


def test_feed_names(monkeypatch):
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {"name": "test"}
    ]
    monkeypatch.setattr(service, "_live_org_ids", lambda _db, _uid: ["team-uuid"])
    assert service.feed_name(db, USER, "all") == "Msanii platform - All Tasks Calendar"
    assert service.feed_name(db, USER, "personal") == "Msanii platform - Personal Calendar"
    assert service.feed_name(db, USER, "team-uuid") == "Msanii platform - Team test Calendar"


def test_feed_name_is_generic_for_an_offboarded_subscriber(monkeypatch):
    """The feed token has no membership component and no expiry, so an
    ex-member's subscribed URL keeps resolving. get_feed_tasks already serves
    them nothing; the TITLE must not keep tracking the org's current name."""
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {"name": "Renamed Since They Left"}
    ]
    monkeypatch.setattr(service, "_live_org_ids", lambda _db, _uid: [])  # no live seat
    assert service.feed_name(db, USER, "team-uuid") == service.GENERIC_FEED_NAME


async def test_list_calendar_feeds_labels_every_scope():
    db = MagicMock()
    db.table.return_value.select.return_value.in_.return_value.execute.return_value.data = [
        {"id": "t2", "name": "test2"},
        {"id": "t1", "name": "test"},
    ]
    with patch.object(service, "_live_org_ids", return_value=["t1", "t2"]):
        feeds = await service.list_calendar_feeds(db, USER)

    assert [f["scope"] for f in feeds] == ["all", "personal", "t1", "t2"]  # teams sorted by name
    assert [f["name"] for f in feeds] == [
        "Msanii platform - All Tasks Calendar",
        "Msanii platform - Personal Calendar",
        "Msanii platform - Team test Calendar",
        "Msanii platform - Team test2 Calendar",
    ]
    # Only team feeds carry team_name — it's what the picker labels rows by.
    assert [f["team_name"] for f in feeds] == [None, None, "test", "test2"]


async def test_solo_user_gets_one_feed():
    """With no teams, "everything" and "personal" cover the same boards — offering both
    would just be two identical links."""
    with patch.object(service, "_live_org_ids", return_value=[]):
        feeds = await service.list_calendar_feeds(MagicMock(), USER)
    assert feeds == [{"scope": "all", "name": "Msanii platform - All Tasks Calendar", "team_name": None}]


def test_calendar_name_lands_in_the_feed():
    out = ics.build_ics([], cal_name="Msanii platform - Team Studio A Calendar")
    assert "X-WR-CALNAME:Msanii platform - Team Studio A Calendar" in out


async def test_team_scope_only_resolves_for_a_member():
    """A team feed for a team the caller isn't in must resolve to no boards, not that team's."""
    with (
        patch.object(service, "_live_org_ids", return_value=["team-a"]),
        patch.object(service, "_team_board_ids", return_value=["board-1"]) as team_boards,
        patch.object(service, "_tasks_in_range", return_value=[]) as in_range,
    ):
        await service.get_feed_tasks(None, USER, "team-b", "2026-01-01", "2026-12-31")
        team_boards.assert_not_called()
        in_range.assert_called_once_with(None, USER, [], "2026-01-01", "2026-12-31")

        await service.get_feed_tasks(None, USER, "team-a", "2026-01-01", "2026-12-31")
        team_boards.assert_called_once_with(None, USER, ["team-a"])


def test_build_ics_emits_all_day_events():
    out = ics.build_ics([{"id": "abc", "title": "Master delivery", "due_date": "2026-08-05"}])
    assert out.startswith("BEGIN:VCALENDAR\r\n")
    assert out.endswith("END:VCALENDAR\r\n")
    assert "UID:abc@msanii" in out
    assert "SUMMARY:Master delivery" in out
    # DTEND is exclusive, so a one-day task ends the following day.
    assert "DTSTART;VALUE=DATE:20260805" in out
    assert "DTEND;VALUE=DATE:20260806" in out


def test_build_ics_escapes_text_and_skips_undated_tasks():
    out = ics.build_ics(
        [
            {"id": "a", "title": "Mix; master, v2\\final", "description": "line1\nline2", "due_date": "2026-08-05"},
            {"id": "b", "title": "No due date", "due_date": None},
            {"id": "c", "title": "Garbage date", "due_date": "not-a-date"},
        ]
    )
    assert "SUMMARY:Mix\\; master\\, v2\\\\final" in out
    assert "DESCRIPTION:line1\\nline2" in out
    assert "UID:b@msanii" not in out
    assert "UID:c@msanii" not in out


def test_long_lines_are_folded_without_splitting_characters():
    out = ics.build_ics([{"id": "a", "title": "é" * 200, "due_date": "2026-08-05"}])
    for line in out.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75
    # Unfolding (drop CRLF + one leading space) must restore the original title.
    assert "é" * 200 in out.replace("\r\n ", "")


# --- Single-task .ics export ---


def test_single_task_ics_has_one_event():
    """The per-task download reuses the feed renderer, so a task looks identical
    whether it arrives via the subscription or a one-off file."""
    out = ics.build_ics([{"id": "t1", "title": "Deliver stems", "due_date": "2026-08-05"}])
    assert out.count("BEGIN:VEVENT") == 1
    assert "SUMMARY:Deliver stems" in out
    assert "DTSTART;VALUE=DATE:20260805" in out


def test_single_task_without_due_date_yields_no_event():
    """The endpoint 400s on a dateless task; the renderer independently emits nothing,
    so a regression can't silently produce an empty calendar file."""
    assert ics.build_ics([{"id": "t1", "title": "No date"}]).count("BEGIN:VEVENT") == 0
