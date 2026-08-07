"""Tests for importing external .ics calendars.

Two things here genuinely must not break: the SSRF guard (the server fetches a
user-supplied URL) and the parser (a malformed feed must degrade, not explode).
"""

from datetime import date
from unittest.mock import patch

import pytest

from boards import calendar_import, ics

WINDOW_START = date(2026, 8, 1)
WINDOW_END = date(2026, 8, 31)


def _feed(*vevents: str) -> str:
    body = "\r\n".join(vevents)
    return f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\n{body}\r\nEND:VCALENDAR\r\n"


# --- URL normalization + SSRF guard ---


def test_webcal_urls_are_accepted():
    assert calendar_import.normalize_url("webcal://example.com/a.ics") == "https://example.com/a.ics"
    assert calendar_import.normalize_url("  https://example.com/a.ics  ") == "https://example.com/a.ics"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/a.ics",
        "gopher://example.com/",
        "javascript:alert(1)",
        "not a url",
    ],
)
def test_non_http_schemes_rejected(url):
    with pytest.raises(calendar_import.FeedError):
        calendar_import.normalize_url(url)


@pytest.mark.parametrize(
    "addr",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # private
        "192.168.1.10",  # private
        "172.16.0.9",  # private
        "169.254.169.254",  # cloud metadata — the one that leaks service-account tokens
        "0.0.0.0",
        "::1",  # loopback v6
        "fd00::1",  # unique-local v6
    ],
)
def test_private_addresses_are_blocked(addr):
    family = 10 if ":" in addr else 2
    with (
        patch("boards.calendar_import.socket.getaddrinfo", return_value=[(family, 1, 6, "", (addr, 0))]),
        pytest.raises(calendar_import.FeedError),
    ):
        calendar_import._assert_public_host("https://evil.test/a.ics")


def test_public_address_allowed():
    with patch("boards.calendar_import.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]):
        calendar_import._assert_public_host("https://example.com/a.ics")


def test_split_horizon_dns_is_blocked():
    """A host resolving to BOTH a public and a private address must be rejected —
    otherwise the private one is still reachable."""
    infos = [(2, 1, 6, "", ("93.184.216.34", 0)), (2, 1, 6, "", ("127.0.0.1", 0))]
    with (
        patch("boards.calendar_import.socket.getaddrinfo", return_value=infos),
        pytest.raises(calendar_import.FeedError),
    ):
        calendar_import._assert_public_host("https://evil.test/a.ics")


# --- Parsing ---


def test_parses_timed_and_all_day_events():
    feed = _feed(
        "BEGIN:VEVENT\r\nUID:a\r\nSUMMARY:Studio session\r\nDTSTART:20260805T140000Z\r\nEND:VEVENT",
        "BEGIN:VEVENT\r\nUID:b\r\nSUMMARY:Album drop\r\nDTSTART;VALUE=DATE:20260810\r\nEND:VEVENT",
    )
    events = ics.parse_ics(feed, WINDOW_START, WINDOW_END, "UTC")

    assert [e["title"] for e in events] == ["Studio session", "Album drop"]
    assert events[0] == {
        "uid": "a@2026-08-05",
        "title": "Studio session",
        "date": "2026-08-05",
        "time": "2:00 PM",
        "all_day": False,
    }
    assert events[1]["all_day"] is True and events[1]["time"] is None


def test_events_outside_the_window_are_dropped():
    feed = _feed(
        "BEGIN:VEVENT\r\nUID:in\r\nSUMMARY:Inside\r\nDTSTART;VALUE=DATE:20260805\r\nEND:VEVENT",
        "BEGIN:VEVENT\r\nUID:out\r\nSUMMARY:Outside\r\nDTSTART;VALUE=DATE:20261225\r\nEND:VEVENT",
    )
    assert [e["title"] for e in ics.parse_ics(feed, WINDOW_START, WINDOW_END, "UTC")] == ["Inside"]


def test_recurring_event_expands_within_the_window():
    """A weekly standup should appear on every matching day, not just the first."""
    feed = _feed(
        "BEGIN:VEVENT\r\nUID:r\r\nSUMMARY:Weekly standup\r\n"
        "DTSTART:20260803T090000Z\r\nRRULE:FREQ=WEEKLY;BYDAY=MO\r\nEND:VEVENT"
    )
    events = ics.parse_ics(feed, WINDOW_START, WINDOW_END, "UTC")
    # Mondays in Aug 2026: 3, 10, 17, 24, 31
    assert [e["date"] for e in events] == [
        "2026-08-03",
        "2026-08-10",
        "2026-08-17",
        "2026-08-24",
        "2026-08-31",
    ]
    assert len({e["uid"] for e in events}) == 5  # distinct per occurrence


def test_cancelled_events_are_skipped():
    feed = _feed(
        "BEGIN:VEVENT\r\nUID:x\r\nSUMMARY:Called off\r\nDTSTART;VALUE=DATE:20260805\r\nSTATUS:CANCELLED\r\nEND:VEVENT"
    )
    assert ics.parse_ics(feed, WINDOW_START, WINDOW_END, "UTC") == []


def test_escapes_are_restored():
    feed = _feed(
        "BEGIN:VEVENT\r\nUID:f\r\nSUMMARY:Mix\\; master\\, take 2\r\nDTSTART;VALUE=DATE:20260805\r\nEND:VEVENT"
    )
    events = ics.parse_ics(feed, WINDOW_START, WINDOW_END, "UTC")
    assert events[0]["title"] == "Mix; master, take 2"


def test_folded_long_line_is_rejoined():
    """A title over 75 octets arrives split across lines with a leading space;
    unfolding must rebuild it exactly (RFC 5545 §3.1)."""
    title = "Mastering session with the whole band plus the engineer and the label rep"
    feed = _feed(
        f"BEGIN:VEVENT\r\nUID:f\r\nSUMMARY:{title[:60]}\r\n {title[60:]}\r\nDTSTART;VALUE=DATE:20260805\r\nEND:VEVENT"
    )
    events = ics.parse_ics(feed, WINDOW_START, WINDOW_END, "UTC")
    assert events[0]["title"] == title


def test_long_title_survives_an_export_import_roundtrip():
    """build_ics folds; parse_ics unfolds. The pair must be lossless."""
    title = "é" * 200
    exported = ics.build_ics([{"id": "t1", "title": title, "due_date": "2026-08-05"}])
    assert ics.parse_ics(exported, WINDOW_START, WINDOW_END, "UTC")[0]["title"] == title


def test_malformed_event_does_not_kill_the_feed():
    """One bad VEVENT should not blank out the rest of someone's calendar."""
    feed = _feed(
        "BEGIN:VEVENT\r\nUID:bad\r\nSUMMARY:No start date\r\nEND:VEVENT",
        "BEGIN:VEVENT\r\nUID:bad2\r\nSUMMARY:Junk date\r\nDTSTART:not-a-date\r\nEND:VEVENT",
        "BEGIN:VEVENT\r\nUID:good\r\nSUMMARY:Survivor\r\nDTSTART;VALUE=DATE:20260805\r\nEND:VEVENT",
    )
    assert [e["title"] for e in ics.parse_ics(feed, WINDOW_START, WINDOW_END, "UTC")] == ["Survivor"]


def test_runaway_recurrence_is_capped():
    """A daily-forever rule must not expand without bound."""
    feed = _feed(
        "BEGIN:VEVENT\r\nUID:r\r\nSUMMARY:Every day\r\nDTSTART:20260101T090000Z\r\nRRULE:FREQ=DAILY\r\nEND:VEVENT"
    )
    events = ics.parse_ics(feed, date(2026, 1, 1), date(2036, 1, 1), "UTC")
    assert len(events) <= ics.MAX_OCCURRENCES_PER_RULE


def test_empty_and_garbage_input_is_safe():
    assert ics.parse_ics("", WINDOW_START, WINDOW_END, "UTC") == []
    assert ics.parse_ics("total garbage", WINDOW_START, WINDOW_END, "UTC") == []


def test_roundtrip_export_then_import():
    """Our own published feed must parse back — the two directions agree on the format."""
    exported = ics.build_ics([{"id": "t1", "title": "Deliver stems", "due_date": "2026-08-05"}])
    events = ics.parse_ics(exported, WINDOW_START, WINDOW_END, "UTC")
    assert len(events) == 1
    assert events[0]["title"] == "Deliver stems"
    assert events[0]["date"] == "2026-08-05"
    assert events[0]["all_day"] is True
