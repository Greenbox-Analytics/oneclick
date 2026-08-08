"""iCalendar (RFC 5545) support for the workspace calendar — both directions.

EXPORT (build_ics): Google Calendar, Apple Calendar and Outlook all natively
subscribe to an .ics URL and re-poll it on their own schedule, so one public
read-only feed covers every client — no per-provider OAuth and no task→event id
bookkeeping to keep in sync. The feed URL is the credential (calendar clients
send no auth headers), so it carries an HMAC of the user id that only the server
can produce.

IMPORT (parse_ics): the same trick in reverse — the user pastes the secret iCal
URL their calendar already publishes and we render those events read-only. No
OAuth, and nothing is persisted, so there is no sync state to reconcile.
"""

import hmac
import os
import re
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256

from dateutil import tz
from dateutil.rrule import rrulestr

# ponytail: feed tokens are derived, not stored, so revoking one means rotating
# INTEGRATION_OAUTH_STATE_SECRET (which invalidates every feed). Add a per-user
# token column if a single user ever needs to revoke their own URL.
_SECRET_ENV = "INTEGRATION_OAUTH_STATE_SECRET"

# ICS PRIORITY is 1 (highest) to 9 (lowest); 0 / absent means undefined.
_PRIORITY = {"urgent": 1, "high": 3, "medium": 5, "low": 7}

FEED_WINDOW_DAYS = 365


def feed_token(user_id: str, scope: str) -> str:
    secret = os.getenv(_SECRET_ENV)
    if not secret:
        raise RuntimeError(f"{_SECRET_ENV} not set.")
    # Domain-separated so a feed token can never be replayed as an OAuth state.
    return hmac.new(secret.encode(), f"calfeed:{user_id}:{scope}".encode(), sha256).hexdigest()[:32]


def verify_feed_token(user_id: str, scope: str, token: str) -> bool:
    try:
        return hmac.compare_digest(feed_token(user_id, scope), token)
    except RuntimeError:
        return False


def feed_url(user_id: str, scope: str) -> str:
    base = os.getenv("VITE_BACKEND_API_URL", "http://localhost:8000").rstrip("/")
    return f"{base}/boards/calendar/{user_id}/{scope}/{feed_token(user_id, scope)}.ics"


def _esc(value: str) -> str:
    """Escape a TEXT property value (RFC 5545 §3.3.11)."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _fold(line: str) -> str:
    """Fold to 75 octets per line (RFC 5545 §3.1), never mid-codepoint."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    parts, i, limit = [], 0, 75
    while i < len(raw):
        chunk = raw[i : i + limit]
        # Back off while the next byte is a UTF-8 continuation byte.
        while chunk and i + len(chunk) < len(raw) and (raw[i + len(chunk)] & 0xC0) == 0x80:
            chunk = chunk[:-1]
        parts.append(chunk.decode("utf-8"))
        i += len(chunk)
        limit = 74  # continuation lines spend one octet on the leading space
    return "\r\n ".join(parts)


def build_ics(tasks: list[dict], cal_name: str = "Msanii platform - Calendar") -> str:
    """Render tasks as all-day VEVENTs on their due date.

    All-day (VALUE=DATE) keeps the feed timezone-free: due_date is a plain date,
    so a task can't drift a day for a user in another timezone.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Msanii//Workspace Calendar//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_esc(cal_name)}",
        "REFRESH-INTERVAL;VALUE=DURATION:PT1H",
        "X-PUBLISHED-TTL:PT1H",
    ]
    for task in tasks:
        try:
            due = date.fromisoformat(str(task.get("due_date"))[:10])
        except (TypeError, ValueError):
            continue
        lines += [
            "BEGIN:VEVENT",
            f"UID:{task['id']}@msanii",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{due:%Y%m%d}",
            f"DTEND;VALUE=DATE:{due + timedelta(days=1):%Y%m%d}",
            f"SUMMARY:{_esc(task.get('title') or 'Untitled task')}",
        ]
        if task.get("description"):
            lines.append(f"DESCRIPTION:{_esc(task['description'])}")
        if task.get("team_name"):
            lines.append(f"CATEGORIES:{_esc(task['team_name'])}")
        if _PRIORITY.get(task.get("priority")):
            lines.append(f"PRIORITY:{_PRIORITY[task['priority']]}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"


# --- Import: parsing a subscribed .ics feed ---------------------------------

# A rule like FREQ=DAILY with no UNTIL expands without bound; this caps it.
MAX_OCCURRENCES_PER_RULE = 400


def _unescape(value: str) -> str:
    r"""Reverse RFC 5545 TEXT escaping. One left-to-right pass over non-overlapping
    matches, so a literal \\ can't be re-read as the start of the next escape."""
    return re.sub(r"\\(.)", lambda m: "\n" if m[1] in "nN" else m[1], value)


def _unfold(text: str) -> list[str]:
    """Reverse RFC 5545 line folding: a leading space/tab continues the line above."""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _split_property(line: str) -> tuple[str, dict[str, str], str]:
    """ "DTSTART;TZID=America/New_York:20260805T140000" → ("DTSTART", {"TZID": ...}, "20260805T140000")."""
    head, _, value = line.partition(":")
    name, *param_parts = head.split(";")
    params = {}
    for part in param_parts:
        key, _, val = part.partition("=")
        params[key.upper()] = val.strip('"')
    return name.upper(), params, value


def _parse_dt(value: str, params: dict[str, str], default_tz) -> tuple[datetime | None, bool]:
    """Return (aware datetime, is_all_day). All-day values carry no meaningful time."""
    value = value.strip()
    if params.get("VALUE", "").upper() == "DATE" or (len(value) == 8 and value.isdigit()):
        try:
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=default_tz), True
        except ValueError:
            return None, True
    fmt = "%Y%m%dT%H%M%SZ" if value.endswith("Z") else "%Y%m%dT%H%M%S"
    try:
        parsed = datetime.strptime(value, fmt)
    except ValueError:
        return None, False
    if value.endswith("Z"):
        return parsed.replace(tzinfo=UTC), False
    # A TZID we don't recognise falls back to the viewer's timezone rather than
    # being dropped — a slightly-off time beats a missing event.
    zone = tz.gettz(params["TZID"]) if params.get("TZID") else None
    return parsed.replace(tzinfo=zone or default_tz), False


def _occurrences(dtstart: datetime, rrule_value: str, window_start: datetime, window_end: datetime):
    """Expand an RRULE inside the window. dateutil does the calendar maths — hand-rolling
    BYDAY/BYSETPOS/leap-year handling is exactly the kind of thing that quietly gets it wrong."""
    try:
        rule = rrulestr(rrule_value, dtstart=dtstart)
    except (ValueError, TypeError, OverflowError):
        return [dtstart]
    try:
        found = rule.between(window_start, window_end, inc=True)
    except (ValueError, OverflowError):
        return [dtstart]
    return found[:MAX_OCCURRENCES_PER_RULE]


def parse_ics(text: str, window_start: date, window_end: date, timezone: str | None = None) -> list[dict]:
    """Parse a subscribed .ics feed into events overlapping [window_start, window_end].

    Returns dicts shaped for the calendar overlay: {uid, title, date, time, all_day}.
    `date` is the local date the event falls on — the calendar grid is date-keyed —
    and `time` is a display string ("2:00 PM") or None for all-day events.

    Malformed events are skipped rather than failing the whole feed: a single bad
    VEVENT in someone's calendar should not blank out the other 200.
    """
    default_tz = (tz.gettz(timezone) if timezone else None) or UTC
    lo = datetime.combine(window_start, datetime.min.time(), tzinfo=default_tz)
    hi = datetime.combine(window_end, datetime.max.time(), tzinfo=default_tz)

    events: list[dict] = []
    current: dict | None = None
    for line in _unfold(text):
        if line.startswith("BEGIN:VEVENT"):
            current = {}
            continue
        if line.startswith("END:VEVENT"):
            if current is not None:
                events.extend(_expand_event(current, lo, hi, default_tz))
            current = None
            continue
        if current is None or ":" not in line:
            continue
        name, params, value = _split_property(line)
        if name in ("SUMMARY", "UID", "RRULE", "DTSTART", "STATUS"):
            current[name] = (params, value)

    return sorted(events, key=lambda e: (e["date"], e["time"] or ""))


def _expand_event(props: dict, lo: datetime, hi: datetime, default_tz) -> list[dict]:
    if "DTSTART" not in props:
        return []
    if (props.get("STATUS") or (None, ""))[1].upper() == "CANCELLED":
        return []

    start, all_day = _parse_dt(props["DTSTART"][1], props["DTSTART"][0], default_tz)
    if start is None:
        return []

    title = _unescape(props.get("SUMMARY", ({}, "Untitled"))[1]).strip() or "Untitled"
    uid = props.get("UID", ({}, ""))[1] or f"{title}-{start.isoformat()}"

    starts = _occurrences(start, props["RRULE"][1], lo, hi) if "RRULE" in props else [start]

    out = []
    for occurrence in starts:
        if not (lo <= occurrence <= hi):
            continue
        local = occurrence.astimezone(default_tz)
        out.append(
            {
                "uid": f"{uid}@{local.date().isoformat()}",
                "title": title,
                "date": local.date().isoformat(),
                "time": None if all_day else local.strftime("%-I:%M %p"),
                "all_day": all_day,
            }
        )
    return out
