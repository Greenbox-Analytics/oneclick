"""iCalendar (RFC 5545) feed for workspace tasks.

Google Calendar, Apple Calendar and Outlook all natively subscribe to an .ics URL
and re-poll it on their own schedule, so one public read-only feed covers every
client — no per-provider OAuth and no task→event id bookkeeping to keep in sync.

The feed URL is the credential (calendar clients send no auth headers), so it
carries an HMAC of the user id that only the server can produce.
"""

import hmac
import os
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256

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
