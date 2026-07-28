"""Public POST /contact-submissions — the Contact page form.

Records a row in contact_submissions AND emails ops via Resend (best-effort),
with any attachments forwarded on the email. The DB row is the durable source
of truth; a Resend failure does not fail the request.

Mirrors subscriptions/pro_requests_router.py, with two additions this endpoint
needs and that one does not: it accepts file uploads, and it is protected by a
honeypot field plus a Postgres-backed rate limit. Both exist because this is an
unauthenticated endpoint that relays attachments into an inbox.

Like pro_requests, this depends on `get_supabase_client()` returning a
service-role client. contact_submissions has SELECT USING (false) deny-all and
no INSERT policy, so anonymous browser INSERTs are blocked by RLS — writes go
only through this endpoint, after validation.
"""

import base64
import html
import logging
import os
import secrets
import string
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import resend
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import EmailStr

# Ensure backend dir is in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_optional_user_id

router = APIRouter(tags=["Contact"])

MODES = ("ticket", "message")

# Field length caps. This is anonymous public input landing in an email body —
# without caps a single request can post an arbitrarily large message.
MAX_NAME = 200
MAX_SUBJECT = 200
MAX_MESSAGE = 5000
MAX_SHORT_FIELD = 200

# Attachment limits. Files are forwarded on the ops email and never persisted.
MAX_FILES = 3
MAX_FILE_BYTES = int(2.5 * 1024 * 1024)
MAX_TOTAL_BYTES = MAX_FILES * MAX_FILE_BYTES
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".pdf", ".doc", ".docx")

# Rate limit: submissions per IP (and per email) within the window.
RATE_LIMIT_WINDOW = timedelta(hours=1)
RATE_LIMIT_MAX = 5

REFERENCE_ALPHABET = string.ascii_uppercase + string.digits


def _generate_reference() -> str:
    """MSN-XXXX-NNNN, matching the format the design shows on the success screen."""
    block = "".join(secrets.choice(REFERENCE_ALPHABET) for _ in range(4))
    digits = "".join(secrets.choice(string.digits) for _ in range(4))
    return f"MSN-{block}-{digits}"


def _client_ip(request: Request) -> str | None:
    """Caller IP, preferring X-Forwarded-For since we sit behind Cloud Run's proxy.

    The left-most entry is the original client; the proxy appends its own hops.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else None


def _clean(value: str | None, limit: int) -> str | None:
    """Trim and truncate one public-input field. Empty/whitespace becomes None."""
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed[:limit]


def _count_recent(sb, column: str, value: str, cutoff: str) -> int:
    """Rows for one column value inside the rate-limit window.

    Deliberately two separate equality queries rather than one `.or_()` filter:
    the PostgREST `or` filter takes a raw filter string, and interpolating
    user-supplied values into it invites filter injection.
    """
    res = (
        sb.table("contact_submissions")
        .select("id", count="exact")
        .eq(column, value)
        .gte("created_at", cutoff)
        .execute()
    )
    return res.count or 0


def _rate_limited(sb, ip: str | None, email: str) -> bool:
    cutoff = (datetime.now(UTC) - RATE_LIMIT_WINDOW).isoformat()
    try:
        if ip and _count_recent(sb, "client_ip", ip, cutoff) >= RATE_LIMIT_MAX:
            return True
        if _count_recent(sb, "email", email, cutoff) >= RATE_LIMIT_MAX:
            return True
    except Exception:
        # A failing counter must not become an outage on the contact form — the
        # honeypot still applies and the row is still recorded.
        logging.exception("contact rate-limit check failed; allowing request")
        return False
    return False


async def _read_attachments(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Validate then read uploads. Raises HTTPException(400) on any violation."""
    real = [f for f in files if f and f.filename]
    if not real:
        return []
    if len(real) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Attach at most {MAX_FILES} files.")

    out: list[tuple[str, bytes]] = []
    total = 0
    for f in real:
        name = Path(f.filename).name  # strip any directory component
        if not name.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Only images and Word/PDF files are supported.")
        # Check the declared size before reading, so an oversized upload is
        # rejected without pulling its bytes through this process.
        if f.size is not None and f.size > MAX_FILE_BYTES:
            raise HTTPException(status_code=400, detail=f"{name} is over the 2.5 MB limit.")
        content = await f.read()
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=400, detail=f"{name} is over the 2.5 MB limit.")
        total += len(content)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(status_code=400, detail="Attachments exceed the total size limit.")
        out.append((name, content))
    return out


@router.post("/contact-submissions")
async def submit_contact(
    request: Request,
    mode: str = Form(...),
    name: str = Form(...),
    email: EmailStr = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
    product: str | None = Form(None),
    account_email: str | None = Form(None),
    company: str | None = Form(None),
    topic: str | None = Form(None),
    # Honeypot: hidden in the UI, so a human never fills it.
    website: str = Form(""),
    attachments: list[UploadFile] = File(default=[]),
    user_id: str | None = Depends(get_optional_user_id),
) -> dict:
    """Public — anyone can submit. Records in contact_submissions + emails ops."""
    from main import get_supabase_client

    if mode not in MODES:
        raise HTTPException(status_code=400, detail="Invalid mode.")

    # Honeypot tripped: return a plausible success so the bot cannot tell it was
    # filtered, but record and send nothing.
    if website.strip():
        logging.info("contact submission rejected by honeypot")
        return {"ok": True, "reference_id": _generate_reference()}

    clean_name = _clean(name, MAX_NAME)
    clean_subject = _clean(subject, MAX_SUBJECT)
    clean_message = _clean(message, MAX_MESSAGE)
    if not clean_name or not clean_subject or not clean_message:
        raise HTTPException(status_code=400, detail="Name, subject, and message are required.")

    files = await _read_attachments(attachments)

    sb = get_supabase_client()
    ip = _client_ip(request)

    if _rate_limited(sb, ip, str(email)):
        raise HTTPException(
            status_code=429,
            detail="Too many messages from here in the last hour. Please try again later.",
        )

    row = {
        "mode": mode,
        "name": clean_name,
        "email": str(email),
        "subject": clean_subject,
        "message": clean_message,
        # Only the pair belonging to this mode is stored; the other stays NULL.
        "product": _clean(product, MAX_SHORT_FIELD) if mode == "ticket" else None,
        "account_email": _clean(account_email, MAX_SHORT_FIELD) if mode == "ticket" else None,
        "company": _clean(company, MAX_SHORT_FIELD) if mode == "message" else None,
        "topic": _clean(topic, MAX_SHORT_FIELD) if mode == "message" else None,
        "user_id": user_id,
        "attachment_count": len(files),
        "client_ip": ip,
        "status": "new",
    }

    # reference_id is UNIQUE; on the (very unlikely) collision, generate another.
    reference_id = None
    last_error: Exception | None = None
    for _ in range(3):
        candidate = _generate_reference()
        try:
            sb.table("contact_submissions").insert({**row, "reference_id": candidate}).execute()
            reference_id = candidate
            break
        except Exception as e:  # noqa: PERF203 - retry loop is 3 iterations
            last_error = e
            logging.warning("contact_submissions insert attempt failed: %s", e)
    if reference_id is None:
        logging.exception("contact_submissions insert failed: %s", last_error)
        raise HTTPException(status_code=500, detail="Failed to record your message. Please try again.")

    # Best-effort email — failure is logged but does not fail the endpoint
    try:
        _send_ops_notification(row, reference_id, files)
    except Exception:
        logging.exception("Contact submission notification email failed")

    try:
        analytics_capture(user_id or str(email), "contact_submitted", {"mode": mode, "attachments": len(files)})
    except Exception:
        logging.exception("contact_submitted analytics capture failed")

    return {"ok": True, "reference_id": reference_id}


def _send_ops_notification(row: dict, reference_id: str, files: list[tuple[str, bytes]]) -> None:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logging.warning("RESEND_API_KEY not set — skipping contact notification")
        return
    resend.api_key = api_key

    # CONTACT_NOTIFICATION_EMAIL first so contact mail can be pointed at a
    # dedicated support address without moving every other ops notification.
    ops_email = os.getenv("CONTACT_NOTIFICATION_EMAIL") or os.getenv(
        "OPS_NOTIFICATION_EMAIL", "tech@greenboxanalytics.ca"
    )
    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    # Escape user-supplied content — these strings come from anonymous public
    # input and end up in HTML-rendered email.
    def esc(key: str) -> str:
        value = row.get(key)
        return html.escape(str(value)) if value else ""

    is_ticket = row["mode"] == "ticket"
    label = "Support ticket" if is_ticket else "General message"

    detail_rows = [("Reference", reference_id), ("From", f"{esc('name')} &lt;{esc('email')}&gt;")]
    if is_ticket:
        detail_rows.append(("Product", esc("product") or "—"))
        if row.get("account_email"):
            detail_rows.append(("Account email", esc("account_email")))
    else:
        if row.get("company"):
            detail_rows.append(("Company / role", esc("company")))
        detail_rows.append(("Topic", esc("topic") or "—"))
    detail_rows.append(("Submitted by", esc("user_id") or "Logged out"))
    if files:
        detail_rows.append(("Attachments", ", ".join(html.escape(n) for n, _ in files)))

    details = "".join(f"<p><strong>{k}:</strong> {v}</p>" for k, v in detail_rows)
    html_body = f"<h2>{label}: {esc('subject')}</h2>{details}<hr /><p style='white-space:pre-wrap'>{esc('message')}</p>"

    payload: dict = {
        "from": from_address,
        "to": [ops_email],
        "reply_to": row["email"],
        "subject": f"[Msanii][{label}] {row['subject']}",
        "html": html_body,
    }
    if files:
        payload["attachments"] = [
            {"filename": n, "content": base64.b64encode(content).decode("ascii")} for n, content in files
        ]

    resend.Emails.send(payload)
