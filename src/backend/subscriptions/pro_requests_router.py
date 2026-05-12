"""Public POST /pro-requests endpoint — captures upgrade interest.

Records a row in pro_requests AND emails ops via Resend (best-effort).
DB record is the durable source of truth; Resend failure does not fail the request.

NOTE: this endpoint depends on `get_supabase_client()` returning a service-role
client (which bypasses RLS). The pro_requests table has SELECT USING (false)
deny-all and no INSERT policy — anonymous JS-client INSERTs from the browser
would also be blocked by RLS, which is intentional. With the service role key
(VITE_SUPABASE_SECRET_KEY), this endpoint inserts successfully. If deployed
with only an anon key, INSERT silently fails — Task 12's integration test
test_insert_blocked_by_rls_for_anon verifies the protection.
"""

import html
import logging
import os
import sys
from pathlib import Path

import resend
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

# Ensure backend dir is in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_optional_user_id

router = APIRouter(tags=["Pro Requests"])


class ProRequestPayload(BaseModel):
    email: EmailStr
    message: str | None = None


@router.post("/pro-requests")
async def submit_pro_request(
    body: ProRequestPayload,
    user_id: str | None = Depends(get_optional_user_id),
) -> dict:
    """Public — anyone can submit. Records in pro_requests + emails ops."""
    from main import get_supabase_client

    sb = get_supabase_client()

    try:
        sb.table("pro_requests").insert(
            {
                "email": body.email,
                "message": body.message,
                "user_id": user_id,
                "status": "new",
            }
        ).execute()
    except Exception as e:
        logging.exception("pro_requests insert failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to record request")

    # Best-effort email — failure is logged but does not fail the endpoint
    try:
        _send_ops_notification(body.email, body.message, user_id)
    except Exception:
        logging.exception("Pro request notification email failed")

    return {"ok": True}


def _send_ops_notification(email: str, message: str | None, user_id: str | None) -> None:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logging.warning("RESEND_API_KEY not set — skipping ops notification")
        return
    resend.api_key = api_key

    ops_email = os.getenv("OPS_NOTIFICATION_EMAIL", "tech@greenboxanalytics.ca")
    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    # Escape user-supplied content — these strings come from anonymous public
    # input and end up in HTML-rendered email. Even though blast radius is just
    # the ops inbox, escape is a one-line free win.
    safe_email = html.escape(email)
    safe_message = html.escape(message) if message else ""
    safe_user_id = html.escape(user_id) if user_id else ""

    user_line = f"User ID: {safe_user_id}" if user_id else "Submitted while logged out"
    message_block = f"<p><strong>Message:</strong> {safe_message}</p>" if message else ""

    html_body = (
        f"<h2>New Pro access request</h2>"
        f"<p><strong>Email:</strong> {safe_email}</p>"
        f"<p>{user_line}</p>"
        f"{message_block}"
        f"<p><em>View in admin: /admin/pro-requests</em></p>"
    )

    resend.Emails.send(
        {
            "from": from_address,
            "to": [ops_email],
            "subject": "[Msanii] New Pro access request",
            "html": html_body,
        }
    )
