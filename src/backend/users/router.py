"""User-scoped endpoints (welcome email, future profile actions)."""

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from auth import get_current_user_id
from users.emails import send_welcome_email

router = APIRouter()


def _get_supabase() -> Client:
    from main import get_supabase_client

    return get_supabase_client()


@router.post("/welcome")
async def send_welcome(user_id: str = Depends(get_current_user_id)):
    """Send the one-time welcome email to a freshly-signed-up user.

    Idempotent: a profile with `welcome_email_sent_at` already set is a no-op.
    Safe to call on every SIGNED_IN event from the frontend.
    """
    db = _get_supabase()

    profile = (
        db.table("profiles")
        .select("welcome_email_sent_at, first_name, given_name, full_name")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    if not profile or not profile.data:
        raise HTTPException(status_code=404, detail="Profile not found")
    if profile.data.get("welcome_email_sent_at"):
        return {"sent": False, "reason": "already_sent"}

    # Fetch the verified email from auth.users via the service-role admin client.
    try:
        auth_user = db.auth.admin.get_user_by_id(user_id)
        recipient_email = auth_user.user.email if auth_user and auth_user.user else None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not resolve user email: {exc}") from exc

    if not recipient_email:
        raise HTTPException(status_code=500, detail="Could not resolve user email")

    p = profile.data
    first_name = p.get("first_name") or p.get("given_name") or ((p.get("full_name") or "").split(" ", 1)[0] or None)

    result = send_welcome_email(recipient_email, first_name)
    if result is None:
        # Don't mark as sent if delivery failed — the next SIGNED_IN will retry.
        raise HTTPException(status_code=502, detail="Failed to send welcome email")

    db.table("profiles").update({"welcome_email_sent_at": "now()"}).eq("id", user_id).execute()
    return {"sent": True}
