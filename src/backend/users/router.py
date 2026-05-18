"""User-scoped endpoints (welcome email, future profile actions)."""

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from analytics import capture as analytics_capture
from analytics import identify as analytics_identify
from auth import get_current_user_id
from subscriptions.admin_auth import is_active_tester_row, is_user_admin
from users.emails import send_welcome_email

router = APIRouter()


def _get_supabase() -> Client:
    from main import get_supabase_client

    return get_supabase_client()


def _fire_signup_analytics(db: Client, user_id: str, email: str, profile_row: dict) -> None:
    """Fire signup_completed + identify exactly once on first welcome.

    Called only when welcome_email_sent_at transitions from null → now. Looks
    up `plan`, `is_admin`, and `is_tester` so the PostHog person profile is
    seeded with the full property set the dashboards depend on.

    `email` must be passed by the caller (resolved via auth.admin.get_user_by_id)
    because profiles.email does not exist — email lives only on auth.users.
    See migration 20260329000000_create_rights_registry.sql line 400.

    Uses direct imports of analytics and admin helpers so test monkeypatching
    (`users.router.analytics_capture`) works and analytics stays disabled
    (no-ops) when POSTHOG_ENABLED != "true".
    """
    try:
        sub_res = db.table("subscriptions").select("tier").eq("user_id", user_id).execute()
        plan = (sub_res.data or [{}])[0].get("tier", "free") if sub_res.data else "free"
    except Exception as exc:
        print(f"Warning: signup analytics — plan lookup failed: {exc}")
        plan = "free"

    try:
        overrides = (
            db.table("tier_overrides")
            .select("reason, expires_at")
            .eq("user_id", user_id)
            .like("reason", "tester%")
            .execute()
        )
        is_tester = any(is_active_tester_row(r) for r in (overrides.data or []))
    except Exception as exc:
        print(f"Warning: signup analytics — tester lookup failed: {exc}")
        is_tester = False

    try:
        is_admin = is_user_admin(db, email, user_id)
    except Exception as exc:
        print(f"Warning: signup analytics — admin lookup failed: {exc}")
        is_admin = False

    analytics_capture(user_id, "signup_completed", {})
    analytics_identify(
        user_id,
        {
            "email": email,
            "signed_up_at": profile_row.get("created_at"),
            "plan": plan,
            "role": profile_row.get("role"),
            "is_admin": is_admin,
            "is_tester": is_tester,
        },
    )


@router.post("/welcome")
async def send_welcome(user_id: str = Depends(get_current_user_id)):
    """Send the one-time welcome email and fire signup analytics.

    Idempotent: a profile with `welcome_email_sent_at` already set is a no-op
    for both the email and the analytics events. Safe to call on every
    SIGNED_IN event from the frontend.
    """
    db = _get_supabase()

    # NOTE on profiles columns: profiles does NOT store email (see migration
    # 20260329000000_create_rights_registry.sql line 400) and does NOT store
    # created_at in some environments (the schema has updated_at only). The
    # `role` column is also conditional — see the IF EXISTS guard in
    # 20260420000000_rename_industry_to_role.sql. Read only the columns we
    # know are universal here; resolve email + created_at from auth.users
    # below; tolerate role missing.
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

    # Fetch the verified email + signup timestamp from auth.users via the
    # service-role admin client (canonical source for both).
    try:
        auth_user = db.auth.admin.get_user_by_id(user_id)
        recipient_email = auth_user.user.email if auth_user and auth_user.user else None
        signed_up_at = getattr(auth_user.user, "created_at", None) if auth_user and auth_user.user else None
        if hasattr(signed_up_at, "isoformat"):
            signed_up_at = signed_up_at.isoformat()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not resolve user email: {exc}") from exc

    if not recipient_email:
        raise HTTPException(status_code=500, detail="Could not resolve user email")

    p = profile.data
    first_name = p.get("first_name") or p.get("given_name") or ((p.get("full_name") or "").split(" ", 1)[0] or None)

    # Tolerant role read — column may not exist in this env.
    try:
        role_res = db.table("profiles").select("role").eq("id", user_id).maybe_single().execute()
        role = (role_res.data or {}).get("role") if role_res else None
    except Exception:
        role = None
    # Stash on `p` so _fire_signup_analytics can read it from the dict (keeps
    # signature stable with `created_at` also coming from p).
    p["role"] = role
    p["created_at"] = signed_up_at

    result = send_welcome_email(recipient_email, first_name)
    if result is None:
        # Don't mark as sent if delivery failed — the next SIGNED_IN will retry.
        raise HTTPException(status_code=502, detail="Failed to send welcome email")

    db.table("profiles").update({"welcome_email_sent_at": "now()"}).eq("id", user_id).execute()

    # Fire signup analytics exactly once per user — gated by the same
    # welcome_email_sent_at column we just transitioned from null → now.
    _fire_signup_analytics(db, user_id, recipient_email, p)

    return {"sent": True}
