"""Entitlements API: surfaces what the current user can do.

Single endpoint used by the frontend's useEntitlements hook (Sub-project 1)
and the Usage tab + paywall components (Sub-project 3).

No /refresh endpoint — frontend calls queryClient.invalidateQueries(['entitlements']).
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

# Ensure backend dir is in path (matches the pattern in boards/router.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_email, get_current_user_id
from subscriptions.admin_auth import is_user_admin
from subscriptions.deps import _get_entitlements_service

router = APIRouter()


@router.get("/me/entitlements")
async def get_my_entitlements(
    user_id: str = Depends(get_current_user_id),
    user_email: str = Depends(get_current_user_email),
):
    """Return the current user's merged entitlements (tier + caps + features + usage).

    Admin users (ADMIN_EMAILS env OR profiles.is_admin=true) receive Pro-shaped
    entitlements regardless of subscription tier. When BYPASS_PAYWALLS=true, all
    users get Pro-shaped entitlements (handled inside get_for_user_safe).
    """
    from main import get_supabase_client

    is_admin = is_user_admin(get_supabase_client(), user_email, user_id)
    ent = _get_entitlements_service().get_for_user_safe(user_id, is_admin=is_admin)
    return ent.to_dict()


@router.get("/me/credits/usage")
async def get_my_credit_usage(user_id: str = Depends(get_current_user_id)):
    """Per-tool credit spend for the current period (Account & Billing usage view).

    Returns {"enabled": false} when CREDITS_ENABLED is off — the frontend hides
    the credit surfaces in that case.
    """
    return _get_entitlements_service().get_credit_usage_safe(user_id)


class BillingPrefsPayload(BaseModel):
    """Sparse update — only provided fields are written."""

    overage_enabled: bool | None = None
    overage_cap_credits: int | None = Field(None, ge=0)
    storage_overage_enabled: bool | None = None


@router.post("/me/billing-prefs")
async def set_billing_prefs(
    body: BillingPrefsPayload,
    user_id: str = Depends(get_current_user_id),
):
    """Opt in/out of pay-per-use overage (spec §4 — always a prompt, never silent).

    Free tier is never offered credit overage: enabling it here is a 400.
    """
    from fastapi import HTTPException

    from main import get_supabase_client

    update = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update:
        raise HTTPException(status_code=400, detail="No preferences provided.")

    sb = get_supabase_client()
    sub_res = sb.table("subscriptions").select("tier").eq("user_id", user_id).execute()
    tier = sub_res.data[0]["tier"] if sub_res.data else "free"

    if body.overage_enabled and tier == "free":
        raise HTTPException(
            status_code=400,
            detail="Pay-per-use is available on Pro plans. Upgrade to keep going past your included credits.",
        )

    # Upsert so a missing subscriptions row is created instead of silently
    # no-oping (an update().eq() matching zero rows persists nothing but still
    # returns 200). The insert path needs no extra fields beyond user_id + the
    # provided prefs: the schema gives tier/status NOT NULL DEFAULTs
    # ('free'/'active', 20260509000001_subscription_foundation.sql) and the
    # overage columns NOT NULL DEFAULT false / nullable cap
    # (20260713000002_credits_schema.sql).
    result = sb.table("subscriptions").upsert({**update, "user_id": user_id}, on_conflict="user_id").execute()
    # Respond with what was actually persisted, not an echo of the request —
    # with sparse updates the request alone can't describe the row's state.
    row = result.data[0] if result.data else {}

    if body.overage_enabled is not None or body.storage_overage_enabled is not None:
        analytics_capture(
            user_id,
            "overage_optin_changed",
            {
                "enabled": body.overage_enabled,
                "storage_enabled": body.storage_overage_enabled,
                "cap": body.overage_cap_credits,
            },
        )
    return {
        "overageEnabled": row.get("overage_enabled"),
        "overageCapCredits": row.get("overage_cap_credits"),
        "storageOverageEnabled": row.get("storage_overage_enabled"),
    }
