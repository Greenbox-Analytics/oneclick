"""Entitlements API: surfaces what the current user can do.

Single endpoint used by the frontend's useEntitlements hook (Sub-project 1)
and the Usage tab + paywall components (Sub-project 3).

No /refresh endpoint — frontend calls queryClient.invalidateQueries(['entitlements']).
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure backend dir is in path (matches the pattern in boards/router.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_email, get_current_user_id
from subscriptions.admin_auth import is_user_admin
from subscriptions.deps import _get_entitlements_service
from subscriptions.service import EntitlementsService, credits_enabled, licensing_enabled

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


async def _my_api_usage(user_id: str, range_: str) -> dict:
    """The /me/api-usage payload — shared by the JSON endpoint and its PDF twin
    so the two can never disagree about the same window.

    Partner-API spend through the caller's OWN keys, in every org where they
    hold an ACTIVE seat.

    MY usage, like /me/credits/usage: the rollup is scoped to keys the caller
    created (`only_created_by`), so a member sees their own keys and never a
    colleague's — the org-wide view stays admin-only on GET /orgs/{id}/usage.
    `credits`/`runs` sum the returned byKey rows, so a hidden (long-inactive)
    key's spend is not counted here even though it still counts for the org.
    """
    if not (credits_enabled() and licensing_enabled()):
        return {"range": range_, "orgs": []}

    from main import get_supabase_client
    from orgs import service as orgs_service

    if range_ not in orgs_service.USAGE_RANGES:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_range", "message": f"range must be one of {', '.join(orgs_service.USAGE_RANGES)}"},
        )
    sb = get_supabase_client()
    memberships = (
        sb.table("org_members").select("org_id").eq("user_id", user_id).eq("status", "active").execute().data or []
    )
    org_ids = [m["org_id"] for m in memberships if m.get("org_id")]
    if not org_ids:
        return {"range": range_, "orgs": []}
    names = {
        o["id"]: o.get("name")
        for o in (sb.table("organizations").select("id, name").in_("id", org_ids).execute().data or [])
    }

    orgs = []
    for org_id in org_ids:
        payload = await orgs_service.org_usage_rollup(sb, org_id, range_=range_, only_created_by=user_id)
        by_key = payload["byKey"]
        if not by_key:
            continue  # no keys of mine here — nothing to show
        orgs.append(
            {
                "orgId": org_id,
                "orgName": names.get(org_id),
                "since": payload["since"],
                "credits": sum(k["credits"] for k in by_key),
                "runs": sum(k["runs"] for k in by_key),
                "byKey": by_key,
                "byFolder": payload["byFolder"],
            }
        )
    return {"range": range_, "orgs": orgs}


@router.get("/me/api-usage")
async def get_my_api_usage(
    range: str = Query("mtd", description="mtd | 7d | 14d | 1y | all"),
    user_id: str = Depends(get_current_user_id),
):
    """Partner-API spend through the caller's OWN keys — see _my_api_usage."""
    return await _my_api_usage(user_id, range)


@router.get("/me/api-usage/report.pdf")
async def get_my_api_usage_report(
    range: str = Query("mtd", description="mtd | 7d | 14d | 1y | all"),
    user_id: str = Depends(get_current_user_id),
):
    """The same data as GET /me/api-usage, as a downloadable PDF: one section
    per org (its own chart, keys and folders) and no member table — a plain
    member's own keys are all this endpoint ever sees. Flags off renders the
    empty report rather than 404ing, matching the JSON's empty `orgs`."""
    from orgs import usage_report

    data = await _my_api_usage(user_id, range)
    orgs = data["orgs"]
    sections = [
        {
            "heading": org.get("orgName") or "Team",
            "series": usage_report.merge_series(k.get("series") for k in org["byKey"]),
            "seats": None,
            "by_key": org["byKey"],
            "by_folder": org["byFolder"],
        }
        for org in orgs
    ]
    pdf = usage_report.render_usage_report(
        title="My API usage",
        subtitle="Usage report",
        range_=data["range"],
        # ponytail: one floor for every section. Two orgs on different billing
        # periods gap-fill their charts from the first org's; per-section
        # `since` if that ever misleads someone.
        since=orgs[0]["since"] if orgs else None,
        series=usage_report.merge_series(s["series"] for s in sections),
        seats=None,
        by_key=[],
        by_folder=[],
        sections=sections,
    )
    return usage_report.pdf_response(pdf, "my-api-usage", range)


class BillingContextPayload(BaseModel):
    """PUT /me/billing-context body. `orgId=null` means switch to personal."""

    orgId: str | None = None


@router.put("/me/billing-context")
async def set_billing_context(
    body: BillingContextPayload,
    user_id: str = Depends(get_current_user_id),
):
    """Set the caller's billing context (spec §5). `orgId=null` → personal (always
    allowed, 200). An org id is accepted ONLY when the caller holds an ACTIVE seat
    in a non-archived org whose status is 'active' or 'pending' — a PENDING org is
    accepted (rule 7: confers nothing until activation, but the preference must
    survive the onboarding order).

    404 with an IDENTICAL body for a nonexistent org, no seat, an archived org, and
    a suspended org — there is NO existence oracle (rule 7): setting the context to
    an arbitrary org id must never reveal whether that org exists.
    `billing_context_org_id` is validated again at every entitlements read
    (_resolve_context), so this write confers nothing on its own.
    """
    from main import get_supabase_client

    sb = get_supabase_client()

    if body.orgId is None:
        sb.table("profiles").update({"billing_context_org_id": None}).eq("id", user_id).execute()
        return {"context": "personal"}

    org_id = body.orgId
    not_found = HTTPException(status_code=404, detail="Organization not found")

    seat = EntitlementsService._first_row(
        sb.table("org_members")
        .select("id, status")
        .eq("org_id", org_id)
        .eq("user_id", user_id)
        .eq("status", "active")
        .execute()
    )
    if not seat:
        raise not_found

    org = EntitlementsService._first_row(
        sb.table("organizations").select("id, status, archived_at").eq("id", org_id).execute()
    )
    if not org or org.get("archived_at") is not None or org.get("status") not in ("active", "pending"):
        raise not_found

    sb.table("profiles").update({"billing_context_org_id": org_id}).eq("id", user_id).execute()
    return {"context": "org", "orgId": org_id}


class BillingPrefsPayload(BaseModel):
    """Sparse update — only provided fields are written."""

    overage_enabled: bool | None = None
    overage_cap_credits: int | None = Field(None, ge=0)


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
    # overage column NOT NULL DEFAULT false / nullable cap
    # (20260713000002_credits_schema.sql).
    result = sb.table("subscriptions").upsert({**update, "user_id": user_id}, on_conflict="user_id").execute()
    # Respond with what was actually persisted, not an echo of the request —
    # with sparse updates the request alone can't describe the row's state.
    row = result.data[0] if result.data else {}

    if body.overage_enabled is not None:
        analytics_capture(
            user_id,
            "overage_optin_changed",
            {"enabled": body.overage_enabled, "cap": body.overage_cap_credits},
        )
    return {
        "overageEnabled": row.get("overage_enabled"),
        "overageCapCredits": row.get("overage_cap_credits"),
    }
