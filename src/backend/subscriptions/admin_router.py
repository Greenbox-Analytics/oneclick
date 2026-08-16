"""FastAPI router for admin operations. All endpoints depend on require_admin."""

import logging
import sys
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# Ensure backend dir is in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_email, get_current_user_id
from orgs.models import OrgDispersalUpdate
from subscriptions.admin_auth import is_env_admin, is_user_admin, require_admin
from subscriptions.admin_service import AdminService
from subscriptions.models import OverridePayload
from subscriptions.service import EntitlementsService


class CreateTesterGrantRequest(BaseModel):
    email: EmailStr
    # Absolute expiry (legacy). Prefer grant_duration_days; days WINS when both
    # are sent, and days is the only shape a pending (pre-signup) grant stores —
    # duration is applied at CLAIM time so day-29 claims still get full terms.
    expires_at: str | None = None
    grant_duration_days: int | None = Field(None, gt=0, le=3650)
    reason: str = "tester"
    # Initial reserve-credit allocation; None = TESTER_INITIAL_CREDITS default.
    credits: int | None = Field(None, gt=0, le=1_000_000)


class CreditGrantPayload(BaseModel):
    # le ceiling is a fat-finger guard: there's no admin revoke/debit endpoint yet,
    # so an over-grant (e.g. an extra zero) is otherwise uncorrectable in-app.
    amount: int = Field(gt=0, le=1_000_000)
    reason: str
    # Client-supplied stable key per user-initiated grant action so a retry /
    # double-submit dedupes at the RPC; two deliberate grants use different keys.
    idempotency_key: str | None = None


class CreditAdjustPayload(BaseModel):
    # le ceiling mirrors CreditGrantPayload's fat-finger guard.
    amount: int = Field(gt=0, le=1_000_000)
    reason: str
    # REQUIRED (unlike the grant payload's optional key): a double-submitted
    # clawback removes a real customer's credits twice. Support uses the
    # Stripe refund/dispute id they already have in hand at this moment.
    idempotency_key: str = Field(min_length=1)


class OrgPoolGrantPayload(BaseModel):
    amount: int = Field(gt=0, le=1_000_000)
    reason: str
    # REQUIRED, mirroring CreditAdjustPayload: an org grant is big enough that
    # "each call is distinct" (the user-grant default) is the wrong failure
    # mode for a double-submit. The dialog mints a UUID per open.
    idempotency_key: str = Field(min_length=1)


class CreateEnterpriseOrgRequest(BaseModel):
    """POST /admin/orgs body — creates an ENTERPRISE org for an EXISTING
    customer account. See admin_service.create_enterprise_org for why
    `created_by` must be the customer's id, never the operator's."""

    name: str = Field(min_length=1)
    admin_email: EmailStr


class SetOrgKindRequest(BaseModel):
    """PUT /admin/orgs/{org_id}/kind body. `covered_by_user_id` is REQUIRED
    when flipping to 'self_serve' (422 at the router without it) — see
    admin_service.set_org_kind for the coverer/slot requirements."""

    kind: Literal["self_serve", "enterprise"]
    covered_by_user_id: str | None = None


router = APIRouter(prefix="/admin", tags=["Admin"])

# Module-level singleton — one AdminService per FastAPI process.
_admin_service: AdminService | None = None


def _get_admin_service() -> AdminService:
    global _admin_service
    if _admin_service is None:
        from main import get_supabase_client

        sb = get_supabase_client()
        _admin_service = AdminService(sb, EntitlementsService(sb))
    return _admin_service


@router.get("/me")
async def admin_me(
    user_email: str = Depends(get_current_user_email),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Status check — returns whether the caller is an admin.

    Unlike the other endpoints in this router, this one does NOT raise 403
    for non-admins. It's a yes/no probe used by the frontend to decide
    whether to render admin UI; surfacing it as 403 produces console noise
    for every non-admin user on every load. Returning 200 + `isAdmin: false`
    gives the frontend the same information without the noise.

    Admin status is computed server-side from ADMIN_EMAILS (GSM-injected env)
    OR profiles.is_admin (DB-managed). The list itself is never leaked to
    the client — only the boolean result.
    """
    from main import get_supabase_client

    is_admin = is_user_admin(get_supabase_client(), user_email, user_id)
    return {"email": user_email, "isAdmin": is_admin}


@router.get("/users")
async def list_users(
    search: str = "",
    page: int = 1,
    per_page: int = 25,
    _admin: str = Depends(require_admin),
) -> dict:
    return _get_admin_service().list_users(search=search, page=page, per_page=per_page)


@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: str,
    _admin: str = Depends(require_admin),
) -> dict:
    return _get_admin_service().get_user_detail(user_id)


@router.post("/users/{user_id}/grant")
async def grant_pro(
    user_id: str,
    _admin: str = Depends(require_admin),
) -> dict:
    try:
        # Admin grants the ENTRY paid tier (keyed 'basic', labeled "Basic").
        _get_admin_service().set_tier(user_id, "basic")
    except Exception as e:
        msg = str(e).lower()
        if "foreign key" in msg or "violates" in msg:
            raise HTTPException(status_code=400, detail="User not found")
        raise
    return {"ok": True}


@router.post("/users/{user_id}/revoke")
async def revoke_pro(
    user_id: str,
    _admin: str = Depends(require_admin),
) -> dict:
    try:
        _get_admin_service().set_tier(user_id, "free")
    except Exception as e:
        msg = str(e).lower()
        if "foreign key" in msg or "violates" in msg:
            raise HTTPException(status_code=400, detail="User not found")
        raise
    return {"ok": True}


@router.post("/users/{user_id}/promote")
async def promote_user(
    user_id: str,
    _admin: str = Depends(require_admin),
) -> dict:
    """Grant admin privileges to *user_id* via profiles.is_admin = true."""
    _get_admin_service().promote_user(user_id)
    return {"ok": True}


@router.post("/users/{user_id}/demote")
async def demote_user(
    user_id: str,
    caller_id: str = Depends(get_current_user_id),
    _admin: str = Depends(require_admin),
) -> dict:
    """Revoke admin privileges. Blocks self-demote, env-admin demote, and
    fails closed when the target's email can't be verified (so the UI never
    shows a misleading "Demoted" toast for a still-admin user)."""
    if user_id == caller_id:
        raise HTTPException(status_code=400, detail="Cannot demote yourself")
    target_email = _get_admin_service().get_email_for_user_id(user_id)
    if target_email is None:
        raise HTTPException(status_code=400, detail="Could not verify target user — try again")
    if is_env_admin(target_email):
        raise HTTPException(
            status_code=400,
            detail=("Cannot demote env-managed admin — remove from ADMIN_EMAILS instead"),
        )
    _get_admin_service().demote_user(user_id)
    return {"ok": True}


@router.post("/users/{user_id}/recalc-storage")
async def recalc_user_storage(
    user_id: str,
    _admin: str = Depends(require_admin),
):
    """Recompute usage_counters.total_storage_bytes from scratch for `user_id`.

    Calls the Postgres function `recalc_user_storage(p_user_id uuid)` which sums
    file_size across project_files + audio_files joined through projects/artists.
    Useful when the storage trigger has drifted (e.g. user predates the trigger,
    or a manual DB edit bypassed the trigger).

    Returns the freshly-computed total.
    """
    from main import get_supabase_client

    sb = get_supabase_client()
    try:
        sb.rpc("recalc_user_storage", {"p_user_id": user_id}).execute()
    except Exception as e:
        logger.warning("recalc_user_storage RPC failed for %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=f"Recalc failed: {e}")

    # Re-read so the caller (frontend) can show the new value without a separate fetch.
    res = sb.table("usage_counters").select("total_storage_bytes").eq("user_id", user_id).execute()
    rows = res.data or []
    total = int(rows[0]["total_storage_bytes"]) if rows else 0
    return {"user_id": user_id, "total_storage_bytes": total}


@router.post("/users/{user_id}/override")
async def apply_override(
    user_id: str,
    body: OverridePayload,
    _admin: str = Depends(require_admin),
) -> dict:
    _get_admin_service().apply_override(user_id, body.model_dump(exclude_none=True))
    return {"ok": True}


@router.delete("/users/{user_id}/override")
async def clear_override(
    user_id: str,
    _admin: str = Depends(require_admin),
) -> dict:
    _get_admin_service().clear_override(user_id)
    return {"ok": True}


@router.get("/pro-requests")
async def list_pro_requests(
    status: str | None = None,
    _admin: str = Depends(require_admin),
) -> list[dict]:
    return _get_admin_service().list_pro_requests(status=status)


@router.get("/tester-grants")
async def list_tester_grants(
    _admin: str = Depends(require_admin),
) -> list[dict]:
    return _get_admin_service().list_tester_grants()


@router.post("/tester-grants")
async def create_tester_grant(
    body: CreateTesterGrantRequest,
    admin_email: str = Depends(require_admin),
) -> dict:
    # create_tester_grant no longer raises ValueError for any input — an
    # unmatched email parks a pending designation instead of 404ing (and the
    # claimed-row recovery path doesn't raise either). No try/except needed.
    return _get_admin_service().create_tester_grant(
        email=body.email,
        expires_at=body.expires_at,
        reason=body.reason,
        credits=body.credits,
        grant_duration_days=body.grant_duration_days,
        created_by=admin_email,
    )


@router.delete("/tester-grants/pending", status_code=204)
async def revoke_pending_tester_grant(
    email: str,
    _admin: str = Depends(require_admin),
) -> Response:
    """Revoke a pre-signup (unclaimed) tester designation by email."""
    _get_admin_service().revoke_pending_tester_grant(email)
    return Response(status_code=204)


@router.delete("/tester-grants/{user_id}", status_code=204)
async def revoke_tester_grant(
    user_id: str,
    _admin: str = Depends(require_admin),
) -> Response:
    _get_admin_service().revoke_tester_grant(user_id)
    return Response(status_code=204)


@router.post("/users/{target_user_id}/credits/grant")
async def admin_grant_credits(
    target_user_id: str,
    body: CreditGrantPayload,
    admin_email: str = Depends(require_admin),
):
    from main import get_supabase_client
    from subscriptions.admin_service import grant_user_credits

    try:
        result = grant_user_credits(
            get_supabase_client(),
            target_user_id,
            body.amount,
            body.reason,
            admin_email,
            request_id=body.idempotency_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"granted": body.amount, "result": result}


@router.post("/orgs/{org_id}/pool/grant")
async def admin_grant_org_credits(
    org_id: str,
    body: OrgPoolGrantPayload,
    admin_email: str = Depends(require_admin),
):
    """Comped-pack grant into an org pool — reserve bucket, counts toward
    activation. See admin_service.grant_org_credits for semantics."""
    from main import get_supabase_client
    from subscriptions.admin_service import grant_org_credits

    try:
        result = grant_org_credits(
            get_supabase_client(),
            org_id,
            body.amount,
            body.reason,
            admin_email,
            request_id=body.idempotency_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"granted": body.amount, "result": result}


@router.post("/users/{target_user_id}/credits/adjust")
async def admin_adjust_credits(
    target_user_id: str,
    body: CreditAdjustPayload,
    admin_email: str = Depends(require_admin),
):
    """Admin clawback (pack refunds / chargebacks) — see
    admin_service.adjust_user_credits for full semantics. The ONLY sanctioned
    way to remove credits; never a hand-written UPDATE against wallet tables.
    """
    from main import get_supabase_client
    from subscriptions.admin_service import adjust_user_credits

    try:
        result = adjust_user_credits(
            get_supabase_client(),
            target_user_id,
            body.amount,
            body.reason,
            admin_email,
            request_id=body.idempotency_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # Verbatim relay: {removed, shortfall, balance_after} on a fresh
    # clawback, or {duplicate, balance_after} on a replayed idempotency key —
    # whatever the RPC reports, not a massaged shape.
    return {"requested": body.amount, "result": result}


@router.get("/users/{target_user_id}/credits/ledger")
async def admin_credit_ledger(
    target_user_id: str,
    _admin: str = Depends(require_admin),
):
    from main import get_supabase_client
    from subscriptions.admin_service import get_user_credit_ledger

    return {"entries": get_user_credit_ledger(get_supabase_client(), target_user_id)}


# ---------------------------------------------------------------------------
# Platform org suspend/reactivate (Licensing Phase B, spec §4/§5).
#
# Deliberately on THIS router, not orgs/router.py: mounting under /orgs would
# both mangle the path to /orgs/admin/orgs/... AND vanish the moment
# LICENSING_ENABLED is off — which is exactly when a platform admin may need
# to suspend an abusive/non-paying org. These two endpoints are therefore
# flag-INDEPENDENT (no licensing_enabled() gate) and gated only by the
# PLATFORM require_admin already used by every other endpoint in this file.
# `organizations.status='suspended'` confers nothing on its own — entitlement
# resolution already requires status='active' (spec §5) — so suspending an
# org is just flipping that column; there is no separate enforcement path to
# wire up here.
# ---------------------------------------------------------------------------


@router.get("/orgs")
async def list_orgs(_admin: str = Depends(require_admin)):
    """Admin Organizations tab data source. See admin_service.list_orgs_admin."""
    from main import get_supabase_client
    from subscriptions.admin_service import list_orgs_admin

    return {"orgs": list_orgs_admin(get_supabase_client())}


@router.post("/orgs")
async def create_enterprise_org_route(
    body: CreateEnterpriseOrgRequest,
    admin_email: str = Depends(require_admin),
) -> dict:
    """Msanii-admin-only: create an ENTERPRISE org for an existing customer
    account. This (and PUT /admin/orgs/{id}/kind's flip-to-enterprise path)
    are the ONLY producers of kind='enterprise' rows post-migration —
    org creation everywhere else (POST /orgs) is self-serve + slot-gated.
    See admin_service.create_enterprise_org for the created_by rationale
    (review r2 hole 5)."""
    from main import get_supabase_client
    from subscriptions.admin_service import create_enterprise_org

    try:
        result = create_enterprise_org(get_supabase_client(), body.name, body.admin_email)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    logger.info("admin %s created enterprise org %s for customer %s", admin_email, result.get("id"), body.admin_email)
    return result


@router.put("/orgs/{org_id}/kind")
async def set_org_kind_route(
    org_id: str,
    body: SetOrgKindRequest,
    admin_email: str = Depends(require_admin),
) -> dict:
    """Msanii-admin-only: flip an org between 'enterprise' and 'self_serve'.
    See admin_service.set_org_kind for the coverer/slot requirements."""
    if body.kind == "self_serve" and not body.covered_by_user_id:
        raise HTTPException(status_code=422, detail="covered_by_user_id is required when flipping to self_serve")

    from main import get_supabase_client

    sb = get_supabase_client()
    org = sb.table("organizations").select("id").eq("id", org_id).maybe_single().execute()
    if not (org and org.data):
        raise HTTPException(status_code=404, detail="Organization not found")

    from orgs.standing import NoSlotError
    from subscriptions.admin_service import set_org_kind

    try:
        result = set_org_kind(sb, org_id, body.kind, body.covered_by_user_id)
    except NoSlotError as e:
        raise HTTPException(status_code=402, detail={"reason": str(e), "upgradeRequired": True})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info("admin %s flipped org %s kind to %s", admin_email, org_id, body.kind)
    return result


@router.put("/orgs/{org_id}/dispersal")
async def set_org_dispersal(
    org_id: str,
    body: OrgDispersalUpdate,
    _admin: str = Depends(require_admin),
) -> dict:
    """Set an org's monthly credit dispersal — the contract volume the sweep
    adds to its pool each period.

    MSANII ADMIN ONLY, and that placement is load-bearing rather than tidiness:
    nothing in the app collects payment for a dispersal (there is no org
    subscription object), any signed-in user may create an org and is auto-made
    its admin, and `wallets.cumulative_paid_in` counts dispersed credits toward
    the activation floor. On an org-admin route those three facts compose into
    "mint yourself unlimited credits and self-activate". An operator setting it
    IS the commercial agreement, so here it's the same statement of fact the
    signed contract is.

    Takes effect at the next period boundary — the sweep is the only writer of
    dispersal credits, which is what keeps its once-per-month idempotency honest.
    """
    from main import get_supabase_client
    from orgs import service as orgs_service

    sb = get_supabase_client()
    org = sb.table("organizations").select("id").eq("id", org_id).maybe_single().execute()
    if not (org and org.data):
        raise HTTPException(status_code=404, detail="Organization not found")
    try:
        return await orgs_service.set_org_dispersal(sb, org_id, body.monthly_dispersal_credits)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/orgs/{org_id}/suspend")
async def suspend_org(
    org_id: str,
    _admin: str = Depends(require_admin),
) -> dict:
    from main import get_supabase_client

    sb = get_supabase_client()
    res = sb.table("organizations").select("id, status").eq("id", org_id).maybe_single().execute()
    org = res.data if res else None
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if org.get("status") != "active":
        raise HTTPException(status_code=409, detail="Only an active org can be suspended")
    updated = sb.table("organizations").update({"status": "suspended"}).eq("id", org_id).execute()
    return updated.data[0] if updated.data else {"id": org_id, "status": "suspended"}


@router.post("/orgs/{org_id}/reactivate")
async def reactivate_org(
    org_id: str,
    _admin: str = Depends(require_admin),
) -> dict:
    from main import get_supabase_client

    sb = get_supabase_client()
    res = sb.table("organizations").select("id, status").eq("id", org_id).maybe_single().execute()
    org = res.data if res else None
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    status = org.get("status")
    if status == "pending":
        raise HTTPException(status_code=409, detail="Org has not been activated yet")
    if status != "suspended":
        raise HTTPException(status_code=409, detail="Only a suspended org can be reactivated")
    updated = sb.table("organizations").update({"status": "active"}).eq("id", org_id).execute()
    return updated.data[0] if updated.data else {"id": org_id, "status": "active"}


# ---------------------------------------------------------------------------
# Archived/live org pool disposition tooling (follow-ups plan 2026-07-22,
# Task 2). Same placement rationale as suspend/reactivate directly above:
# PLATFORM require_admin, flag-INDEPENDENT (no licensing_enabled() gate) —
# support needs to dispose of a pool precisely in cases where licensing may
# be off/broken, not only in the steady state. Deliberately NO analytics on
# the clawback (admin support action, not user behavior — same stance as the
# per-user clawback endpoint above).
# ---------------------------------------------------------------------------


@router.post("/orgs/{org_id}/pool/clawback")
async def clawback_org_pool(
    org_id: str,
    body: CreditAdjustPayload,
    admin_email: str = Depends(require_admin),
):
    """Admin clawback on an org's pool wallet — see
    admin_service.adjust_org_pool for the full runbook (refund pair /
    goodwill-migration pair) and the reserve-only completeness invariant.
    The ONLY sanctioned way to remove credits from an org pool; never a
    hand-written UPDATE against wallet tables.
    """
    from main import get_supabase_client
    from subscriptions.admin_service import adjust_org_pool

    try:
        result = adjust_org_pool(
            get_supabase_client(),
            org_id,
            body.amount,
            body.reason,
            admin_email,
            request_id=body.idempotency_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # Verbatim relay: {removed, shortfall, balance_after} on a fresh
    # clawback, or {duplicate, balance_after} on a replayed idempotency key —
    # whatever the RPC reports, not a massaged shape.
    return result


@router.get("/orgs/{org_id}/pool")
async def get_org_pool(
    org_id: str,
    _admin: str = Depends(require_admin),
):
    """Support visibility into an org's pool BEFORE disposing of it — see
    admin_service.get_org_pool for the full shape/rationale.
    """
    from main import get_supabase_client
    from subscriptions.admin_service import get_org_pool as _get_org_pool

    try:
        return _get_org_pool(get_supabase_client(), org_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
