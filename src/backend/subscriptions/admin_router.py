"""FastAPI router for admin operations. All endpoints depend on require_admin."""

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# Ensure backend dir is in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_email, get_current_user_id
from subscriptions.admin_auth import is_env_admin, is_user_admin, require_admin
from subscriptions.admin_service import AdminService
from subscriptions.models import OverridePayload
from subscriptions.service import EntitlementsService


class CreateTesterGrantRequest(BaseModel):
    email: EmailStr
    expires_at: str | None = None
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
        _get_admin_service().set_tier(user_id, "pro")
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
    _admin: str = Depends(require_admin),
) -> dict:
    try:
        return _get_admin_service().create_tester_grant(
            email=body.email,
            expires_at=body.expires_at,
            reason=body.reason,
            credits=body.credits,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


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


@router.get("/users/{target_user_id}/credits/ledger")
async def admin_credit_ledger(
    target_user_id: str,
    _admin: str = Depends(require_admin),
):
    from main import get_supabase_client
    from subscriptions.admin_service import get_user_credit_ledger

    return {"entries": get_user_credit_ledger(get_supabase_client(), target_user_id)}
