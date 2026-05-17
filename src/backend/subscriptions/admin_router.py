"""FastAPI router for admin operations. All endpoints depend on require_admin."""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr

# Ensure backend dir is in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from subscriptions.admin_auth import is_env_admin, require_admin
from subscriptions.admin_service import AdminService
from subscriptions.models import OverridePayload
from subscriptions.service import EntitlementsService


class CreateTesterGrantRequest(BaseModel):
    email: EmailStr
    expires_at: str | None = None
    reason: str = "tester"


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
async def admin_me(email: str = Depends(require_admin)) -> dict:
    return {"email": email, "isAdmin": True}


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
