"""Org-admin (JWT) partner-key management — the portal surface.

Mounted at /orgs on the PRODUCT backend. Gated on the DB column
organizations.partner_api_enabled, NOT on the PARTNER_API_ENABLED env flag:
that flag marks the partner HOST, whose lockdown middleware 404s this router
before it is reached.
"""

from fastapi import APIRouter, Depends, HTTPException

from analytics import capture as analytics_capture
from auth import get_current_user_id
from orgs import authz
from orgs import service as orgs_service
from partner_api import service as psvc
from partner_api.models import KeyFolderAssign, KeyFolderCreate, PartnerKeyCreate
from subscriptions.service import credits_enabled, licensing_enabled


def require_org_key_surface() -> None:
    """Keys spend an org pool, so both flags are load-bearing. 404 = true
    rollback."""
    if not (credits_enabled() and licensing_enabled()):
        raise HTTPException(status_code=404, detail="Not found")


router = APIRouter(dependencies=[Depends(require_org_key_surface)])


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


def _partner_api_enabled(sb, org_id: str) -> bool:
    """Own read, not a column on the shared `_first_org` select: widening that
    would 500 every org lifecycle guard until the migration lands."""
    res = sb.table("organizations").select("partner_api_enabled").eq("id", org_id).execute()
    return bool(res.data and res.data[0].get("partner_api_enabled"))


def _gate(sb, user_id: str, org_id: str, *, mutating: bool) -> dict:
    """Authz FIRST, so a non-admin learns nothing about the org (existence,
    capability, lifecycle). Then the capability bit, then lifecycle for writes
    only — reads on an archived org still show what keys existed."""
    authz.require_admin(sb, user_id, org_id)
    org = orgs_service._first_org(sb, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Not found")
    if not _partner_api_enabled(sb, org_id):
        raise HTTPException(status_code=403, detail={"code": "access_disabled"})
    if mutating:
        orgs_service._require_live_org(org)
    return org


@router.get("/{org_id}/partner-keys")
async def list_org_partner_keys(org_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    """The org's listed keys plus its folders. list_keys selects explicit
    columns, so secrets and hashes can't appear."""
    sb = _get_supabase()
    _gate(sb, user_id, org_id, mutating=False)
    return psvc.key_console(sb, org_id)


@router.post("/{org_id}/partner-keys")
async def create_org_partner_key(
    org_id: str, body: PartnerKeyCreate, user_id: str = Depends(get_current_user_id)
) -> dict:
    """Same body as the Msanii-admin mint. The response carries the plaintext
    secret EXACTLY ONCE."""
    sb = _get_supabase()
    _gate(sb, user_id, org_id, mutating=True)
    try:
        key = psvc.mint_key(
            sb,
            org_id,
            label=body.label,
            created_by=user_id,
            expires_at=body.expires_at.isoformat() if body.expires_at else None,
            folder_id=body.folder_id,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail={"code": "unknown_folder"})
    analytics_capture(user_id, "partner_key_created", {"org_id": org_id, "via": "portal"})
    return key


@router.post("/{org_id}/partner-key-folders", status_code=201)
async def create_org_key_folder(
    org_id: str, body: KeyFolderCreate, user_id: str = Depends(get_current_user_id)
) -> dict:
    """Idempotent on the name; only a real insert is worth an event."""
    sb = _get_supabase()
    _gate(sb, user_id, org_id, mutating=True)
    try:
        folder = psvc.create_folder(sb, org_id, body.name)
    except ValueError:
        raise HTTPException(status_code=422, detail={"code": "invalid_folder_name"})
    if folder.pop("created", False):
        analytics_capture(user_id, "partner_key_folder_created", {"org_id": org_id, "via": "portal"})
    return folder


@router.put("/{org_id}/partner-keys/{key_id}/folder")
async def set_org_key_folder(
    org_id: str, key_id: str, body: KeyFolderAssign, user_id: str = Depends(get_current_user_id)
) -> dict:
    """Move a key between folders (null = unfile). Spend follows the key's
    CURRENT folder, so this moves its history too. The folder is re-read only
    to tell "unknown folder" (422) from "not your key" (404)."""
    sb = _get_supabase()
    _gate(sb, user_id, org_id, mutating=True)
    if not psvc.set_key_folder(sb, org_id, key_id, body.folder_id):
        if body.folder_id is not None and not psvc.folder_belongs(sb, org_id, body.folder_id):
            raise HTTPException(status_code=422, detail={"code": "unknown_folder"})
        raise HTTPException(status_code=404, detail="Key not found")
    return {"ok": True}


@router.delete("/{org_id}/partner-keys/{key_id}")
async def revoke_org_partner_key(org_id: str, key_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    """revoke_key scopes on org_id and reports no match as False — a foreign
    id, or another admin winning the race — so False IS the 404."""
    sb = _get_supabase()
    _gate(sb, user_id, org_id, mutating=True)
    if not psvc.revoke_key(sb, org_id, key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    analytics_capture(user_id, "partner_key_revoked", {"org_id": org_id, "via": "portal"})
    return {"status": "revoked"}
