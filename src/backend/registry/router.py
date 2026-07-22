"""FastAPI router for the Rights & Ownership Registry."""

import re
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from postgrest.exceptions import APIError

from analytics import capture as analytics_capture
from auth import get_current_user_id
from registry import contract_splits, derive_service, grants_service, service, work_links_service
from registry.access import get_work_access
from registry.models import (
    AccessLevelUpdate,
    AgreementCreate,
    CollaboratorInvite,
    CollaboratorInviteWithStakes,
    DeriveFromContractsBody,
    FolderCreate,
    FolderUpdate,
    GrantsBody,
    LicenseCreate,
    LicenseUpdate,
    NoteCreate,
    NoteUpdate,
    ProjectAboutUpdate,
    StakeCreate,
    StakeUpdate,
    TeamCardUpdate,
    WorkCreate,
    WorkRoleUpdate,
    WorkUpdate,
)
from subscriptions.enforcement import free_credit_grant, gated_create, gated_credits, gated_feature
from subscriptions.models import Action, CreditAction
from subscriptions.service import credits_enabled
from utils.llm.tracking import set_llm_context

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


# ============================================================
# Works
# ============================================================


@router.get("/works")
async def list_works(
    user_id: str = Depends(get_current_user_id),
    artist_id: str | None = Query(None),
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    result = await service.get_works(_get_supabase(), user_id, artist_id, page, page_size)
    if isinstance(result, list):
        return {"works": result}
    return result


@router.get("/works/my-collaborations")
async def list_my_collaborations(
    user_id: str = Depends(get_current_user_id),
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    result = await service.get_works_as_collaborator(_get_supabase(), user_id, page, page_size)
    if isinstance(result, list):
        return {"works": result}
    return result


@router.get("/works/by-project/{project_id}")
async def list_works_by_project(project_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        works = await service.get_works_by_project(_get_supabase(), user_id, project_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"works": works}


@router.get("/works/{work_id}")
async def get_work(work_id: str, user_id: str = Depends(get_current_user_id)):
    # 404 for both missing and no-access — don't leak existence.
    work = await service.get_work_filtered(_get_supabase(), user_id, work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    return work


@router.get("/works/{work_id}/full")
async def get_work_full(work_id: str, user_id: str = Depends(get_current_user_id)):
    data = await service.get_work_full(_get_supabase(), user_id, work_id)
    if not data:
        raise HTTPException(status_code=404, detail="Work not found")
    return data


def _raise_503_if_schema_stale(e: APIError) -> None:
    """Turn PostgREST's opaque schema-cache miss into actionable feedback.
    Raised when a migration has been written to repo but not yet applied to the
    DB (or applied, but PostgREST hasn't reloaded its schema cache yet)."""
    if getattr(e, "code", None) == "PGRST204":
        raise HTTPException(
            status_code=503,
            detail="Database schema is out of date — please run pending Supabase migrations and try again.",
        )


@router.post("/works")
async def create_work(body: WorkCreate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    # max_works ships with the credits launch (spec: everything behind
    # CREDITS_ENABLED); the migration seeds the cap but it must not bite
    # before the flag flips.
    if credits_enabled():
        # SP3/credits: per-tier max_works cap — 402 when the user is at the limit.
        count_res = _get_supabase().table("works_registry").select("id", count="exact").eq("user_id", user_id).execute()
        # Licensing Phase C (rule 9): pass the target project so a work created in
        # an org-linked project where the caller holds a seat gets the org's
        # unlimited count cap. body.project_id is already in scope (required on
        # WorkCreate); no new query. Ownership is validated downstream in
        # service.create_work — derivation can only ever UPGRADE the cap, so it is
        # safe even before that check (a miss falls through to today's behavior).
        gated_create(user_id, "work", count_res.count or 0, resource_project_id=body.project_id)
    data = body.model_dump(exclude_none=True)
    if "release_date" in data and data["release_date"]:
        data["release_date"] = data["release_date"].isoformat()
    try:
        work = await service.create_work(_get_supabase(), user_id, data)
    except APIError as e:
        _raise_503_if_schema_stale(e)
        raise
    if not work:
        raise HTTPException(status_code=500, detail="Failed to create work")
    analytics_capture(user_id, "work_created", {"work_type": data.get("work_type", "unknown")})
    return work


@router.put("/works/{work_id}")
async def update_work(work_id: str, body: WorkUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    # exclude_unset (not exclude_none) so callers can explicitly null out a
    # field — needed for unreleasing a track (release_date = null), unsetting
    # an ISRC, etc. exclude_none would silently strip those nulls.
    data = body.model_dump(exclude_unset=True)
    if data.get("release_date") is not None and not isinstance(data["release_date"], str):
        data["release_date"] = data["release_date"].isoformat()
    try:
        work = await service.update_work(_get_supabase(), user_id, work_id, data)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except APIError as e:
        _raise_503_if_schema_stale(e)
        raise
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    return work


@router.delete("/works/{work_id}")
async def delete_work(work_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        await service.delete_work(_get_supabase(), user_id, work_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"ok": True}


@router.post("/works/{work_id}/submit-for-approval")
async def submit_for_approval(work_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    result, error = await service.submit_for_approval(_get_supabase(), user_id, work_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    analytics_capture(user_id, "work_submitted_for_registration", {"work_id": work_id})

    # Re-send invitation emails to all collaborators
    renotify = (result or {}).pop("_renotify_collabs", [])
    if renotify:
        from registry.emails import send_invitation_email

        profile = _get_supabase().table("profiles").select("full_name").eq("id", user_id).single().execute()
        inviter_name = (profile.data or {}).get("full_name") or "A Msanii user"
        work_title = (result or {}).get("title", "Untitled Work")
        for c in renotify:
            send_invitation_email(
                recipient_email=c["email"],
                recipient_name=c["name"],
                inviter_name=inviter_name,
                work_title=work_title,
                role="collaborator",
                invite_token=str(c.get("invite_token", "")),
            )

    return result


@router.get("/works/{work_id}/export")
async def export_proof_of_ownership(
    work_id: str,
    hide_split: list[str] = Query(default=[]),
    user_id: str = Depends(get_current_user_id),
):
    from registry.pdf_generator import generate_proof_of_ownership_pdf

    data = await service.get_work_full(_get_supabase(), user_id, work_id)
    if not data:
        raise HTTPException(status_code=404, detail="Work not found")
    buffer = generate_proof_of_ownership_pdf(data, hidden_parties=set(hide_split))
    safe_title = re.sub(r"[^a-zA-Z0-9._-]", "_", data.get("title", "work"))
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="Work_Metadata_{safe_title}.pdf"'},
    )


# ============================================================
# Ownership Stakes
# ============================================================


@router.get("/stakes")
async def list_stakes(work_id: str = Query(...), user_id: str = Depends(get_current_user_id)):
    stakes = await service.get_stakes_filtered(_get_supabase(), user_id, work_id)
    if stakes is None:
        raise HTTPException(status_code=403, detail="Not allowed")
    return {"stakes": stakes}


@router.post("/stakes")
async def create_stake(body: StakeCreate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    valid = await service.validate_stake_percentage(
        _get_supabase(), user_id, body.work_id, body.stake_type, body.percentage
    )
    if not valid:
        raise HTTPException(
            status_code=400, detail=f"Adding {body.percentage}% would exceed 100% for {body.stake_type}"
        )
    data = body.model_dump(exclude_none=True)
    try:
        stake = await service.create_stake(_get_supabase(), user_id, data)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not stake:
        raise HTTPException(status_code=500, detail="Failed to create stake")
    return stake


@router.put("/stakes/{stake_id}")
async def update_stake(stake_id: str, body: StakeUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    if body.percentage is not None:
        existing = (
            _get_supabase()
            .table("ownership_stakes")
            .select("work_id, stake_type")
            .eq("id", stake_id)
            .single()
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Stake not found")
        stake_type = body.stake_type or existing.data["stake_type"]
        valid = await service.validate_stake_percentage(
            _get_supabase(),
            user_id,
            existing.data["work_id"],
            stake_type,
            body.percentage,
            exclude_stake_id=stake_id,
        )
        if not valid:
            raise HTTPException(status_code=400, detail=f"Exceeds 100% for {stake_type}")
    data = body.model_dump(exclude_none=True)
    try:
        stake = await service.update_stake(_get_supabase(), user_id, stake_id, data)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not stake:
        raise HTTPException(status_code=404, detail="Stake not found")
    return stake


@router.delete("/stakes/{stake_id}")
async def delete_stake(stake_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        await service.delete_stake(_get_supabase(), user_id, stake_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"ok": True}


# ============================================================
# Licensing Rights
# ============================================================


@router.get("/licenses")
async def list_licenses(work_id: str = Query(...), user_id: str = Depends(get_current_user_id)):
    licenses = await service.get_licenses_filtered(_get_supabase(), user_id, work_id)
    if licenses is None:
        raise HTTPException(status_code=403, detail="Not allowed")
    return {"licenses": licenses}


@router.post("/licenses")
async def create_license(body: LicenseCreate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    for field in ("start_date", "end_date"):
        if field in data and data[field]:
            data[field] = data[field].isoformat()
    try:
        lic = await service.create_license(_get_supabase(), user_id, data)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not lic:
        raise HTTPException(status_code=500, detail="Failed to create license")
    return lic


@router.put("/licenses/{license_id}")
async def update_license(license_id: str, body: LicenseUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    for field in ("start_date", "end_date"):
        if field in data and data[field]:
            data[field] = data[field].isoformat()
    try:
        lic = await service.update_license(_get_supabase(), user_id, license_id, data)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not lic:
        raise HTTPException(status_code=404, detail="License not found")
    return lic


@router.delete("/licenses/{license_id}")
async def delete_license(license_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        await service.delete_license(_get_supabase(), user_id, license_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"ok": True}


# ============================================================
# Agreements (immutable)
# ============================================================


@router.get("/agreements")
async def list_agreements(work_id: str = Query(...), user_id: str = Depends(get_current_user_id)):
    agreements = await service.get_agreements_filtered(_get_supabase(), user_id, work_id)
    if agreements is None:
        raise HTTPException(status_code=403, detail="Not allowed")
    return {"agreements": agreements}


@router.post("/agreements")
async def create_agreement(body: AgreementCreate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    if "effective_date" in data and data["effective_date"]:
        data["effective_date"] = data["effective_date"].isoformat()
    if "parties" in data:
        data["parties"] = [p if isinstance(p, dict) else p.model_dump() for p in data["parties"]]
    try:
        agreement = await service.create_agreement(_get_supabase(), user_id, data)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not agreement:
        raise HTTPException(status_code=500, detail="Failed to create agreement")
    return agreement


# ============================================================
# Collaboration
# ============================================================


@router.get("/collaborators")
async def list_collaborators(work_id: str = Query(...), user_id: str = Depends(get_current_user_id)):
    """List collaborators. Access + roster visibility are gated by get_work_access:
    elevated/project members see the full roster; work-only viewers see only the owner
    (plus themselves, and limited info on others when granted ownership breakdown)."""
    db = _get_supabase()
    collabs = await service.get_collaborators_filtered(db, user_id, work_id)
    if collabs is None:
        raise HTTPException(status_code=403, detail="Not authorized to view collaborators")
    return {"collaborators": collabs}


@router.post("/collaborators/invite")
async def invite_collaborator(body: CollaboratorInvite, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    wa = await get_work_access(_get_supabase(), user_id, body.work_id)
    if not wa.can_manage:
        raise HTTPException(status_code=403, detail="Access denied")
    data = body.model_dump(exclude_none=True)
    work = _get_supabase().table("works_registry").select("title").eq("id", body.work_id).single().execute()
    work_title = (work.data or {}).get("title") or "Untitled Work"

    collab = await service.invite_collaborator(_get_supabase(), user_id, data, work_title=work_title)
    if not collab:
        raise HTTPException(status_code=500, detail="Failed to invite collaborator")

    analytics_capture(
        user_id,
        "registry_collaborator_invited",
        {"tool": "registry", "role": data.get("role", "collaborator")},
    )

    # Always send email
    from registry.emails import send_invitation_email

    profile = _get_supabase().table("profiles").select("full_name").eq("id", user_id).single().execute()
    inviter_name = (profile.data or {}).get("full_name") or "A Msanii user"
    send_invitation_email(
        recipient_email=body.email,
        recipient_name=body.name,
        inviter_name=inviter_name,
        work_title=work_title,
        role=body.role,
        invite_token=str(collab.get("invite_token", "")),
    )
    return collab


@router.post("/collaborators/claim")
async def claim_invitation(invite_token: str = Query(...), user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    collab, error = await service.claim_invitation(_get_supabase(), invite_token, user_id)
    if error == "expired":
        raise HTTPException(status_code=410, detail="Invitation expired — ask the project owner to resend")
    if error:
        raise HTTPException(status_code=404, detail="Invitation not found or already claimed")
    return collab


@router.post("/collaborators/{collaborator_id}/confirm")
async def confirm_stake(collaborator_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    collab = await service.confirm_stake(_get_supabase(), collaborator_id, user_id)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.get("/collaborators/invite/{token}/preview")
async def invite_preview(token: str, user_id: str = Depends(get_current_user_id)):
    db = _get_supabase()
    data = await service.get_invite_preview(db, token, user_id)
    if not data:
        raise HTTPException(404, "Invitation not found")
    if data.get("expired"):
        raise HTTPException(410, "Invitation expired — ask the owner to resend")
    return data  # may be {email_mismatch: True, ...} — frontend shows the mismatch message


@router.post("/collaborators/{collaborator_id}/revoke")
async def revoke_collaborator(collaborator_id: str, user_id: str = Depends(get_current_user_id)):
    """Revoke a collaborator: remove access, keep ownership. Gated by can_manage."""
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        collab = await service.revoke_collaborator(_get_supabase(), user_id, collaborator_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.post("/collaborators/{collaborator_id}/resend")
async def resend_invitation(collaborator_id: str, user_id: str = Depends(get_current_user_id)):
    """Resend an expired or pending invitation with a fresh token and 48h expiry."""
    gated_feature(user_id, Action.USE_REGISTRY)
    collab, error = await service.resend_invitation(_get_supabase(), user_id, collaborator_id)
    if error == "not_found":
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Re-send the email
    from registry.emails import send_invitation_email

    work = _get_supabase().table("works_registry").select("title").eq("id", collab["work_id"]).single().execute()
    work_title = (work.data or {}).get("title") or "Untitled Work"
    profile = _get_supabase().table("profiles").select("full_name").eq("id", user_id).single().execute()
    inviter_name = (profile.data or {}).get("full_name") or "A Msanii user"
    send_invitation_email(
        recipient_email=collab["email"],
        recipient_name=collab["name"],
        inviter_name=inviter_name,
        work_title=work_title,
        role=collab["role"],
        invite_token=str(collab.get("invite_token", "")),
    )
    return collab


# ============================================================
# Work File Links
# ============================================================


@router.get("/works/{work_id}/files")
async def list_work_files(work_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        files = await work_links_service.get_work_files(_get_supabase(), user_id, work_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"files": files}


@router.post("/works/{work_id}/files")
async def link_file(work_id: str, file_id: str = Query(...), user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        result = await work_links_service.link_file_to_work(_get_supabase(), user_id, work_id, file_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"link": result}


@router.delete("/works/{work_id}/files/{link_id}")
async def unlink_file(work_id: str, link_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        return await work_links_service.unlink_file_from_work(_get_supabase(), user_id, work_id, link_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")


@router.get("/works/{work_id}/files/{file_id}/download-url")
async def file_download_url(work_id: str, file_id: str, user_id: str = Depends(get_current_user_id)):
    """Access-checked, short-lived signed download URL for a project file.

    Replaces direct public bucket URLs in the registry UI so file access
    follows the same WorkAccess gate as the rest of the work's data.
    """
    url = await service.get_file_download_url(_get_supabase(), user_id, work_id, file_id)
    return {"url": url}


# ============================================================
# Work Audio Links
# ============================================================


@router.get("/works/{work_id}/audio")
async def list_work_audio(work_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        audio = await work_links_service.get_work_audio(_get_supabase(), user_id, work_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"audio": audio}


@router.post("/works/{work_id}/audio")
async def link_audio(work_id: str, audio_file_id: str = Query(...), user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        result = await work_links_service.link_audio_to_work(_get_supabase(), user_id, work_id, audio_file_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"link": result}


@router.delete("/works/{work_id}/audio/{link_id}")
async def unlink_audio(work_id: str, link_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        return await work_links_service.unlink_audio_from_work(_get_supabase(), user_id, work_id, link_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")


@router.get("/works/{work_id}/audio/{audio_id}/download-url")
async def audio_download_url(work_id: str, audio_id: str, user_id: str = Depends(get_current_user_id)):
    """Access-checked, short-lived signed download URL for a linked audio file."""
    url = await service.get_audio_download_url(_get_supabase(), user_id, work_id, audio_id)
    return {"url": url}


# ============================================================
# Enhanced Collaboration
# ============================================================


@router.post("/collaborators/invite-with-stakes")
async def invite_with_stakes_endpoint(body: CollaboratorInviteWithStakes, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        result = await service.invite_with_stakes(_get_supabase(), user_id, body)
        # Send rich invite email
        try:
            from registry.emails import send_rich_invitation_email

            db = _get_supabase()
            work = db.table("works_registry").select("title, project_id").eq("id", body.work_id).single().execute()
            project = (
                db.table("projects").select("name").eq("id", work.data["project_id"]).single().execute()
                if work.data
                else None
            )
            artist_id = db.table("works_registry").select("artist_id").eq("id", body.work_id).single().execute()
            artist = (
                db.table("artists").select("name").eq("id", artist_id.data["artist_id"]).single().execute()
                if artist_id.data
                else None
            )
            inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
            send_rich_invitation_email(
                recipient_email=body.email,
                recipient_name=body.name,
                inviter_name=inviter.data.get("full_name", "Someone") if inviter.data else "Someone",
                work_title=work.data["title"] if work.data else "Unknown",
                project_name=project.data["name"] if project and project.data else "Unknown",
                artist_name=artist.data["name"] if artist and artist.data else "Unknown",
                role=body.role,
                stakes=body.stakes,
                notes=body.notes,
                invite_token=result["invite_token"],
            )
        except Exception as e:
            print(f"Warning: Failed to send invitation email: {e}")
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))  # already-active re-invite / bad grant
    except APIError as e:
        # The validate_stake_total trigger raises on >100% totals. Translate its
        # raw message ("Total % for master would exceed 100% (current: %100.00,
        # adding: %40.00)") into something a non-technical user can act on.
        msg = (e.message if hasattr(e, "message") else str(e)) or ""
        m = re.search(r"Total % for (\w+) would exceed 100% \(current: %([\d.]+), adding: %([\d.]+)\)", msg)
        if m:
            stake_type, current, adding = m.group(1), float(m.group(2)), float(m.group(3))
            remaining = max(0.0, 100.0 - current)
            friendly = (
                f"This person's {stake_type} share ({adding:g}%) would push the total above 100%. "
                f"{current:g}% is already allocated — at most {remaining:g}% is available."
            )
            raise HTTPException(status_code=400, detail=friendly)
        raise


@router.post("/collaborators/{collaborator_id}/decline")
async def decline_invitation_endpoint(collaborator_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        return await service.decline_invitation(_get_supabase(), user_id, collaborator_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/collaborators/{collaborator_id}/accept-from-dashboard")
async def accept_from_dashboard_endpoint(collaborator_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    try:
        return await service.accept_from_dashboard(_get_supabase(), user_id, collaborator_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/collaborators/my-invites")
async def my_invites_endpoint(user_id: str = Depends(get_current_user_id)):
    invites = await service.get_my_invites(_get_supabase(), user_id)
    return {"invites": invites}


# ============================================================
# Per-collaborator visibility grants & access level
# ============================================================


@router.get("/works/{work_id}/access")
async def work_access(work_id: str, user_id: str = Depends(get_current_user_id)):
    db = _get_supabase()
    wa = await get_work_access(db, user_id, work_id)
    return {
        "work_role": wa.work_role,
        "project_role": wa.project_role,
        "can_view": wa.can_view,
        "can_edit": wa.can_edit,
        "can_manage": wa.can_manage,
        "can_delete": wa.can_delete,
        "can_see_full_ownership": wa.can_see_full_ownership,
        "is_project_member": wa.is_project_member,
        "my_collaborator_id": wa.my_collaborator_id,
        "all_visible": wa.all_visible(),
        # sets -> lists (sets are not JSON-serializable)
        "visible_stake_ids": list(wa.visible_stake_ids),
        "visible_file_ids": list(wa.visible_file_ids),
        "visible_audio_ids": list(wa.visible_audio_ids),
        "visible_license_ids": list(wa.visible_license_ids),
        "visible_agreement_ids": list(wa.visible_agreement_ids),
    }


@router.get("/works/{work_id}/grants")
async def list_grants(work_id: str, user_id: str = Depends(get_current_user_id)):
    db = _get_supabase()
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_manage:
        raise HTTPException(403, "Not allowed to manage this work")
    return await grants_service.get_grant_matrix(db, work_id)


@router.post("/collaborators/{collaborator_id}/grants")
async def add_grants(collaborator_id: str, body: GrantsBody, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    db = _get_supabase()
    collab = db.table("registry_collaborators").select("work_id").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise HTTPException(404, "Collaborator not found")
    work_id = collab.data["work_id"]
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_manage:
        raise HTTPException(403, "Not allowed")
    try:
        for g in body.grants:
            await grants_service.add_grant(db, collaborator_id, work_id, g.resource_type, g.resource_id, user_id)
        if body.ownership_breakdown is True:
            await grants_service.add_grant(db, collaborator_id, work_id, "ownership_breakdown", None, user_id)
        if body.ownership_breakdown is False:
            await grants_service.remove_grant(db, collaborator_id, "ownership_breakdown", None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    analytics_capture(
        user_id,
        "registry_access_granted",
        {
            "resource_types": sorted({g.resource_type for g in body.grants}),
            "ownership_breakdown": bool(body.ownership_breakdown),
        },
    )
    return {"ok": True}


@router.delete("/collaborators/{collaborator_id}/grants")
async def delete_grants(collaborator_id: str, body: GrantsBody, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    db = _get_supabase()
    collab = db.table("registry_collaborators").select("work_id").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise HTTPException(404, "Collaborator not found")
    wa = await get_work_access(db, user_id, collab.data["work_id"])
    if not wa.can_manage:
        raise HTTPException(403, "Not allowed")
    for g in body.grants:
        await grants_service.remove_grant(db, collaborator_id, g.resource_type, g.resource_id)
    analytics_capture(
        user_id,
        "registry_access_revoked",
        {"resource_types": sorted({g.resource_type for g in body.grants})},
    )
    return {"ok": True}


@router.put("/collaborators/{collaborator_id}/access-level")
async def set_access_level(collaborator_id: str, body: AccessLevelUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    db = _get_supabase()
    collab = db.table("registry_collaborators").select("work_id").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise HTTPException(404, "Collaborator not found")
    wa = await get_work_access(db, user_id, collab.data["work_id"])
    if not wa.can_manage:
        raise HTTPException(403, "Not allowed")
    try:
        await grants_service.set_access_level(db, collaborator_id, body.access_level)
    except ValueError as e:
        raise HTTPException(400, str(e))
    analytics_capture(
        user_id,
        "registry_admin_assigned" if body.access_level == "admin" else "registry_admin_revoked",
        {"collaborator_id": collaborator_id},
    )
    return {"ok": True}


@router.put("/collaborators/{collaborator_id}/role")
async def set_collaborator_role(
    collaborator_id: str, body: WorkRoleUpdate, user_id: str = Depends(get_current_user_id)
):
    gated_feature(user_id, Action.USE_REGISTRY)
    db = _get_supabase()
    collab = db.table("registry_collaborators").select("work_id").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise HTTPException(404, "Collaborator not found")
    wa = await get_work_access(db, user_id, collab.data["work_id"])
    if not wa.can_manage:
        raise HTTPException(403, "Not allowed")
    try:
        await grants_service.set_work_role(db, collaborator_id, body.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@router.post("/collaborators/derive-from-contracts")
async def derive_from_contracts(body: DeriveFromContractsBody, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    db = _get_supabase()
    wa = await get_work_access(db, user_id, body.work_id)
    if not wa.can_manage:
        raise HTTPException(403, "Not allowed to manage this work")
    for cid in body.contract_file_ids or []:
        link = db.table("work_files").select("id").eq("work_id", body.work_id).eq("file_id", cid).execute()
        if not link.data:
            raise HTTPException(status_code=403, detail="Access denied")
    result = await derive_service.derive_for_collaborator(
        db, body.work_id, body.name, body.email, body.contract_file_ids
    )

    # Auto-link the source contract to the work — when a split was successfully
    # derived from a file, that file is by definition related to this work and
    # belongs in Related Documents. Pre-filter against work_files to skip files
    # already linked (the common case when deriving from already-linked contracts).
    if result.get("found") and result.get("matched_file_ids"):
        already = (db.table("work_files").select("file_id").eq("work_id", body.work_id).execute()).data or []
        already_ids = {r["file_id"] for r in already}
        for fid in result["matched_file_ids"]:
            if fid in already_ids:
                continue
            try:
                await work_links_service.link_file_to_work(db, user_id, body.work_id, fid)
            except Exception:
                # Best-effort — a late duplicate or RLS edge shouldn't fail the derive.
                pass

    analytics_capture(
        user_id,
        "registry_contract_derived",
        {"found": result["found"], "confidence": result["confidence"]},
    )
    return result


# ============================================================
# TeamCard
# ============================================================


@router.get("/teamcard")
async def get_my_team_card(user_id: str = Depends(get_current_user_id)):
    card = await service.get_team_card(_get_supabase(), user_id)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found — complete onboarding first")
    return card


@router.put("/teamcard")
async def update_team_card(body: TeamCardUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    card = await service.update_team_card(_get_supabase(), user_id, data)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found")
    return card


@router.get("/teamcard/{collaborator_user_id}")
async def get_collaborator_team_card(collaborator_user_id: str, user_id: str = Depends(get_current_user_id)):
    """Get a collaborator's visible TeamCard fields."""
    db = _get_supabase()
    # Verify collaboration relationship exists
    shared = (
        db.table("registry_collaborators")
        .select("id")
        .or_(
            f"and(invited_by.eq.{user_id},collaborator_user_id.eq.{collaborator_user_id}),"
            f"and(invited_by.eq.{collaborator_user_id},collaborator_user_id.eq.{user_id})"
        )
        .neq("status", "revoked")
        .execute()
    )
    if not shared.data:
        raise HTTPException(status_code=403, detail="No collaboration link with this user")
    card = await service.get_collaborator_team_card(db, collaborator_user_id)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found for this user")
    return card


# ============================================================
# Artist with TeamCard overlay (Option C merge)
# ============================================================


@router.get("/artists/{artist_id}/with-teamcard")
async def get_artist_with_teamcard(artist_id: str, user_id: str = Depends(get_current_user_id)):
    from main import verify_user_owns_artist

    if not verify_user_owns_artist(user_id, artist_id):
        raise HTTPException(status_code=403, detail="Access denied")
    data = await service.get_artist_with_teamcard(_get_supabase(), artist_id)
    if not data:
        raise HTTPException(status_code=404, detail="Artist not found")
    return data


@router.get("/artists/with-teamcards")
async def list_artists_with_teamcards(user_id: str = Depends(get_current_user_id)):
    """Batch endpoint: returns all of a user's artists with TeamCard overlays applied."""
    artists = await service.get_artists_with_teamcards(_get_supabase(), user_id)
    return {"artists": artists}


# ============================================================
# Notes
# ============================================================


@router.get("/notes")
async def list_notes(
    user_id: str = Depends(get_current_user_id),
    artist_id: str | None = Query(None),
    project_id: str | None = Query(None),
    folder_id: str | None = Query(None),
):
    notes = await service.get_notes(_get_supabase(), user_id, artist_id, project_id, folder_id)
    return {"notes": notes}


@router.get("/notes/{note_id}")
async def get_note(note_id: str, user_id: str = Depends(get_current_user_id)):
    note = await service.get_note(_get_supabase(), user_id, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.post("/notes")
async def create_note(body: NoteCreate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    note = await service.create_note(_get_supabase(), user_id, data)
    if not note:
        raise HTTPException(status_code=500, detail="Failed to create note")
    return note


@router.put("/notes/{note_id}")
async def update_note(note_id: str, body: NoteUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    note = await service.update_note(_get_supabase(), user_id, note_id, data)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    await service.delete_note(_get_supabase(), user_id, note_id)
    return {"ok": True}


@router.get("/folders")
async def list_folders(
    user_id: str = Depends(get_current_user_id),
    artist_id: str | None = Query(None),
    project_id: str | None = Query(None),
):
    folders = await service.get_folders(_get_supabase(), user_id, artist_id, project_id)
    return {"folders": folders}


@router.post("/folders")
async def create_folder(body: FolderCreate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    folder = await service.create_folder(_get_supabase(), user_id, data)
    if not folder:
        raise HTTPException(status_code=500, detail="Failed to create folder")
    return folder


@router.put("/folders/{folder_id}")
async def update_folder(folder_id: str, body: FolderUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    data = body.model_dump(exclude_none=True)
    folder = await service.update_folder(_get_supabase(), user_id, folder_id, data)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder


@router.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    await service.delete_folder(_get_supabase(), user_id, folder_id)
    return {"ok": True}


# ============================================================
# Project About
# ============================================================


@router.get("/projects/{project_id}/about")
async def get_project_about(project_id: str, user_id: str = Depends(get_current_user_id)):
    """Get project about content."""
    db = _get_supabase()
    # Check access: owner or collaborator
    project = db.table("projects").select("artist_id").eq("id", project_id).single().execute()
    if not project.data:
        raise HTTPException(status_code=404, detail="Project not found")
    artist = (
        db.table("artists").select("user_id, linked_user_id").eq("id", project.data["artist_id"]).single().execute()
    )
    is_owner = artist.data and (artist.data["user_id"] == user_id or artist.data.get("linked_user_id") == user_id)
    is_collab = db.table("works_registry").select("id").eq("project_id", project_id).execute()
    collab_work_ids = [w["id"] for w in (is_collab.data or [])]
    has_collab_access = False
    if collab_work_ids:
        check = (
            db.table("registry_collaborators")
            .select("id")
            .in_("work_id", collab_work_ids)
            .eq("collaborator_user_id", user_id)
            .neq("status", "revoked")
            .execute()
        )
        has_collab_access = bool(check.data)
    if not is_owner and not has_collab_access:
        raise HTTPException(status_code=403, detail="Not authorized to view this project")
    content = await service.get_project_about(db, project_id)
    return {"about_content": content}


@router.put("/projects/{project_id}/about")
async def update_project_about(project_id: str, body: ProjectAboutUpdate, user_id: str = Depends(get_current_user_id)):
    gated_feature(user_id, Action.USE_REGISTRY)
    result = await service.update_project_about(_get_supabase(), user_id, project_id, body.about_content)
    if not result:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}


# ============================================================
# Notifications
# ============================================================


@router.get("/notifications")
async def list_notifications(user_id: str = Depends(get_current_user_id), unread_only: bool = Query(False)):
    notifications = await service.get_notifications(_get_supabase(), user_id, unread_only)
    return {"notifications": notifications}


@router.post("/notifications/{notification_id}/read")
async def mark_read(notification_id: str, user_id: str = Depends(get_current_user_id)):
    await service.mark_notification_read(_get_supabase(), user_id, notification_id)
    return {"ok": True}


@router.post("/notifications/read-all")
async def mark_all_read(user_id: str = Depends(get_current_user_id)):
    await service.mark_all_notifications_read(_get_supabase(), user_id)
    return {"ok": True}


# ============================================================
# Contract → royalty-splits parsing (Add Work wizard)
# ============================================================


@router.post("/parse-contract-splits")
async def parse_contract_splits(
    main_artist_name: str = Form(""),
    contract_file_id: str = Form(""),
    file: UploadFile | None = File(None),
    user_id: str = Depends(get_current_user_id),
):
    """Pivot a contract PDF into per-party master/publishing royalty splits.

    Accepts either:
      - a multipart `file` upload (PDF), OR
      - a `contract_file_id` referencing an existing `project_files` row that
        the user can access. The file is downloaded from the `project-files`
        Supabase storage bucket.

    Powers the Add Work wizard's "Pull from the contract" step. Returns:
        {parties: [{name, role, aliases, master_pct, publishing_pct,
                    soundexchange_pct, is_main_artist}],
         main_artist_found: bool}

    If `main_artist_found` is false, the main artist is omitted from `parties`
    and the frontend prompts the user to enter their own split manually.
    """
    gated_feature(user_id, Action.USE_REGISTRY)

    import os
    import tempfile

    has_upload = file is not None and file.filename
    has_picked = bool(contract_file_id)
    if has_upload == has_picked:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: `file` (upload) or `contract_file_id` (existing).",
        )

    contents: bytes | None = None
    stored_md: str | None = None
    if has_upload:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
    else:
        from main import verify_user_owns_contract

        if not verify_user_owns_contract(user_id, contract_file_id):
            raise HTTPException(status_code=403, detail="Access denied")
        db = _get_supabase()
        row = (
            db.table("project_files")
            .select("file_path, file_name, contract_markdown")
            .eq("id", contract_file_id)
            .maybe_single()
            .execute()
        )
        if not row.data:
            raise HTTPException(status_code=404, detail="Contract file not found")
        stored_md = row.data.get("contract_markdown")
        if not stored_md:
            file_path = row.data.get("file_path")
            if not file_path:
                raise HTTPException(status_code=400, detail="Contract file has no storage path")
            try:
                contents = db.storage.from_("project-files").download(file_path)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Failed to download contract: {exc}") from exc

    from utils.contract_parsing.cache import get_or_parse, peek_cached_parse
    from utils.ingestion.pdf_markdown import pdf_to_markdown

    def _load_text() -> str:
        # Prefer the stored canonical markdown (shares cache entries with OneClick and the
        # Add-Work collaborator derivation); otherwise convert the uploaded/downloaded PDF.
        if stored_md:
            return stored_md
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            return pdf_to_markdown(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # Load the text FIRST (pdf→markdown is local compute, no LLM cost), then peek the parse
    # cache: a guaranteed hit is free (spec §3) and must not be walled at zero balance.
    # Legacy mode (credits off) skips the peek and always takes the gate below.
    try:
        text = _load_text()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract parsing failed: {e}")

    cached_peek = peek_cached_parse(_get_supabase(), text) if credits_enabled() else None
    if credits_enabled() and cached_peek is not None:
        parse_grant = free_credit_grant(CreditAction.REGISTRY_PARSE)
    else:
        # SP3/credits: gate the contract parse; 402 without access or credits.
        # Resource-derived billing (Licensing Phase C, rule 5): a PICKED existing
        # contract (contract_file_id) derives its project's linked-org billing —
        # passed as a one-element list; an UPLOAD has no resource → ambient.
        # Derivation-vs-access ordering (rule 4, Phase A access-first):
        # verify_user_owns_contract on the picked file ran above BEFORE this gate,
        # so derivation can never bill an org for a contract the caller can't
        # access; charge-on-success is the backstop.
        parse_grant = gated_credits(
            user_id,
            CreditAction.REGISTRY_PARSE,
            resource_contract_ids=[contract_file_id] if has_picked else None,
        )

    try:
        with set_llm_context(user_id, "registry"):
            # Route through the shared parse cache so this Add-Work parse is cached and
            # canonicalized (marker-stripped) like every other contract parse.
            cache_missed = {"v": False}
            contract_data = get_or_parse(_get_supabase(), lambda: text, on_miss=lambda: cache_missed.update(v=True))
            result = contract_splits.parse_royalty_splits(
                contract_data=contract_data,
                main_artist_name=main_artist_name or "",
            )
        # Charge only when a real LLM parse happened — cache hits are free (spec §3).
        if cache_missed["v"]:
            from subscriptions.deps import _get_entitlements_service

            _get_entitlements_service().debit_for_action(user_id, parse_grant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract parsing failed: {e}")

    analytics_capture(
        user_id,
        "registry_contract_splits_parsed",
        properties={
            "party_count": len(result["parties"]),
            "main_artist_found": result["main_artist_found"],
            "source": "upload" if has_upload else "project_file",
        },
    )
    return result
