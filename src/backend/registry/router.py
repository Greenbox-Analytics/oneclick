"""FastAPI router for the Rights & Ownership Registry."""

import re
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from registry import service
from registry.models import (
    WorkCreate, WorkUpdate,
    StakeCreate, StakeUpdate,
    LicenseCreate, LicenseUpdate,
    AgreementCreate,
    CollaboratorInvite, DisputeRequest,
    TeamCardUpdate,
    NoteCreate, NoteUpdate, FolderCreate, FolderUpdate,
    ProjectAboutUpdate,
)

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


# ============================================================
# Works
# ============================================================

@router.get("/works")
async def list_works(user_id: str = Query(...), artist_id: Optional[str] = Query(None)):
    works = await service.get_works(_get_supabase(), user_id, artist_id)
    return {"works": works}


@router.get("/works/my-collaborations")
async def list_my_collaborations(user_id: str = Query(...)):
    works = await service.get_works_as_collaborator(_get_supabase(), user_id)
    return {"works": works}


@router.get("/works/by-project/{project_id}")
async def list_works_by_project(project_id: str, user_id: str = Query(...)):
    works = await service.get_works_by_project(_get_supabase(), user_id, project_id)
    return {"works": works}


@router.get("/works/{work_id}")
async def get_work(work_id: str, user_id: str = Query(...)):
    work = await service.get_work(_get_supabase(), user_id, work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    return work


@router.get("/works/{work_id}/full")
async def get_work_full(work_id: str, user_id: str = Query(...)):
    data = await service.get_work_full(_get_supabase(), user_id, work_id)
    if not data:
        raise HTTPException(status_code=404, detail="Work not found")
    return data


@router.post("/works")
async def create_work(body: WorkCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    if "release_date" in data and data["release_date"]:
        data["release_date"] = data["release_date"].isoformat()
    work = await service.create_work(_get_supabase(), user_id, data)
    if not work:
        raise HTTPException(status_code=500, detail="Failed to create work")
    return work


@router.put("/works/{work_id}")
async def update_work(work_id: str, body: WorkUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    if "release_date" in data and data["release_date"]:
        data["release_date"] = data["release_date"].isoformat()
    work = await service.update_work(_get_supabase(), user_id, work_id, data)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    return work


@router.delete("/works/{work_id}")
async def delete_work(work_id: str, user_id: str = Query(...)):
    await service.delete_work(_get_supabase(), user_id, work_id)
    return {"ok": True}


@router.post("/works/{work_id}/submit-for-approval")
async def submit_for_approval(work_id: str, user_id: str = Query(...)):
    result, error = await service.submit_for_approval(_get_supabase(), user_id, work_id)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Re-send invitation emails to all collaborators (especially those reset from disputed)
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
async def export_proof_of_ownership(work_id: str, user_id: str = Query(...)):
    from registry.pdf_generator import generate_proof_of_ownership_pdf
    data = await service.get_work_full(_get_supabase(), user_id, work_id)
    if not data:
        raise HTTPException(status_code=404, detail="Work not found")
    buffer = generate_proof_of_ownership_pdf(data)
    safe_title = re.sub(r"[^a-zA-Z0-9._-]", "_", data.get("title", "work"))
    return StreamingResponse(
        buffer, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="Proof_of_Ownership_{safe_title}.pdf"'},
    )


# ============================================================
# Ownership Stakes
# ============================================================

@router.get("/stakes")
async def list_stakes(work_id: str = Query(...), user_id: str = Query(...)):
    stakes = await service.get_stakes(_get_supabase(), user_id, work_id)
    return {"stakes": stakes}


@router.post("/stakes")
async def create_stake(body: StakeCreate, user_id: str = Query(...)):
    valid = await service.validate_stake_percentage(
        _get_supabase(), user_id, body.work_id, body.stake_type, body.percentage
    )
    if not valid:
        raise HTTPException(status_code=400, detail=f"Adding {body.percentage}% would exceed 100% for {body.stake_type}")
    data = body.model_dump(exclude_none=True)
    stake = await service.create_stake(_get_supabase(), user_id, data)
    if not stake:
        raise HTTPException(status_code=500, detail="Failed to create stake")
    return stake


@router.put("/stakes/{stake_id}")
async def update_stake(stake_id: str, body: StakeUpdate, user_id: str = Query(...)):
    if body.percentage is not None:
        existing = (
            _get_supabase().table("ownership_stakes")
            .select("work_id, stake_type").eq("id", stake_id).eq("user_id", user_id)
            .single().execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Stake not found")
        stake_type = body.stake_type or existing.data["stake_type"]
        valid = await service.validate_stake_percentage(
            _get_supabase(), user_id, existing.data["work_id"],
            stake_type, body.percentage, exclude_stake_id=stake_id,
        )
        if not valid:
            raise HTTPException(status_code=400, detail=f"Exceeds 100% for {stake_type}")
    data = body.model_dump(exclude_none=True)
    stake = await service.update_stake(_get_supabase(), user_id, stake_id, data)
    if not stake:
        raise HTTPException(status_code=404, detail="Stake not found")
    return stake


@router.delete("/stakes/{stake_id}")
async def delete_stake(stake_id: str, user_id: str = Query(...)):
    await service.delete_stake(_get_supabase(), user_id, stake_id)
    return {"ok": True}


# ============================================================
# Licensing Rights
# ============================================================

@router.get("/licenses")
async def list_licenses(work_id: str = Query(...), user_id: str = Query(...)):
    licenses = await service.get_licenses(_get_supabase(), user_id, work_id)
    return {"licenses": licenses}


@router.post("/licenses")
async def create_license(body: LicenseCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    for field in ("start_date", "end_date"):
        if field in data and data[field]:
            data[field] = data[field].isoformat()
    lic = await service.create_license(_get_supabase(), user_id, data)
    if not lic:
        raise HTTPException(status_code=500, detail="Failed to create license")
    return lic


@router.put("/licenses/{license_id}")
async def update_license(license_id: str, body: LicenseUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    for field in ("start_date", "end_date"):
        if field in data and data[field]:
            data[field] = data[field].isoformat()
    lic = await service.update_license(_get_supabase(), user_id, license_id, data)
    if not lic:
        raise HTTPException(status_code=404, detail="License not found")
    return lic


@router.delete("/licenses/{license_id}")
async def delete_license(license_id: str, user_id: str = Query(...)):
    await service.delete_license(_get_supabase(), user_id, license_id)
    return {"ok": True}


# ============================================================
# Agreements (immutable)
# ============================================================

@router.get("/agreements")
async def list_agreements(work_id: str = Query(...), user_id: str = Query(...)):
    agreements = await service.get_agreements(_get_supabase(), user_id, work_id)
    return {"agreements": agreements}


@router.post("/agreements")
async def create_agreement(body: AgreementCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    if "effective_date" in data and data["effective_date"]:
        data["effective_date"] = data["effective_date"].isoformat()
    if "parties" in data:
        data["parties"] = [p if isinstance(p, dict) else p.model_dump() for p in data["parties"]]
    agreement = await service.create_agreement(_get_supabase(), user_id, data)
    if not agreement:
        raise HTTPException(status_code=500, detail="Failed to create agreement")
    return agreement


# ============================================================
# Collaboration
# ============================================================

@router.get("/collaborators")
async def list_collaborators(work_id: str = Query(...), user_id: str = Query(...)):
    """List collaborators. Only the work creator or an existing collaborator can view."""
    db = _get_supabase()
    work = db.table("works_registry").select("user_id").eq("id", work_id).single().execute()
    if not work.data:
        raise HTTPException(status_code=404, detail="Work not found")
    is_creator = work.data["user_id"] == user_id
    is_collab = db.table("registry_collaborators").select("id").eq("work_id", work_id).eq("collaborator_user_id", user_id).neq("status", "revoked").execute()
    if not is_creator and not (is_collab.data):
        raise HTTPException(status_code=403, detail="Not authorized to view collaborators")
    collabs = await service.get_collaborators(db, work_id)
    return {"collaborators": collabs}


@router.post("/collaborators/invite")
async def invite_collaborator(body: CollaboratorInvite, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    work = _get_supabase().table("works_registry").select("title").eq("id", body.work_id).single().execute()
    work_title = (work.data or {}).get("title") or "Untitled Work"

    collab = await service.invite_collaborator(_get_supabase(), user_id, data, work_title=work_title)
    if not collab:
        raise HTTPException(status_code=500, detail="Failed to invite collaborator")

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
async def claim_invitation(invite_token: str = Query(...), user_id: str = Query(...)):
    collab, error = await service.claim_invitation(_get_supabase(), invite_token, user_id)
    if error == "expired":
        raise HTTPException(status_code=410, detail="Invitation expired — ask the project owner to resend")
    if error:
        raise HTTPException(status_code=404, detail="Invitation not found or already claimed")
    return collab


@router.post("/collaborators/{collaborator_id}/confirm")
async def confirm_stake(collaborator_id: str, user_id: str = Query(...)):
    collab = await service.confirm_stake(_get_supabase(), collaborator_id, user_id)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.post("/collaborators/{collaborator_id}/dispute")
async def dispute_stake(collaborator_id: str, body: DisputeRequest, user_id: str = Query(...)):
    collab = await service.dispute_stake(_get_supabase(), collaborator_id, user_id, body.reason)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.post("/collaborators/{collaborator_id}/revoke")
async def revoke_collaborator(collaborator_id: str, user_id: str = Query(...)):
    """Revoke a collaborator invitation. Only the inviter can do this."""
    collab = await service.revoke_collaborator(_get_supabase(), user_id, collaborator_id)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaborator record not found")
    return collab


@router.post("/collaborators/{collaborator_id}/resend")
async def resend_invitation(collaborator_id: str, user_id: str = Query(...)):
    """Resend an expired or pending invitation with a fresh token and 48h expiry."""
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
# TeamCard
# ============================================================

@router.get("/teamcard")
async def get_my_team_card(user_id: str = Query(...)):
    card = await service.get_team_card(_get_supabase(), user_id)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found — complete onboarding first")
    return card


@router.put("/teamcard")
async def update_team_card(body: TeamCardUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    card = await service.update_team_card(_get_supabase(), user_id, data)
    if not card:
        raise HTTPException(status_code=404, detail="TeamCard not found")
    return card


@router.get("/teamcard/{collaborator_user_id}")
async def get_collaborator_team_card(collaborator_user_id: str, user_id: str = Query(...)):
    """Get a collaborator's visible TeamCard fields."""
    db = _get_supabase()
    # Verify collaboration relationship exists
    shared = db.table("registry_collaborators").select("id").or_(
        f"and(invited_by.eq.{user_id},collaborator_user_id.eq.{collaborator_user_id}),"
        f"and(invited_by.eq.{collaborator_user_id},collaborator_user_id.eq.{user_id})"
    ).neq("status", "revoked").execute()
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
async def get_artist_with_teamcard(artist_id: str, user_id: str = Query(...)):
    data = await service.get_artist_with_teamcard(_get_supabase(), artist_id)
    if not data:
        raise HTTPException(status_code=404, detail="Artist not found")
    return data


@router.get("/artists/with-teamcards")
async def list_artists_with_teamcards(user_id: str = Query(...)):
    """Batch endpoint: returns all of a user's artists with TeamCard overlays applied."""
    artists = await service.get_artists_with_teamcards(_get_supabase(), user_id)
    return {"artists": artists}


# ============================================================
# Notes
# ============================================================

@router.get("/notes")
async def list_notes(
    user_id: str = Query(...),
    artist_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    folder_id: Optional[str] = Query(None),
):
    notes = await service.get_notes(_get_supabase(), user_id, artist_id, project_id, folder_id)
    return {"notes": notes}


@router.get("/notes/{note_id}")
async def get_note(note_id: str, user_id: str = Query(...)):
    note = await service.get_note(_get_supabase(), user_id, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.post("/notes")
async def create_note(body: NoteCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    note = await service.create_note(_get_supabase(), user_id, data)
    if not note:
        raise HTTPException(status_code=500, detail="Failed to create note")
    return note


@router.put("/notes/{note_id}")
async def update_note(note_id: str, body: NoteUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    note = await service.update_note(_get_supabase(), user_id, note_id, data)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str, user_id: str = Query(...)):
    await service.delete_note(_get_supabase(), user_id, note_id)
    return {"ok": True}


@router.get("/folders")
async def list_folders(
    user_id: str = Query(...),
    artist_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
):
    folders = await service.get_folders(_get_supabase(), user_id, artist_id, project_id)
    return {"folders": folders}


@router.post("/folders")
async def create_folder(body: FolderCreate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    folder = await service.create_folder(_get_supabase(), user_id, data)
    if not folder:
        raise HTTPException(status_code=500, detail="Failed to create folder")
    return folder


@router.put("/folders/{folder_id}")
async def update_folder(folder_id: str, body: FolderUpdate, user_id: str = Query(...)):
    data = body.model_dump(exclude_none=True)
    folder = await service.update_folder(_get_supabase(), user_id, folder_id, data)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder


@router.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str, user_id: str = Query(...)):
    await service.delete_folder(_get_supabase(), user_id, folder_id)
    return {"ok": True}


# ============================================================
# Project About
# ============================================================

@router.get("/projects/{project_id}/about")
async def get_project_about(project_id: str, user_id: str = Query(...)):
    """Get project about content."""
    db = _get_supabase()
    # Check access: owner or collaborator
    project = db.table("projects").select("artist_id").eq("id", project_id).single().execute()
    if not project.data:
        raise HTTPException(status_code=404, detail="Project not found")
    artist = db.table("artists").select("user_id, linked_user_id").eq("id", project.data["artist_id"]).single().execute()
    is_owner = artist.data and (artist.data["user_id"] == user_id or artist.data.get("linked_user_id") == user_id)
    is_collab = db.table("works_registry").select("id").eq("project_id", project_id).execute()
    collab_work_ids = [w["id"] for w in (is_collab.data or [])]
    has_collab_access = False
    if collab_work_ids:
        check = db.table("registry_collaborators").select("id").in_("work_id", collab_work_ids).eq("collaborator_user_id", user_id).neq("status", "revoked").execute()
        has_collab_access = bool(check.data)
    if not is_owner and not has_collab_access:
        raise HTTPException(status_code=403, detail="Not authorized to view this project")
    content = await service.get_project_about(db, project_id)
    return {"about_content": content}


@router.put("/projects/{project_id}/about")
async def update_project_about(project_id: str, body: ProjectAboutUpdate, user_id: str = Query(...)):
    result = await service.update_project_about(_get_supabase(), user_id, project_id, body.about_content)
    if not result:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}


# ============================================================
# Notifications
# ============================================================

@router.get("/notifications")
async def list_notifications(user_id: str = Query(...), unread_only: bool = Query(False)):
    notifications = await service.get_notifications(_get_supabase(), user_id, unread_only)
    return {"notifications": notifications}


@router.post("/notifications/{notification_id}/read")
async def mark_read(notification_id: str, user_id: str = Query(...)):
    await service.mark_notification_read(_get_supabase(), user_id, notification_id)
    return {"ok": True}


@router.post("/notifications/read-all")
async def mark_all_read(user_id: str = Query(...)):
    await service.mark_all_notifications_read(_get_supabase(), user_id)
    return {"ok": True}
