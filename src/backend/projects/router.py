import os
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from projects import service
from projects.models import MemberAdd, MemberUpdate
from projects.service import DuplicateInviteError

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


@router.get("/{project_id}/members")
async def list_members(project_id: str, user_id: str = Depends(get_current_user_id)):
    members = await service.get_members(_get_supabase(), user_id, project_id)
    return {"members": members}


def _send_invite_email_background(
    db_url: str,
    db_key: str,
    project_id: str,
    invite_id: str,
    user_id: str,
    email: str,
    role: str,
):
    """Background task: fetch context, send invite email, and persist the outcome."""
    from supabase import create_client

    from projects.emails import send_project_invite_email

    db = create_client(db_url, db_key)
    error_message: str | None = None
    try:
        project = db.table("projects").select("name").eq("id", project_id).single().execute()
        inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
        send_project_invite_email(
            recipient_email=email,
            project_name=project.data["name"] if project.data else "Unknown",
            inviter_name=inviter.data.get("full_name", "Someone") if inviter.data else "Someone",
            role=role,
        )
    except Exception as exc:
        error_message = str(exc)
        print(f"Background: Failed to send project invite email: {exc}")

    try:
        db.table("pending_project_invites").update(
            {
                "last_email_error": error_message,
                "last_email_attempt_at": "now()",
            }
        ).eq("id", invite_id).execute()
    except Exception as exc:
        print(f"Background: Failed to record invite email status for {invite_id}: {exc}")


def _schedule_invite_email(
    background_tasks: BackgroundTasks, invite_id: str, user_id: str, project_id: str, email: str, role: str
):
    background_tasks.add_task(
        _send_invite_email_background,
        db_url=os.getenv("VITE_SUPABASE_URL"),
        db_key=os.getenv("VITE_SUPABASE_SECRET_KEY"),
        project_id=project_id,
        invite_id=invite_id,
        user_id=user_id,
        email=email,
        role=role,
    )


@router.post("/{project_id}/members")
async def add_member(
    project_id: str, body: MemberAdd, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.add_member(_get_supabase(), user_id, project_id, body.email, body.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except DuplicateInviteError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result["type"] == "pending" and result.get("invite"):
        _schedule_invite_email(
            background_tasks,
            invite_id=result["invite"]["id"],
            user_id=user_id,
            project_id=project_id,
            email=body.email,
            role=body.role,
        )
    return result


@router.put("/{project_id}/members/{member_id}")
async def update_member(
    project_id: str, member_id: str, body: MemberUpdate, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.update_member_role(_get_supabase(), user_id, project_id, member_id, body.role)
        return {"member": result}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{project_id}/members/{member_id}")
async def remove_member(project_id: str, member_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.remove_member(_get_supabase(), user_id, project_id, member_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/pending-invites")
async def list_pending_invites(project_id: str, user_id: str = Depends(get_current_user_id)):
    invites = await service.get_pending_invites(_get_supabase(), user_id, project_id)
    return {"invites": invites}


@router.delete("/{project_id}/pending-invites/{invite_id}")
async def cancel_pending_invite(project_id: str, invite_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.delete_pending_invite(_get_supabase(), user_id, project_id, invite_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{project_id}/pending-invites/{invite_id}/resend")
async def resend_invite(
    project_id: str,
    invite_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    db = _get_supabase()
    caller_role = await service.get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only admins can resend invites")

    invite = await service.get_pending_invite(db, project_id, invite_id)
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")

    _schedule_invite_email(
        background_tasks,
        invite_id=invite_id,
        user_id=user_id,
        project_id=project_id,
        email=invite["email"],
        role=invite["role"],
    )
    return {"scheduled": True}
