from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from projects import service
from projects.models import MemberAdd, MemberUpdate

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


@router.get("/{project_id}/members")
async def list_members(project_id: str, user_id: str = Query(...)):
    members = await service.get_members(_get_supabase(), user_id, project_id)
    return {"members": members}


@router.post("/{project_id}/members")
async def add_member(project_id: str, body: MemberAdd, user_id: str = Query(...)):
    try:
        result = await service.add_member(
            _get_supabase(), user_id, project_id, body.email, body.role
        )
        if result["type"] == "pending":
            try:
                from projects.emails import send_project_invite_email
                db = _get_supabase()
                project = db.table("projects").select("name").eq("id", project_id).single().execute()
                inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
                send_project_invite_email(
                    recipient_email=body.email,
                    project_name=project.data["name"] if project.data else "Unknown",
                    inviter_name=inviter.data.get("full_name", "Someone") if inviter.data else "Someone",
                    role=body.role,
                )
            except Exception as e:
                print(f"Warning: Failed to send project invite email: {e}")
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{project_id}/members/{member_id}")
async def update_member(project_id: str, member_id: str, body: MemberUpdate, user_id: str = Query(...)):
    try:
        result = await service.update_member_role(
            _get_supabase(), user_id, project_id, member_id, body.role
        )
        return {"member": result}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{project_id}/members/{member_id}")
async def remove_member(project_id: str, member_id: str, user_id: str = Query(...)):
    try:
        return await service.remove_member(_get_supabase(), user_id, project_id, member_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/pending-invites")
async def list_pending_invites(project_id: str, user_id: str = Query(...)):
    invites = await service.get_pending_invites(_get_supabase(), user_id, project_id)
    return {"invites": invites}


@router.delete("/{project_id}/pending-invites/{invite_id}")
async def cancel_pending_invite(project_id: str, invite_id: str, user_id: str = Query(...)):
    try:
        return await service.delete_pending_invite(_get_supabase(), user_id, project_id, invite_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
