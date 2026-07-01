"""FastAPI router for teams & membership. Mounted at /teams in main.py.
Mirrors projects/router.py (background-task email, exception->HTTP mapping)."""

import os
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_email, get_current_user_id
from teams import service
from teams.models import InviteCreate, MemberRoleUpdate, TeamCreate, TeamUpdate
from teams.service import DuplicateInviteError, InviteInvalidError, LastAdminError

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


def _send_team_invite_email_background(db_url: str, db_key: str, team_id: str, user_id: str, email: str, role: str):
    from supabase import create_client

    from teams.emails import send_team_invite_email

    db = create_client(db_url, db_key)
    try:
        team = db.table("teams").select("name").eq("id", team_id).single().execute()
        inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
        send_team_invite_email(
            recipient_email=email,
            team_name=team.data["name"] if team.data else "a team",
            inviter_name=(inviter.data or {}).get("full_name", "Someone"),
            role=role,
        )
    except Exception as exc:
        print(f"Background: failed to send team invite email: {exc}")


def _schedule_invite_email(bg: BackgroundTasks, team_id: str, user_id: str, email: str, role: str):
    bg.add_task(
        _send_team_invite_email_background,
        db_url=os.getenv("VITE_SUPABASE_URL"),
        db_key=os.getenv("VITE_SUPABASE_SECRET_KEY"),
        team_id=team_id,
        user_id=user_id,
        email=email,
        role=role,
    )


# --- Invites by token (MUST come before /{team_id} routes) ---


@router.post("/invites/{token}/accept")
async def accept_invite(
    token: str, user_id: str = Depends(get_current_user_id), email: str = Depends(get_current_user_email)
):
    try:
        return await service.accept_invite(_get_supabase(), user_id, email, token)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except InviteInvalidError as e:
        raise HTTPException(status_code=410, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/invites/{token}/decline")
async def decline_invite(
    token: str, user_id: str = Depends(get_current_user_id), email: str = Depends(get_current_user_email)
):
    try:
        return await service.decline_invite(_get_supabase(), user_id, email, token)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Teams ---


@router.get("")
async def list_teams(user_id: str = Depends(get_current_user_id)):
    return {"teams": await service.list_my_teams(_get_supabase(), user_id)}


@router.post("")
async def create_team(body: TeamCreate, user_id: str = Depends(get_current_user_id)):
    return await service.create_team(_get_supabase(), user_id, body.name, body.description)


@router.get("/{team_id}")
async def get_team(team_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.get_team(_get_supabase(), user_id, team_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{team_id}")
async def update_team(team_id: str, body: TeamUpdate, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.update_team(_get_supabase(), user_id, team_id, body.model_dump(exclude_none=True))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{team_id}")
async def archive_team(team_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.archive_team(_get_supabase(), user_id, team_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


# --- Members ---


@router.get("/{team_id}/members")
async def list_members(team_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return {"members": await service.list_members(_get_supabase(), user_id, team_id)}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.put("/{team_id}/members/{member_id}")
async def update_member(
    team_id: str, member_id: str, body: MemberRoleUpdate, user_id: str = Depends(get_current_user_id)
):
    try:
        return {"member": await service.update_member_role(_get_supabase(), user_id, team_id, member_id, body.role)}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except LastAdminError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{team_id}/members/{member_id}")
async def remove_member(team_id: str, member_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.remove_member(_get_supabase(), user_id, team_id, member_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except LastAdminError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Invites ---


@router.post("/{team_id}/invites")
async def invite_member(
    team_id: str, body: InviteCreate, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)
):
    db = _get_supabase()
    try:
        result = await service.invite_member(db, user_id, team_id, body.email, body.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except DuplicateInviteError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    invite = result.get("invite") or {}
    if result.get("notify_user_id"):
        # Existing user → in-app Accept/Decline notification ONLY. No signup email — its
        # "Sign Up to Join" copy + tokenless /auth link is wrong for an existing account.
        try:
            team = db.table("teams").select("name").eq("id", team_id).single().execute()
            inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
            service.create_team_invite_notification(
                db,
                target_user_id=result["notify_user_id"],
                team_id=team_id,
                team_name=(team.data or {}).get("name", "a team"),
                inviter_name=(inviter.data or {}).get("full_name", "Someone"),
                token=invite.get("token"),
            )
        except Exception as exc:
            print(f"Failed to create team_invite notification: {exc}")
    else:
        # New user (no account yet) → email invite; the signup trigger auto-converts on join.
        _schedule_invite_email(background_tasks, team_id=team_id, user_id=user_id, email=body.email, role=body.role)
    return result


@router.get("/{team_id}/invites")
async def list_invites(team_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return {"invites": await service.get_pending_invites(_get_supabase(), user_id, team_id)}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{team_id}/invites/{invite_id}")
async def cancel_invite(team_id: str, invite_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.cancel_invite(_get_supabase(), user_id, team_id, invite_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
