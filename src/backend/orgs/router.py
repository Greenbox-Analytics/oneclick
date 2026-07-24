"""FastAPI router for organizations & seat licensing. Mounted at /orgs in
main.py. Mirrors teams/router.py's structure; authz denials mostly surface
as HTTPException raised directly from within orgs/service.py (via
orgs/authz.py's require_member/require_admin), so this router only needs to
map the business-logic exceptions that aren't plain authz checks."""

import os
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_email, get_current_user_id
from orgs import projects as org_projects
from orgs import service
from orgs.models import (
    AllocateCredits,
    CreditRequestApprove,
    CreditRequestCreate,
    CreditRequestDeny,
    InviteCreate,
    MemberRoleUpdate,
    OrgCreate,
    OrgUpdate,
    ProjectMemberRoleUpdate,
    ReclaimCredits,
)
from orgs.service import (
    CreditRequestAlreadyResolvedError,
    CreditRequestNotFoundError,
    DuplicateInviteError,
    DuplicatePendingRequestError,
    InviteInvalidError,
    LastAdminError,
    PoolBalanceInsufficientError,
    ReclaimFailedError,
    SeatBalanceChangedError,
    SeatBalanceNotZeroError,
)
from subscriptions.service import licensing_enabled


def require_licensing() -> None:
    """Router-level gate (spec §10 rollback discipline): every /orgs/* route
    404s when LICENSING_ENABLED is off, so the whole surface is a true no-op
    for rollback — same idiom as the credits system's flag gates, applied at
    the router level (rather than per-endpoint) since NOTHING under /orgs
    should be reachable while the flag is off.
    """
    if not licensing_enabled():
        raise HTTPException(status_code=404, detail="Not found")


router = APIRouter(dependencies=[Depends(require_licensing)])


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


def _send_org_invite_email_background(
    db_url: str, db_key: str, org_id: str, user_id: str, email: str, role: str, existing_user: bool = False
):
    """Runs on a FastAPI BackgroundTask — its own service-role client, not
    the request's (mirrors teams.router._send_team_invite_email_background)."""
    from supabase import create_client

    from orgs.emails import send_org_invite_email

    db = create_client(db_url, db_key)
    try:
        org = db.table("organizations").select("name").eq("id", org_id).single().execute()
        inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
        send_org_invite_email(
            recipient_email=email,
            org_name=org.data["name"] if org.data else "an organization",
            inviter_name=(inviter.data or {}).get("full_name", "Someone"),
            role=role,
            existing_user=existing_user,
        )
    except Exception as exc:
        print(f"Background: failed to send org invite email: {exc}")


def _send_credit_request_email_background(
    db_url: str,
    db_key: str,
    org_id: str,
    request_id: str,
    requester_user_id: str,
    requested_credits: int | None,
    note: str | None,
):
    """Runs on a FastAPI BackgroundTask — its own service-role client, not
    the request's (mirrors _send_org_invite_email_background above).
    Recipients are every ACTIVE admin member's email, resolved via the auth
    admin API (org_members only carries user_id — mirrors
    orgs.service._resolve_user_email / registry.service._resolve_auth_email,
    the same idiom teams uses for email-by-user_id lookups elsewhere)."""
    from supabase import create_client

    from orgs.emails import send_credit_request_email

    db = create_client(db_url, db_key)
    try:
        org = db.table("organizations").select("name").eq("id", org_id).single().execute()
        requester_profile = (
            db.table("profiles").select("full_name").eq("id", requester_user_id).maybe_single().execute()
        )
        requester_name = (requester_profile.data or {}).get("full_name") if requester_profile else None
        if not requester_name:
            requester_name = service._resolve_user_email(db, requester_user_id) or "A member"

        admin_rows = (
            db.table("org_members")
            .select("user_id")
            .eq("org_id", org_id)
            .eq("role", "admin")
            .eq("status", "active")
            .execute()
        )
        admin_emails = [
            email
            for email in (service._resolve_user_email(db, row["user_id"]) for row in (admin_rows.data or []))
            if email
        ]
        if not admin_emails:
            print(f"Background: no active admin emails found for org {org_id} credit request {request_id}")
            return

        send_credit_request_email(
            recipient_emails=admin_emails,
            org_name=org.data["name"] if org.data else "your organization",
            requester_name=requester_name,
            requested_credits=requested_credits,
            note=note,
        )
    except Exception as exc:
        print(f"Background: failed to send credit request email: {exc}")


# --- Invites by token (MUST come before /{org_id} routes — same convention
# as teams/router.py, even though these currently can't collide: org_id is a
# single path segment and every route below has more segments than it) ---


@router.post("/invites/{token}/accept")
async def accept_org_invite(
    token: str, user_id: str = Depends(get_current_user_id), email: str = Depends(get_current_user_email)
):
    try:
        result = await service.accept_invite(_get_supabase(), user_id, email, token)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except InviteInvalidError as e:
        raise HTTPException(status_code=410, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # Gate on a FRESH acceptance (not an "already_accepted" replay) — same
    # duplicate-gating discipline as topup_purchased etc.
    if result.get("type") == "accepted":
        analytics_capture(user_id, "org_license_accepted", {"org_id": result.get("org_id")})
    return result


@router.post("/invites/{token}/decline")
async def decline_org_invite(
    token: str, user_id: str = Depends(get_current_user_id), email: str = Depends(get_current_user_email)
):
    try:
        return await service.decline_invite(_get_supabase(), user_id, email, token)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("")
async def create_org(body: OrgCreate, user_id: str = Depends(get_current_user_id)):
    db = _get_supabase()
    org = await service.create_org(db, user_id, body.name)
    analytics_capture(user_id, "org_created", {"org_id": org.get("id")})
    return org


@router.get("")
async def list_orgs(user_id: str = Depends(get_current_user_id)):
    return {"organizations": await service.list_my_orgs(_get_supabase(), user_id)}


@router.get("/{org_id}")
async def get_org(org_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.get_org(_get_supabase(), user_id, org_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{org_id}")
async def update_org(org_id: str, body: OrgUpdate, user_id: str = Depends(get_current_user_id)):
    fields = body.model_dump(exclude_unset=True)
    return await service.update_org(_get_supabase(), user_id, org_id, fields)


@router.post("/{org_id}/archive")
async def archive_org(org_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.archive_org(_get_supabase(), user_id, org_id)
    except SeatBalanceNotZeroError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/{org_id}/usage")
async def get_org_usage(org_id: str, user_id: str = Depends(get_current_user_id)):
    """Admin-only per-seat usage rollup (Task 7) — pool balance, cumulative
    purchased, and every seat's balance/spend/storage-vs-cap. Authz denial
    (403) is raised directly from orgs.authz.require_admin inside the
    service, same as every other admin-gated endpoint in this router."""
    return await service.get_org_usage(_get_supabase(), user_id, org_id)


# --- Members: role, offboarding, reactivation ---


@router.put("/{org_id}/members/{member_id}/role")
async def update_member_role(
    org_id: str, member_id: str, body: MemberRoleUpdate, user_id: str = Depends(get_current_user_id)
):
    try:
        return await service.update_member_role(_get_supabase(), user_id, org_id, member_id, body.role)
    except LastAdminError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{org_id}/members/{member_id}/suspend")
async def suspend_member(org_id: str, member_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        result = await service.suspend_member(_get_supabase(), user_id, org_id, member_id)
    except LastAdminError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ReclaimFailedError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    analytics_capture(user_id, "org_license_revoked", {"org_id": org_id, "member_id": member_id, "action": "suspend"})
    return result


@router.delete("/{org_id}/members/{member_id}")
async def remove_member(org_id: str, member_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        result = await service.remove_member(_get_supabase(), user_id, org_id, member_id)
    except LastAdminError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ReclaimFailedError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    analytics_capture(user_id, "org_license_revoked", {"org_id": org_id, "member_id": member_id, "action": "remove"})
    return result


@router.post("/{org_id}/members/{member_id}/reactivate")
async def reactivate_member(org_id: str, member_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.reactivate_member(_get_supabase(), user_id, org_id, member_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Allocate / reclaim (Task 4) ---


def _is_completed_transfer(result: dict) -> bool:
    """True only for a FRESH (non-duplicate) transfer_credits result —
    `{"duplicate": False, ...}`. False for a duplicate replay
    (`{"duplicate": True}`) and false for the reclaim-all no-op
    (`{"removed": 0}`, which never called the RPC at all). Analytics for
    both endpoints below is gated on this, same duplicate-gating discipline
    as Phase A's topup_purchased."""
    return result.get("duplicate") is False


@router.post("/{org_id}/members/{member_id}/allocate")
async def allocate_credits(
    org_id: str, member_id: str, body: AllocateCredits, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.allocate_credits(
            _get_supabase(), user_id, org_id, member_id, body.amount, body.idempotency_key
        )
    except PoolBalanceInsufficientError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if _is_completed_transfer(result):
        analytics_capture(
            user_id, "org_credits_allocated", {"org_id": org_id, "member_id": member_id, "amount": body.amount}
        )
    return result


@router.post("/{org_id}/members/{member_id}/reclaim")
async def reclaim_credits(
    org_id: str, member_id: str, body: ReclaimCredits, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.reclaim_credits(
            _get_supabase(), user_id, org_id, member_id, body.amount, body.idempotency_key
        )
    except SeatBalanceChangedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if _is_completed_transfer(result):
        analytics_capture(user_id, "org_credits_reclaimed", {"org_id": org_id, "member_id": member_id})
    return result


# --- Invites ---


@router.post("/{org_id}/invites")
async def invite_member(
    org_id: str, body: InviteCreate, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)
):
    db = _get_supabase()
    try:
        result = await service.invite_member(db, user_id, org_id, body.email, body.role)
    except DuplicateInviteError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analytics_capture(user_id, "org_license_invited", {"org_id": org_id, "role": body.role})
    existing_user = bool(result.get("notify_user_id"))
    background_tasks.add_task(
        _send_org_invite_email_background,
        db_url=os.getenv("VITE_SUPABASE_URL"),
        db_key=os.getenv("VITE_SUPABASE_SECRET_KEY"),
        org_id=org_id,
        user_id=user_id,
        email=body.email,
        role=body.role,
        existing_user=existing_user,
    )
    return result


@router.get("/{org_id}/invites")
async def list_invites(org_id: str, user_id: str = Depends(get_current_user_id)):
    return {"invites": await service.get_pending_invites(_get_supabase(), user_id, org_id)}


@router.delete("/{org_id}/invites/{invite_id}")
async def cancel_invite(org_id: str, invite_id: str, user_id: str = Depends(get_current_user_id)):
    return await service.cancel_invite(_get_supabase(), user_id, org_id, invite_id)


# --- Credit requests (Task 9) — member ask -> admin approve ---


@router.post("/{org_id}/credit-requests")
async def submit_credit_request(
    org_id: str,
    body: CreditRequestCreate,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    db = _get_supabase()
    try:
        result = await service.submit_credit_request(db, user_id, org_id, body.requested_credits, body.note)
    except DuplicatePendingRequestError as e:
        raise HTTPException(status_code=409, detail=str(e))

    request = result["request"]
    analytics_capture(
        user_id, "credit_request_submitted", {"org_id": org_id, "requested_credits": body.requested_credits}
    )
    background_tasks.add_task(
        _send_credit_request_email_background,
        db_url=os.getenv("VITE_SUPABASE_URL"),
        db_key=os.getenv("VITE_SUPABASE_SECRET_KEY"),
        org_id=org_id,
        request_id=request["id"],
        requester_user_id=user_id,
        requested_credits=body.requested_credits,
        note=body.note,
    )
    return request


@router.get("/{org_id}/credit-requests")
async def list_credit_requests(org_id: str, user_id: str = Depends(get_current_user_id)):
    return {"requests": await service.list_credit_requests(_get_supabase(), user_id, org_id)}


@router.post("/{org_id}/credit-requests/{request_id}/approve")
async def approve_credit_request(
    org_id: str, request_id: str, body: CreditRequestApprove, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.approve_credit_request(_get_supabase(), user_id, org_id, request_id, body.credits)
    except CreditRequestNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except CreditRequestAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PoolBalanceInsufficientError as e:
        raise HTTPException(status_code=409, detail=str(e))

    analytics_capture(
        user_id,
        "credit_request_resolved",
        {"org_id": org_id, "request_id": request_id, "status": "approved", "credits": result.get("resolved_credits")},
    )
    return result


@router.post("/{org_id}/credit-requests/{request_id}/deny")
async def deny_credit_request(
    org_id: str, request_id: str, body: CreditRequestDeny, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.deny_credit_request(_get_supabase(), user_id, org_id, request_id, body.note)
    except CreditRequestNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except CreditRequestAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e))

    analytics_capture(
        user_id,
        "credit_request_resolved",
        {"org_id": org_id, "request_id": request_id, "status": "denied", "credits": None},
    )
    return result


# --- Project links (Task 2 — spec §6 rule 1: linking = consent) ---


@router.post("/{org_id}/projects/{project_id}/link")
async def link_project(org_id: str, project_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        result = await org_projects.link_project(_get_supabase(), user_id, org_id, project_id)
    except org_projects.ProjectAlreadyLinkedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    analytics_capture(user_id, "org_project_linked", {"org_id": org_id, "project_id": project_id})
    return result


@router.delete("/{org_id}/projects/{project_id}/link")
async def unlink_project(org_id: str, project_id: str, user_id: str = Depends(get_current_user_id)):
    result = await org_projects.unlink_project(_get_supabase(), user_id, org_id, project_id)
    analytics_capture(
        user_id,
        "org_project_unlinked",
        {"org_id": org_id, "project_id": project_id, "revoked": result.get("revoked")},
    )
    return result


@router.get("/{org_id}/projects")
async def list_org_projects(org_id: str, user_id: str = Depends(get_current_user_id)):
    return {"projects": await org_projects.list_org_projects(_get_supabase(), user_id, org_id)}


# --- Admin membership management on linked projects (Task 3, rules 2-3) ---
# No analytics_capture here: `projects/router.py` doesn't import the
# analytics module at all and fires no event for add_member/
# update_member_role/remove_member, so there is no project-membership event
# to reuse (Task 3 AC 3 — reported, not invented).


@router.put("/{org_id}/projects/{project_id}/members/{member_id}")
async def set_org_project_member_role(
    org_id: str,
    project_id: str,
    member_id: str,
    body: ProjectMemberRoleUpdate,
    user_id: str = Depends(get_current_user_id),
):
    try:
        return await org_projects.set_org_project_member_role(
            _get_supabase(), user_id, org_id, project_id, member_id, body.role
        )
    except org_projects.ProjectMemberIsOwnerError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/{org_id}/projects/{project_id}/members/{member_id}")
async def remove_org_project_member(
    org_id: str, project_id: str, member_id: str, user_id: str = Depends(get_current_user_id)
):
    try:
        return await org_projects.remove_org_project_member(_get_supabase(), user_id, org_id, project_id, member_id)
    except org_projects.ProjectMemberIsOwnerError as e:
        raise HTTPException(status_code=409, detail=str(e))
