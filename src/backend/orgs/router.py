"""FastAPI router for organizations & seat licensing. Mounted at /orgs in
main.py. Mirrors teams/router.py's structure; authz denials mostly surface
as HTTPException raised directly from within orgs/service.py (via
orgs/authz.py's require_member/require_admin), so this router only needs to
map the business-logic exceptions that aren't plain authz checks."""

import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_email, get_current_user_id
from orgs import artists as org_artists
from orgs import projects as org_projects
from orgs import service
from orgs.models import (
    CreditRequestApprove,
    CreditRequestCreate,
    CreditRequestDeny,
    DissolveBody,
    InviteCreate,
    MemberCapUpdate,
    MemberRoleUpdate,
    OrgCreate,
    OrgUpdate,
    ProjectMemberRoleUpdate,
    TransferCreditsBody,
)
from orgs.service import (
    CreditRequestAlreadyResolvedError,
    CreditRequestNotFoundError,
    DuplicateInviteError,
    DuplicatePendingRequestError,
    InviteInvalidError,
    LastAdminError,
    TeamFullError,
    TeamLapsedError,
)
from orgs.standing import NoSlotError
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
    org_id: str, user_id: str, email: str, role: str, token: str, existing_user: bool = False
):
    """Runs on a FastAPI BackgroundTask (the module-global service-role client
    outlives the request scope)."""
    from orgs.emails import send_org_invite_email

    db = _get_supabase()
    try:
        org = db.table("organizations").select("name").eq("id", org_id).single().execute()
        inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
        send_org_invite_email(
            recipient_email=email,
            org_name=org.data["name"] if org.data else "an organization",
            inviter_name=(inviter.data or {}).get("full_name", "Someone"),
            role=role,
            token=token,
            existing_user=existing_user,
        )
    except Exception as exc:
        print(f"Background: failed to send org invite email: {exc}")


def _send_credit_request_email_background(
    org_id: str,
    request_id: str,
    requester_user_id: str,
    requested_cap: int | None,
    note: str | None,
):
    """Runs on a FastAPI BackgroundTask (mirrors _send_org_invite_email_background
    above). Recipients are every ACTIVE admin member's email, resolved via the auth
    admin API (org_members only carries user_id — mirrors
    orgs.service._resolve_user_email / registry.service._resolve_auth_email,
    the same idiom teams uses for email-by-user_id lookups elsewhere)."""
    from orgs.emails import send_credit_request_email

    db = _get_supabase()
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
            requested_cap=requested_cap,
            note=note,
        )
    except Exception as exc:
        print(f"Background: failed to send credit request email: {exc}")


def _send_credit_request_resolved_email_background(
    org_id: str,
    org_member_id: str,
    approved: bool,
    new_cap: int | None,
    note: str | None,
):
    """Notify the requesting member that an admin resolved their cap-raise
    request. Runs on a FastAPI BackgroundTask (mirrors
    _send_credit_request_email_background). The member's email comes
    off their org_members row (denormalized at invite-accept), falling back to
    the auth admin API for legacy/creator rows."""
    from orgs.emails import send_credit_request_resolved_email

    db = _get_supabase()
    try:
        org = db.table("organizations").select("name").eq("id", org_id).single().execute()
        member = (
            db.table("org_members")
            .select("user_id, email")
            .eq("id", org_member_id)
            .eq("org_id", org_id)
            .maybe_single()
            .execute()
        )
        row = member.data if member else None
        if not row:
            print(f"Background: credit-request member {org_member_id} not found in org {org_id} — skipping email")
            return
        email = row.get("email") or service._resolve_user_email(db, row["user_id"])
        if not email:
            print(f"Background: no email resolvable for org member {org_member_id} — skipping resolved email")
            return
        send_credit_request_resolved_email(
            recipient_email=email,
            org_name=org.data["name"] if org.data else "your organization",
            approved=approved,
            new_cap=new_cap,
            note=note,
        )
    except Exception as exc:
        print(f"Background: failed to send credit request resolved email: {exc}")


def _notify_billing_reverted_background(
    org_id: str,
    member_user_ids: list[str] | None = None,
):
    """Tell offboarded member(s) their Msanii usage now bills to their personal
    plan (audit FIX: suspend/remove/archive used to be silent). Runs on a
    FastAPI BackgroundTask. `member_user_ids`
    None means "every ACTIVE member" (the org-archive case — archiving doesn't
    transition seat statuses, so active seats are exactly who just lost
    coverage). Each member gets an email (orgs/emails.py) plus an in-app
    notification on the unified `notifications` table — type 'status_change'
    (an existing CHECK-allowed type) with entity_type='org'."""
    from orgs.emails import send_billing_reverted_email

    db = _get_supabase()
    try:
        org_name = service._org_name(db, org_id, "your organization")

        query = db.table("org_members").select("user_id, email").eq("org_id", org_id)
        if member_user_ids is None:
            query = query.eq("status", "active")
        else:
            query = query.in_("user_id", member_user_ids)
        rows = query.execute().data or []

        for row in rows:
            user_id = row.get("user_id")
            if not user_id:
                continue
            email = row.get("email") or service._resolve_user_email(db, user_id)
            if email:
                send_billing_reverted_email(recipient_email=email, org_name=org_name)
            try:
                db.table("notifications").insert(
                    {
                        "user_id": user_id,
                        "type": "status_change",
                        "title": "Billing moved to your personal plan",
                        "message": (
                            f'Your seat on "{org_name}" is no longer active — '
                            "your Msanii usage now bills to your personal plan."
                        ),
                        "entity_type": "org",
                        "entity_id": org_id,
                        "metadata": {"org_id": org_id},
                    }
                ).execute()
            except Exception as exc:
                print(f"Background: failed to write billing-reverted notification for {user_id}: {exc}")
    except Exception as exc:
        print(f"Background: failed to send billing-reverted notifications for org {org_id}: {exc}")


# --- Invites by token (MUST come before /{org_id} routes — same convention
# as teams/router.py, even though these currently can't collide: org_id is a
# single path segment and every route below has more segments than it) ---


@router.get("/invites/{token}/preview")
async def get_org_invite_preview(token: str):
    """Claim-page preview: name what the user is joining BEFORE they commit
    (the OrgInviteClaim page previously had nothing to show but generic copy).

    DELIBERATELY UNAUTHENTICATED (stated decision): the claim page renders
    before sign-in, and the unguessable token — delivered only to the invited
    email — is the capability. The body is minimal on purpose: org name and
    kind, both of which the invite email already told the recipient. Nothing
    else (no invitee email, inviter, role, or expiry) is
    exposed to a bare token holder. 404 for an unknown, expired, or
    already-resolved token — email-match and acceptance stay with the
    authed accept endpoint."""
    db = _get_supabase()
    invite = await service.get_invite_by_token(db, token)
    if not invite or invite.get("status") != "pending":
        raise HTTPException(status_code=404, detail="Invite not found")
    expires_at = invite.get("expires_at")
    if expires_at and datetime.fromisoformat(expires_at) < datetime.now(UTC):
        raise HTTPException(status_code=404, detail="Invite not found")

    org = db.table("organizations").select("name, kind").eq("id", invite["org_id"]).maybe_single().execute()
    return {
        "orgName": (org.data or {}).get("name") if org else None,
        # "self_serve" | "enterprise" | None (pre-migration row) — still minimal
        # and still unauthenticated: the invite email already says team vs
        # organization, so this reveals nothing the recipient doesn't have.
        "kind": (org.data or {}).get("kind") if org else None,
    }


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
        try:
            service.create_org_join_notifications(_get_supabase(), result["org_id"], user_id, email)
        except Exception as exc:
            # The seat is already active — never fail an accepted invite over a
            # notification write.
            print(f"Failed to create org join notifications for {user_id}: {exc}")
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
    try:
        org = await service.create_org(db, user_id, body.name)
    except NoSlotError as e:
        raise HTTPException(status_code=402, detail={"reason": str(e), "upgradeRequired": True})
    # `POST /orgs` is self-serve-only now (enterprise creation moved to
    # `POST /admin/orgs`) — event name matches spec §9's `team_created`,
    # not the pre-self-serve `org_created` name.
    analytics_capture(user_id, "team_created", {"org_id": org.get("id")})
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


@router.get("/{org_id}/members")
async def list_org_members(org_id: str, user_id: str = Depends(get_current_user_id)):
    """Member-visible roster (names/avatars, no emails) for board pickers.
    `authz.require_member` raises the 404 itself, so no try/except here."""
    return {"members": await service.list_org_roster(_get_supabase(), user_id, org_id)}


@router.put("/{org_id}")
async def update_org(org_id: str, body: OrgUpdate, user_id: str = Depends(get_current_user_id)):
    fields = body.model_dump(exclude_unset=True)
    return await service.update_org(_get_supabase(), user_id, org_id, fields)


@router.post("/{org_id}/archive")
async def archive_org(org_id: str, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)):
    """Whatever the POOL still holds survives archiving — disposing of it is a
    support decision (admin clawback), so there is no balance precondition and
    nothing here can 409. Every active member is notified their usage now
    bills to their personal plan (background, best-effort)."""
    result = await service.archive_org(_get_supabase(), user_id, org_id)
    background_tasks.add_task(
        _notify_billing_reverted_background,
        org_id=org_id,
        member_user_ids=None,  # archive: notify every ACTIVE member
    )
    analytics_capture(user_id, "team_archived", {"org_id": org_id})
    return result


@router.post("/{org_id}/unarchive")
async def unarchive_org(org_id: str, user_id: str = Depends(get_current_user_id)):
    """Self-serve reactivation of an archived org (active admin, self_serve
    only — enterprise 409). Requires a free slot AND the reactivation storage
    guard, since an archived team holds no slot but its bytes still count.
    Does NOT restore org-granted project memberships or members' billing
    contexts torn down at archive time — see service.unarchive_org's
    docstring for that pre-existing limitation."""
    try:
        result = await service.unarchive_org(_get_supabase(), user_id, org_id)
    except NoSlotError as e:
        raise HTTPException(status_code=402, detail={"reason": str(e), "upgradeRequired": True})
    analytics_capture(user_id, "team_unarchived", {"org_id": org_id})
    return result


@router.get("/{org_id}/dissolve-preview")
async def dissolve_preview(org_id: str, user_id: str = Depends(get_current_user_id)):
    """What dissolving would do: which person each team artist reverts to
    (`fallback: true` means the artist's creator no longer holds a seat, so it
    comes to YOU), how many purchased credits are forfeited, how much monthly
    dispersal simply expires with the team, and how many seats close. Admin +
    self_serve only, enforced in the service alongside the execute."""
    return await service.dissolve_preview(_get_supabase(), user_id, org_id)


@router.post("/{org_id}/dissolve")
async def dissolve_org(org_id: str, body: DissolveBody, user_id: str = Depends(get_current_user_id)):
    """Terminal, one-way, name-confirmed (400 on a mismatch, nothing written).
    Idempotent: a second call returns {"already": true} — gate the event on
    that, same duplicate-discipline as accept_invite, so a double-click can't
    report two dissolutions. A 500 here is RETRY-SAFE by construction: it can
    only come from the artist reverts or the terminal patch, whatever landed
    stays landed, and re-POSTing the same body finishes the job."""
    result = await service.dissolve_org(_get_supabase(), user_id, org_id, body.confirm_name)
    if not result.get("already"):
        analytics_capture(user_id, "team_dissolved", {"org_id": org_id})
    return result


@router.post("/{org_id}/coverage/claim")
async def claim_coverage(org_id: str, user_id: str = Depends(get_current_user_id)):
    """An admin with a free slot claims coverage of this org (spec §2 rev 2,
    Task 5) — claiming an org you already actively cover is a no-op 200."""
    db = _get_supabase()
    try:
        result = await service.claim_coverage(db, user_id, org_id)
    except NoSlotError as e:
        raise HTTPException(status_code=402, detail={"reason": str(e), "upgradeRequired": True})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    analytics_capture(user_id, "coverage_claimed", {"org_id": org_id})
    return result


@router.post("/{org_id}/coverage/release")
async def release_coverage(org_id: str, user_id: str = Depends(get_current_user_id)):
    """Only the CURRENT coverer may release. covered_by stays put — the sweep
    is the single writer that starts grace."""
    result = await service.release_coverage(_get_supabase(), user_id, org_id)
    analytics_capture(user_id, "coverage_released", {"org_id": org_id})
    return result


@router.get("/{org_id}/usage")
async def get_org_usage(org_id: str, user_id: str = Depends(get_current_user_id)):
    """Admin-only per-member usage rollup — pool balance, cumulative paid-in,
    and every member's cap / spend-against-cap / storage-vs-cap. Authz denial
    (403) is raised directly from orgs.authz.require_admin inside the
    service, same as every other admin-gated endpoint in this router."""
    return await service.get_org_usage(_get_supabase(), user_id, org_id)


@router.get("/{org_id}/ledger")
async def get_org_ledger(org_id: str, user_id: str = Depends(get_current_user_id)):
    """Admin-only: newest 50 pool ledger rows for the org billing console's
    Pool activity feed (Task 15, spec §6). Authz denial (403) is raised
    directly from orgs.authz.require_admin inside the service."""
    return {"ledger": await service.get_org_ledger(_get_supabase(), user_id, org_id)}


@router.post("/{org_id}/transfer-credits")
async def transfer_credits_to_pool(org_id: str, body: TransferCreditsBody, user_id: str = Depends(get_current_user_id)):
    """Owner-requested funding inlet (spec §4.1): an active admin moves
    credits from their OWN personal reserve into this org's pool. 409s
    (managed-by-Msanii, archived, dissolved, insufficient reserve) are all
    raised directly from orgs.service.transfer_credits_to_pool.

    Idempotent: a retried request (same client-side dedupe) surfaces the
    RPC's `duplicate: true` result as a plain 200. Analytics fires only for a
    FRESH transfer — same duplicate-gating discipline as dissolve_org /
    accept_invite, so a retry can't report two transfers."""
    result = await service.transfer_credits_to_pool(_get_supabase(), user_id, org_id, body.amount)
    if not result.get("duplicate"):
        analytics_capture(user_id, "credits_transferred", {"org_id": org_id, "amount": body.amount})
    return result


@router.post("/{org_id}/cancel-topup")
async def cancel_topup(org_id: str, user_id: str = Depends(get_current_user_id)):
    """Stop this team's recurring credit top-up. ANY active admin may cancel,
    not just the one whose card it is — an admin who left shouldn't be able to
    strand the team on a subscription nobody else can stop.

    Returns {"canceled": false} when there was nothing to cancel (idempotent
    double-click), and the event fires only on a real cancel — same
    duplicate-gating discipline as dissolve_org / transfer-credits. A Stripe
    failure propagates (500) with the org's columns intact so the admin can
    retry; already-purchased credits are never clawed back."""
    from orgs.authz import require_admin

    db = _get_supabase()
    require_admin(db, user_id, org_id)
    # Task 17 RULING: deliberately NOT gated by _require_live_org. This
    # endpoint is the documented RETRY path for a failed Stripe cancel
    # (service.py's _cancel_topup_if_purchaser and dissolve_org docstrings
    # both name it), and archive_org never cancels the top-up itself — so an
    # archived org keeps billing the admin's card every month with no other
    # way to stop it, and a dissolved org whose step-2 Stripe cancel failed
    # has no escape at all (there's no unarchive from dissolved). Canceling a
    # live charge on a dead org is remedial, same logic as release_coverage's
    # exemption, stronger because real money is bleeding.
    canceled = service.cancel_topup(db, org_id)
    if canceled:
        analytics_capture(user_id, "org_topup_canceled", {"org_id": org_id, "trigger": "manual"})
    return {"canceled": canceled}


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


def _schedule_billing_reverted_notice(background_tasks: BackgroundTasks, org_id: str, member_user_id: str | None):
    """Shared by suspend/remove: the offboarded member is told their usage now
    bills to their personal plan (background, best-effort)."""
    if not member_user_id:
        return
    background_tasks.add_task(
        _notify_billing_reverted_background,
        org_id=org_id,
        member_user_ids=[member_user_id],
    )


@router.post("/{org_id}/members/{member_id}/suspend")
async def suspend_member(
    org_id: str, member_id: str, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.suspend_member(_get_supabase(), user_id, org_id, member_id)
    except LastAdminError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    analytics_capture(user_id, "org_license_revoked", {"org_id": org_id, "member_id": member_id, "action": "suspend"})
    _schedule_billing_reverted_notice(background_tasks, org_id, result.get("user_id"))
    return result


@router.delete("/{org_id}/members/{member_id}")
async def remove_member(
    org_id: str, member_id: str, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id)
):
    try:
        result = await service.remove_member(_get_supabase(), user_id, org_id, member_id)
    except LastAdminError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    analytics_capture(user_id, "org_license_revoked", {"org_id": org_id, "member_id": member_id, "action": "remove"})
    _schedule_billing_reverted_notice(background_tasks, org_id, result.get("user_id"))
    return result


@router.post("/{org_id}/members/{member_id}/reactivate")
async def reactivate_member(org_id: str, member_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return await service.reactivate_member(_get_supabase(), user_id, org_id, member_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Caps: per-member ceilings and the org's contract dials ---


@router.put("/{org_id}/members/{member_id}/cap")
async def set_member_cap(
    org_id: str, member_id: str, body: MemberCapUpdate, user_id: str = Depends(get_current_user_id)
):
    """Set one member's monthly ceiling on the pool. Nothing moves, so there is
    no duplicate-transfer case to gate analytics on."""
    try:
        result = await service.set_member_cap(_get_supabase(), user_id, org_id, member_id, body.cap)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    analytics_capture(user_id, "org_member_cap_set", {"org_id": org_id, "member_id": member_id, "cap": body.cap})
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
    except TeamLapsedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TeamFullError as e:
        analytics_capture(user_id, "team_seat_wall_hit", {"org_id": org_id, "limit": e.limit, "next_step": e.next_step})
        raise HTTPException(status_code=402, detail={"reason": str(e), "limit": e.limit, "nextStep": e.next_step})

    analytics_capture(user_id, "org_license_invited", {"org_id": org_id, "role": body.role})
    invite = result.get("invite") or {}
    token = invite.get("token")
    existing_user = bool(result.get("notify_user_id"))

    # Existing user -> also drop an in-app Accept/Decline row, same as teams
    # (teams/router.py invite_member). A brand-new invitee has no account to
    # notify, so for them the emailed claim link is the whole path in.
    if existing_user and token:
        try:
            service.create_org_invite_notification(db, result["notify_user_id"], org_id, user_id, token)
        except Exception as exc:
            print(f"Failed to create org invite notification: {exc}")

    background_tasks.add_task(
        _send_org_invite_email_background,
        org_id=org_id,
        user_id=user_id,
        email=body.email,
        role=body.role,
        token=token,
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
        result = await service.submit_credit_request(db, user_id, org_id, body.requested_cap, body.note)
    except DuplicatePendingRequestError as e:
        raise HTTPException(status_code=409, detail=str(e))

    request = result["request"]
    analytics_capture(user_id, "credit_request_submitted", {"org_id": org_id, "requested_cap": body.requested_cap})
    background_tasks.add_task(
        _send_credit_request_email_background,
        org_id=org_id,
        request_id=request["id"],
        requester_user_id=user_id,
        requested_cap=body.requested_cap,
        note=body.note,
    )
    return request


@router.get("/{org_id}/credit-requests")
async def list_credit_requests(org_id: str, user_id: str = Depends(get_current_user_id)):
    return {"requests": await service.list_credit_requests(_get_supabase(), user_id, org_id)}


@router.post("/{org_id}/credit-requests/{request_id}/approve")
async def approve_credit_request(
    org_id: str,
    request_id: str,
    body: CreditRequestApprove,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    try:
        result = await service.approve_credit_request(_get_supabase(), user_id, org_id, request_id, body.cap)
    except CreditRequestNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except CreditRequestAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e))

    analytics_capture(
        user_id,
        "credit_request_resolved",
        {"org_id": org_id, "request_id": request_id, "status": "approved", "cap": result.get("resolved_cap")},
    )
    if result.get("org_member_id"):
        background_tasks.add_task(
            _send_credit_request_resolved_email_background,
            org_id=org_id,
            org_member_id=result["org_member_id"],
            approved=True,
            new_cap=result.get("resolved_cap"),
            note=None,
        )
    return result


@router.post("/{org_id}/credit-requests/{request_id}/deny")
async def deny_credit_request(
    org_id: str,
    request_id: str,
    body: CreditRequestDeny,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
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
    if result.get("org_member_id"):
        background_tasks.add_task(
            _send_credit_request_resolved_email_background,
            org_id=org_id,
            org_member_id=result["org_member_id"],
            approved=False,
            new_cap=None,
            note=body.note,
        )
    return result


# --- Project links (Task 2 — spec §6 rule 1: linking = consent) ---


@router.get("/{org_id}/projects")
async def list_org_projects(org_id: str, user_id: str = Depends(get_current_user_id)):
    return {"projects": await org_projects.list_org_projects(_get_supabase(), user_id, org_id)}


@router.post("/{org_id}/artists/{artist_id}/transfer")
async def transfer_artist_to_team(org_id: str, artist_id: str, user_id: str = Depends(get_current_user_id)):
    """Move a personal artist into this team. One-way — see orgs/artists.py."""
    try:
        result = await org_artists.transfer_artist_to_team(_get_supabase(), user_id, org_id, artist_id)
    except org_artists.ArtistAlreadyTeamOwnedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    analytics_capture(user_id, "artist_transferred", {"org_id": org_id, "artist_id": artist_id})
    return result


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
