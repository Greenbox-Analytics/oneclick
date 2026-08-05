"""Organizations & seat-licensing business logic (Licensing Phase B).

Mirrors teams/service.py. The backend uses the service-role client (RLS-
bypassing), so authz goes through orgs/authz.py per call — those helpers
raise HTTPException directly (mirrors boards/service.py's use of
teams.authz.require_board_access/require_team_admin), so most endpoints here
need no try/except of their own for authz denials.

create_org is a single insert — the creator's admin membership is added
atomically by the auto_create_org_admin DB trigger (migration
20260721000001_licensing_core.sql), keyed on created_by since auth.uid() is
NULL under the service role. This service module must NEVER insert an
org_members row for the creator itself: that invariant (an org can never
exist without an admin) belongs to the trigger, one write, no orphan-org
window.
"""

import os
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from supabase import Client

from orgs import authz, wallets


class DuplicateInviteError(Exception):
    """Already an active member, or a duplicate pending invite for (org, email)."""


class LastAdminError(Exception):
    """The DB last-admin guard (org_members_admin_guard_trigger) rejected a
    role change / suspend / remove that would leave the org with no active
    admin."""


class InviteInvalidError(Exception):
    """Invite is expired, declined, or otherwise no longer actionable."""


class DuplicatePendingRequestError(Exception):
    """The DB partial unique index (org_member_id, WHERE status='pending')
    rejected a second open credit request for this seat. Mapped by the
    router to 409 "You already have a pending request."."""


class CreditRequestNotFoundError(Exception):
    """No credit_requests row with that id in that org. Mapped by the
    router to 404 (same no-existence-oracle stance as require_member)."""


class CreditRequestAlreadyResolvedError(Exception):
    """The credit_requests row is no longer status='pending' — a second
    approve/deny on an already-resolved request. Mapped by the router to
    409."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _epoch(iso_ts: str) -> int:
    """Integer epoch seconds for a stored TIMESTAMPTZ string — the offboard
    reclaim's request_id keys off THIS (the STORED revoked_at), never a
    freshly computed now() (rule 5)."""
    return int(datetime.fromisoformat(iso_ts).timestamp())


def _is_last_admin_error(exc: Exception) -> bool:
    return "only admin" in str(exc).lower()


def _find_user_id_by_email(db: Client, email: str) -> str | None:
    """Look up a user id by email via the get_user_id_by_email(lookup_email) SECURITY DEFINER
    RPC (pre-existing, from the registry migration 20260329000000; also used by
    teams/service.py and projects/service.py — this project's PostgREST does not expose the
    `auth` schema, so db.schema("auth") fails (PGRST106) and the RPC reads auth.users on the
    backend's behalf). Returns None if unknown.
    """
    try:
        result = db.rpc("get_user_id_by_email", {"lookup_email": email}).execute()
    except Exception as exc:
        print(f"get_user_id_by_email failed for {email!r}: {exc}")
        return None
    data = result.data
    if isinstance(data, list):  # defensive: some PostgREST/client versions wrap scalars
        data = data[0] if data else None
    return data or None


def _resolve_user_email(db: Client, user_id: str) -> str | None:
    """Return the verified email for `user_id` from auth.users (Supabase).

    `profiles` doesn't store email — mirrors registry.service._resolve_auth_email
    exactly (this project's PostgREST doesn't expose the `auth` schema, so the
    auth admin API is the only path). Used to resolve the credit-request
    notification's admin recipient list from org_members rows, which only
    carry user_id. Returns None on lookup failure rather than raising so
    callers can skip an unresolvable recipient instead of failing the whole
    notification.
    """
    try:
        res = db.auth.admin.get_user_by_id(user_id)
        return res.user.email if res and res.user else None
    except Exception as exc:
        print(f"Failed to resolve auth email for user_id={user_id}: {exc}")
        return None


def _default_min_initial_purchase_credits() -> int:
    """Platform default activation floor when an org's own
    min_initial_purchase_credits is NULL (spec §4)."""
    return int(os.getenv("ENTERPRISE_MIN_INITIAL_CREDITS", "10000"))


async def create_org(db: Client, user_id: str, name: str) -> dict:
    """Create an org. Status defaults to 'pending' in the DB (CHECK default).
    A single insert — see module docstring: the auto_create_org_admin trigger
    adds the creator as admin; this function must NOT insert into
    org_members itself. Any authed user may create an org (it confers
    nothing while pending)."""
    res = db.table("organizations").insert({"name": name, "created_by": user_id}).execute()
    org = res.data[0] if res.data else None
    if not org:
        raise RuntimeError("Failed to create organization")
    # The creator is always the admin (auto_create_org_admin), so annotate
    # my_role like list_my_orgs — no follow-up GET needed to learn the role.
    return {**org, "my_role": "admin"}


async def list_my_orgs(db: Client, user_id: str) -> list[dict]:
    """Orgs where the caller holds ANY org_members row with status != 'removed',
    each annotated with my_role/my_status. Archived orgs are NOT filtered out
    here (unlike teams' list_my_teams excluding archived teams) — an admin
    still needs to see/manage an archived org (e.g. the frozen-pool support
    case), and Task 2 has no dedicated archived-orgs view."""
    memberships = (
        db.table("org_members").select("org_id, role, status").eq("user_id", user_id).neq("status", "removed").execute()
    )
    rows = memberships.data or []
    if not rows:
        return []
    info_by_org = {m["org_id"]: {"role": m["role"], "status": m["status"]} for m in rows}
    orgs = db.table("organizations").select("*").in_("id", list(info_by_org.keys())).order("created_at").execute()
    out = []
    for o in orgs.data or []:
        info = info_by_org.get(o["id"], {})
        o["my_role"] = info.get("role")
        o["my_status"] = info.get("status")
        out.append(o)
    return out


async def get_org(db: Client, user_id: str, org_id: str) -> dict:
    """Fetch an org the caller belongs to, with computed pool/activation
    fields. Member-only (authz.require_member 404s for non-members — same
    response as a nonexistent org, no existence oracle)."""
    authz.require_member(db, user_id, org_id)

    res = db.table("organizations").select("*").eq("id", org_id).maybe_single().execute()
    org = res.data if res else None
    if not org:
        raise ValueError("Organization not found")

    member_row = (
        db.table("org_members")
        .select("role")
        .eq("org_id", org_id)
        .eq("user_id", user_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    my_role = (member_row.data or {}).get("role") if member_row else None

    wallet_res = (
        db.table("credit_wallets")
        .select("id, bundle_balance, reserve_balance")
        .eq("owner_type", "org")
        .eq("owner_id", org_id)
        .execute()
    )
    wallet_rows = wallet_res.data or []
    wallet = wallet_rows[0] if wallet_rows else None
    pool_balance = (wallet.get("bundle_balance", 0) + wallet.get("reserve_balance", 0)) if wallet else 0

    cumulative_paid_in = wallets.cumulative_paid_in(db, wallet["id"]) if wallet else 0

    effective_min = org.get("min_initial_purchase_credits") or _default_min_initial_purchase_credits()
    remaining_to_activate = max(0, effective_min - cumulative_paid_in)

    member_count_res = (
        db.table("org_members").select("id", count="exact").eq("org_id", org_id).eq("status", "active").execute()
    )
    member_count = member_count_res.count or 0

    return {
        **org,
        "my_role": my_role,
        "pool_balance": pool_balance,
        "cumulative_paid_in": cumulative_paid_in,
        "remaining_to_activate": remaining_to_activate,
        "member_count": member_count,
    }


async def update_org(db: Client, user_id: str, org_id: str, fields: dict) -> dict:
    """Update org name / default_member_cap. Admin only.

    `fields` is expected to come from `OrgUpdate.model_dump(exclude_unset=True)`
    at the router — keys explicitly present in the request (including an
    explicit `None` for default_member_cap, clearing members to uncapped) are
    written; omitted keys are left untouched. The dispersal has its own endpoint
    (set_org_dispersal) because it is the contract, not a display preference."""
    authz.require_admin(db, user_id, org_id)
    if not fields:
        res = db.table("organizations").select("*").eq("id", org_id).maybe_single().execute()
        return res.data
    res = db.table("organizations").update(fields).eq("id", org_id).execute()
    return res.data[0] if res.data else None


def _teardown_archived_org_grants(db: Client, org_id: str) -> None:
    """Licensing Phase C, Task 4 (rule 12): drop every `project_members` row
    THIS org granted, once an org is archived. Runs AFTER `archived_at` has
    already landed (the load-bearing write for `archive_org`) and is
    best-effort/never-raising, mirroring `_offboard`'s money-first-then-cleanup
    posture: a teardown failure here must not undo or block an already-committed
    archive.

    `revoke_org_granted_memberships` (this codebase's single implementation of
    rule 3, imported lazily to avoid a module-level import cycle — `orgs.projects`
    imports `_resolve_user_email` from this module at its own top level) is
    called org-scoped only (no user_id/project_id narrowing), i.e. every
    `project_members` row THIS org ever granted, on every project it touched.
    Organic rows are untouched by construction (the helper's `org_id` filter is
    the entire mechanism).

    There is no link row to clean up any more: `org_project_links` was retired
    in 20260804000001. An archived org's `artists.team_id` rows are deliberately
    LEFT in place — `can_access_artist` already denies on `archived_at`, so the
    roster is inert without being destroyed, and support can still reach it.
    """
    try:
        from orgs.projects import revoke_org_granted_memberships

        revoke_org_granted_memberships(db, org_id)
    except Exception as exc:
        print(f"archive_org: revoke_org_granted_memberships failed org_id={org_id}: {exc}")


async def archive_org(db: Client, user_id: str, org_id: str) -> dict:
    """Archive an org. Admin only. Sets archived_at, then tears down every access
    grant and link this org ever created (Task 4, rule 12) — see
    `_teardown_archived_org_grants`.

    No balance precondition: members hold no credits, so there is nothing
    stranded to reclaim first. Whatever is left in the POOL stays in the pool —
    an archived org's money is a support/refund decision (see the admin
    clawback endpoint), never something archiving silently disposes of.
    """
    authz.require_admin(db, user_id, org_id)

    res = db.table("organizations").update({"archived_at": _now_iso()}).eq("id", org_id).execute()
    _teardown_archived_org_grants(db, org_id)
    return res.data[0] if res.data else {"archived": org_id}


async def get_org_usage(db: Client, user_id: str, org_id: str) -> dict:
    """Admin-only per-member usage rollup for the org admin console.

    One pool, so one wallet read. Per-member spend comes from the pool's ledger
    grouped by `metadata.org_member_id` (written by debit_credits) rather than
    from per-member wallets, which no longer exist. The scan is floored on the
    pool's `period_start` so "spent" means THIS period — the same window a
    member's cap resets on, which is the only comparison an admin can act on.

    Also surfaces the storage ceiling: the finite ENTERPRISE_SEAT_STORAGE_BYTES
    must be visible in the console BEFORE an upload fails, not just at the wall.

    Member visibility: every non-removed member is included. A removed member is
    included only if they spent something this period — an admin reading the
    console wants the month's spend attributed, including to someone who has
    since left.

    Email resolution: `org_members.email` is captured going forward at
    invite-accept, so the common case reads it straight off the row with zero
    auth-admin calls. A NULL email (a pre-migration row, or a creator row — the
    auto_create_org_admin trigger never goes through accept_invite) falls back to
    `_resolve_user_email` ONCE and is written back onto the row, best-effort and
    deliberately non-raising: a failed heal must never break the usage read.
    """
    authz.require_admin(db, user_id, org_id)

    members_res = (
        db.table("org_members")
        .select("id, user_id, role, status, email, monthly_cap, cap_used, cap_period_end")
        .eq("org_id", org_id)
        .execute()
    )
    members = members_res.data or []

    org_res = (
        db.table("organizations").select("default_member_cap, monthly_dispersal_credits").eq("id", org_id).execute()
    )
    org_row = (org_res.data or [{}])[0]
    default_cap = org_row.get("default_member_cap")

    pool_rows = (
        db.table("credit_wallets")
        .select("id, bundle_balance, reserve_balance, period_start, period_end")
        .eq("owner_type", "org")
        .eq("owner_id", org_id)
        .execute()
        .data
        or []
    )
    pool_wallet = pool_rows[0] if pool_rows else None
    pool_balance = (pool_wallet.get("bundle_balance", 0) + pool_wallet.get("reserve_balance", 0)) if pool_wallet else 0

    ledger_rows: list[dict] = []
    if pool_wallet:
        query = db.table("credit_ledger").select("delta, kind, metadata").eq("wallet_id", pool_wallet["id"])
        if pool_wallet.get("period_start"):
            query = query.gte("created_at", pool_wallet["period_start"])
        ledger_rows = query.execute().data or []

    cumulative_paid_in = wallets.cumulative_paid_in(db, pool_wallet["id"]) if pool_wallet else 0

    # spentThisPeriod is sum(|delta|) over kind='debit' rows grouped by the
    # member who spent them. Pools have no overage path (rule 8), so there is
    # no 'overage_debit' kind to fold in as there is on the personal view.
    spent_by_member: dict[str, int] = {}
    for r in ledger_rows:
        if r.get("kind") != "debit":
            continue
        member_id = (r.get("metadata") or {}).get("org_member_id")
        if not member_id:
            continue
        spent_by_member[member_id] = spent_by_member.get(member_id, 0) + abs(r.get("delta", 0))

    user_ids = [m["user_id"] for m in members]
    storage_by_user: dict[str, int] = {}
    if user_ids:
        usage_res = db.table("usage_counters").select("user_id, total_storage_bytes").in_("user_id", user_ids).execute()
        storage_by_user = {r["user_id"]: r.get("total_storage_bytes", 0) for r in (usage_res.data or [])}

    # Reuse the ONE existing reader of ENTERPRISE_SEAT_STORAGE_BYTES (spec
    # rule 12) rather than re-parsing the env var here — lazy import mirrors
    # this module's existing subscriptions.service call sites' avoidance of a
    # module-level cross-package import.
    from subscriptions.service import _enterprise_seat_storage_bytes

    storage_cap = _enterprise_seat_storage_bytes()

    seats = []
    for m in members:
        spent = spent_by_member.get(m["id"], 0)
        if m.get("status") == "removed" and spent == 0:
            continue

        email = m.get("email")
        if not email:
            email = _resolve_user_email(db, m["user_id"])
            if email:
                try:
                    db.table("org_members").update({"email": email}).eq("id", m["id"]).execute()
                except Exception as exc:
                    # Non-raising by design (see docstring): a healing write
                    # failure must never break the usage read.
                    print(f"Failed to heal org_members.email for id={m['id']}: {exc}")

        effective_cap = m.get("monthly_cap")
        if effective_cap is None:
            effective_cap = default_cap
        seats.append(
            {
                "orgMemberId": m["id"],
                "userId": m["user_id"],
                "email": email,
                "role": m.get("role"),
                "status": m.get("status"),
                "monthlyCap": m.get("monthly_cap"),
                "effectiveCap": effective_cap,
                "capUsed": m.get("cap_used") or 0,
                "spentThisPeriod": spent,
                "storageBytes": storage_by_user.get(m["user_id"], 0),
                "storageCapBytes": storage_cap,
            }
        )

    return {
        "poolBalance": pool_balance,
        "cumulativePaidIn": cumulative_paid_in,
        "monthlyDispersalCredits": org_row.get("monthly_dispersal_credits") or 0,
        "defaultMemberCap": default_cap,
        "periodStart": pool_wallet.get("period_start") if pool_wallet else None,
        "periodEnd": pool_wallet.get("period_end") if pool_wallet else None,
        "seats": seats,
    }


# ============================================================================
# Invite flow (mirrors teams/service.py's invite_member/accept_invite/
# decline_invite; deltas noted inline where org semantics diverge)
# ============================================================================


async def invite_member(db: Client, user_id: str, org_id: str, email: str, role: str) -> dict:
    """Idempotent invite (teams-style). Admin only.

    - already an ACTIVE member       -> DuplicateInviteError
    - existing pending invite row    -> UPDATE it (role/expiry/status=pending), resend
    - otherwise                      -> INSERT a new pending invite

    Re-inviting a SUSPENDED or REMOVED member is explicitly ALLOWED here:
    is_org_member only counts ACTIVE seats, so this dedupe check does not
    block it — accept_invite (below) reactivates the existing org_members
    row rather than inserting, which IS the designed re-invite path for a
    soft-removed seat (rule 13).
    """
    authz.require_admin(db, user_id, org_id)
    if role not in ("admin", "member"):
        raise ValueError("Invalid role")
    email_l = email.lower()

    existing_user_id = _find_user_id_by_email(db, email)
    if existing_user_id and authz.is_org_member(db, existing_user_id, org_id):
        raise DuplicateInviteError("User is already a member of this organization")

    existing = (
        db.table("pending_org_invites").select("*").eq("org_id", org_id).eq("email", email_l).maybe_single().execute()
    )
    if existing and existing.data:
        updated = (
            db.table("pending_org_invites")
            .update(
                {
                    "role": role,
                    "status": "pending",
                    "expires_at": (datetime.now(UTC) + timedelta(days=7)).isoformat(),
                    "invited_by": user_id,
                }
            )
            .eq("id", existing.data["id"])
            .execute()
        )
        invite = updated.data[0] if updated.data else None
    else:
        try:
            created = (
                db.table("pending_org_invites")
                .insert({"org_id": org_id, "email": email_l, "role": role, "invited_by": user_id})
                .execute()
            )
        except Exception as exc:
            # Race / pre-existing row on the (org_id, LOWER(email)) unique index -> clean 409
            # instead of a raw 500 (mirrors teams.service.invite_member's 23505 handling).
            if "23505" in str(exc) or "duplicate key" in str(exc).lower():
                raise DuplicateInviteError("An invite for this email already exists on this organization") from exc
            raise
        invite = created.data[0] if created.data else None

    return {"type": "invited", "invite": invite, "notify_user_id": existing_user_id}


async def get_pending_invites(db: Client, user_id: str, org_id: str) -> list[dict]:
    """List an org's PENDING invites. Admin only.

    Unlike teams.service.get_pending_invites (which returns every invite
    regardless of status), this filters status='pending' explicitly — the
    spec calls this endpoint out as "pending only".
    """
    authz.require_admin(db, user_id, org_id)
    res = (
        db.table("pending_org_invites")
        .select("*")
        .eq("org_id", org_id)
        .eq("status", "pending")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


async def cancel_invite(db: Client, user_id: str, org_id: str, invite_id: str) -> dict:
    """Delete a pending invite. Admin only."""
    authz.require_admin(db, user_id, org_id)
    db.table("pending_org_invites").delete().eq("id", invite_id).eq("org_id", org_id).execute()
    return {"deleted": invite_id}


async def get_invite_by_token(db: Client, token: str) -> dict | None:
    res = db.table("pending_org_invites").select("*").eq("token", token).maybe_single().execute()
    return res.data if res else None


async def accept_invite(db: Client, user_id: str, user_email: str, token: str) -> dict:
    """Accept an org invite by token. The caller's email must match the
    invite (case-insensitive).

    Reactivation (rule 13): if an org_members row ALREADY exists for this
    (org, user) pair — because the member was previously suspended or
    removed — UNIQUE(org_id, user_id) makes a fresh INSERT impossible, and
    reactivating that existing row (not inserting a second one) is the
    designed re-invite path: it restores the same seat-wallet audit anchor
    and clears the suspended/removed state instead of orphaning it. An
    ALREADY-ACTIVE row is left untouched (no silent role change via
    re-invite of an already-active member).

    Also sets the accepter's `billing_context_org_id` to this org (spec §5
    default-context rule) via a plain table update — deliberately not
    validated here: `profiles.billing_context_org_id` is user-writable by
    design, and it's EntitlementsService's resolution (Task 5), not this
    write, that decides whether it confers anything.

    Also persists the invite's (validated) email onto the member row —
    both on a fresh insert and on invite-driven reactivation of a removed/
    suspended row (migration 20260722000001_org_members_email.sql). This is
    the ONLY place the email is captured going forward: it powers
    get_org_usage's per-seat rollup without an auth-admin lookup per row.
    The admin-facing reactivate_member endpoint below is NOT a source for
    this — there is no invite in that flow.
    """
    invite = await get_invite_by_token(db, token)
    if not invite:
        raise ValueError("Invite not found")
    if invite["email"].lower() != user_email.lower():
        raise PermissionError("This invite was sent to a different email")
    if invite["status"] == "accepted":
        return {"type": "already_accepted", "org_id": invite["org_id"]}
    if invite["status"] != "pending":
        raise InviteInvalidError("This invite is no longer valid")
    if datetime.fromisoformat(invite["expires_at"]) < datetime.now(UTC):
        raise InviteInvalidError("This invite has expired")

    existing = (
        db.table("org_members")
        .select("*")
        .eq("org_id", invite["org_id"])
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    existing_row = existing.data if existing else None
    if existing_row and existing_row.get("status") != "active":
        db.table("org_members").update(
            {
                "status": "active",
                "revoked_at": None,
                "role": invite["role"],
                "invited_by": invite["invited_by"],
                "email": invite["email"],
            }
        ).eq("id", existing_row["id"]).execute()
    elif not existing_row:
        db.table("org_members").insert(
            {
                "org_id": invite["org_id"],
                "user_id": user_id,
                "role": invite["role"],
                "status": "active",
                "invited_by": invite["invited_by"],
                "email": invite["email"],
            }
        ).execute()
    # else: existing_row is already active -> leave it untouched.

    db.table("pending_org_invites").update({"status": "accepted"}).eq("id", invite["id"]).execute()
    # Plain write, not an RPC — see docstring re: validation happening at resolution.
    db.table("profiles").update({"billing_context_org_id": invite["org_id"]}).eq("id", user_id).execute()

    return {"type": "accepted", "org_id": invite["org_id"]}


async def decline_invite(db: Client, user_id: str, user_email: str, token: str) -> dict:
    """Decline an invite by token. The caller's email must match the invite."""
    invite = await get_invite_by_token(db, token)
    if not invite:
        raise ValueError("Invite not found")
    if invite["email"].lower() != user_email.lower():
        raise PermissionError("This invite was sent to a different email")
    db.table("pending_org_invites").update({"status": "declined"}).eq("id", invite["id"]).execute()
    return {"type": "declined", "org_id": invite["org_id"]}


# ============================================================================
# Roles & offboarding (spec rule 5 + 13)
# ============================================================================


async def update_member_role(db: Client, user_id: str, org_id: str, member_id: str, role: str) -> dict:
    """Change a member's role. Admin only. The DB last-admin guard
    (org_members_admin_guard_trigger) may reject a demotion away from the
    only active admin -> LastAdminError (409 at the router, friendly copy
    from the guard's own RAISE message)."""
    authz.require_admin(db, user_id, org_id)
    if role not in ("admin", "member"):
        raise ValueError("Invalid role")
    try:
        res = db.table("org_members").update({"role": role}).eq("id", member_id).eq("org_id", org_id).execute()
    except Exception as exc:
        if _is_last_admin_error(exc):
            raise LastAdminError("You are the only admin of this organization — promote another member first") from exc
        raise
    if not res.data:
        raise ValueError("Member not found")
    return res.data[0]


def _revoke_offboarded_member_access(db: Client, org_id: str, member_user_id: str | None) -> None:
    """Licensing Phase C, Task 4 (rule 3 extended to seat offboarding):
    called AFTER `_offboard`'s reclaim step succeeds (whether or not any
    money actually moved — a zero-balance seat still needs its org-granted
    project access revoked, since the member is being suspended/removed
    regardless of wallet state). Best-effort and never raises: a revocation
    failure logs and does NOT undo the offboard — the status transition (and
    any reclaim) has already landed by the time this runs (money-first
    ordering, Phase B rule 5); a retry of the same offboard, or a later
    admin action, can clean up a grant this attempt didn't reach.

    Delegates to `orgs.projects.revoke_org_granted_memberships` (Task 2's
    single implementation of rule 3), imported lazily to avoid a
    module-level import cycle: `orgs.projects` imports `_resolve_user_email`
    from this module at its own top level."""
    if not member_user_id:
        return
    try:
        from orgs.projects import revoke_org_granted_memberships

        revoke_org_granted_memberships(db, org_id, user_id=member_user_id)
    except Exception as exc:
        print(f"_offboard: revoke_org_granted_memberships failed org_id={org_id} user_id={member_user_id}: {exc}")


async def _offboard(db: Client, user_id: str, org_id: str, member_id: str, final_status: str) -> dict:
    """Shared reclaim-then-transition for suspend/remove (spec rule 5 + 13).
    Admin only. NEVER a hard DELETE — `final_status` lands as a SOFT status on
    the surviving org_members row, which is both the audit anchor for everything
    this member spent from the pool and the storage-billing exemption marker for
    an ex-member (rule 13).

    Nothing is reclaimed. A member never held credits — only a monthly ceiling on
    the shared pool — so offboarding is one status transition plus revoking the
    access their membership granted. `revoked_at` is still stamped: it is the
    offboarding audit timestamp, and re-invite clears it.
    """
    if final_status not in ("suspended", "removed"):
        raise ValueError(f"invalid final_status {final_status!r}")
    authz.require_admin(db, user_id, org_id)

    current = db.table("org_members").select("*").eq("id", member_id).eq("org_id", org_id).maybe_single().execute()
    member = current.data if current else None
    if not member:
        raise ValueError("Member not found")

    if member.get("status") == final_status and member.get("revoked_at"):
        row = member
    else:
        try:
            updated = (
                db.table("org_members")
                .update({"status": final_status, "revoked_at": _now_iso()})
                .eq("id", member_id)
                .eq("org_id", org_id)
                .execute()
            )
        except Exception as exc:
            if _is_last_admin_error(exc):
                raise LastAdminError(
                    "You are the only admin of this organization — promote another member first"
                ) from exc
            raise
        if not updated.data:
            raise ValueError("Member not found")
        reread = db.table("org_members").select("*").eq("id", member_id).eq("org_id", org_id).maybe_single().execute()
        row = reread.data if (reread and reread.data) else updated.data[0]

    _revoke_offboarded_member_access(db, org_id, row.get("user_id"))
    return row


async def suspend_member(db: Client, user_id: str, org_id: str, member_id: str) -> dict:
    """Suspend a member: status='suspended'. Admin only (via _offboard's
    authz.require_admin). Their cap stops mattering because the membership no
    longer resolves an org billing context at all."""
    return await _offboard(db, user_id, org_id, member_id, "suspended")


async def remove_member(db: Client, user_id: str, org_id: str, member_id: str) -> dict:
    """Remove a member: status='removed' — SOFT, NEVER a hard DELETE (rule 13).
    Admin only (via _offboard's authz.require_admin)."""
    return await _offboard(db, user_id, org_id, member_id, "removed")


async def reactivate_member(db: Client, user_id: str, org_id: str, member_id: str) -> dict:
    """Reverse a suspend/remove. Admin only. Only valid from 'suspended' or
    'removed' — reactivating an already-active member is a caller error."""
    authz.require_admin(db, user_id, org_id)
    current = db.table("org_members").select("*").eq("id", member_id).eq("org_id", org_id).maybe_single().execute()
    member = current.data if current else None
    if not member:
        raise ValueError("Member not found")
    if member.get("status") not in ("suspended", "removed"):
        raise ValueError("Member is not suspended or removed")
    res = (
        db.table("org_members")
        .update({"status": "active", "revoked_at": None})
        .eq("id", member_id)
        .eq("org_id", org_id)
        .execute()
    )
    return res.data[0] if res.data else {**member, "status": "active", "revoked_at": None}


# ============================================================================
# Caps (the enforcement mechanism) — admins set a ceiling per member and the
# contract dials on the org. No money moves through any of this; debit_credits
# is what actually holds members to their cap, under the pool's lock.
# ============================================================================


def _require_org_member(db: Client, org_id: str, member_id: str) -> None:
    """Bind an admin-supplied member_id to org_id BEFORE it is used as a seat
    wallet owner_id. require_admin only proves the caller runs THIS org — it
    says nothing about the target member, and the seat wallet is keyed on
    member_id alone. Without this bind, an admin of any org (including a free
    self-created one) could pass another org's member_id and move credits
    into/out of that org's seat wallet; the RLS-bypassing service role makes
    this Python check the only authorization. No status filter — reclaim must
    still recover stranded balances from suspended/removed seats. 404s
    identically for a nonexistent member and one in a different org (no
    existence oracle), matching projects._require_active_org_seat_member."""
    res = db.table("org_members").select("id").eq("id", member_id).eq("org_id", org_id).maybe_single().execute()
    if not (res and res.data):
        raise HTTPException(status_code=404, detail="Member not found")


async def set_member_cap(db: Client, user_id: str, org_id: str, member_id: str, cap: int | None) -> dict:
    """Admin-only: set a member's monthly ceiling on the shared pool.

    Nothing moves. A cap is a limit the debit RPC enforces, so raising one costs
    the pool nothing until the member actually spends, and lowering one below
    what they've already used this period simply means they're done until it
    resets — no clawback, no negative balance, nothing to reconcile.

    `cap=None` clears the member's own cap so they fall through to the org's
    `default_member_cap` (and to uncapped if that is NULL too). Caps may sum to
    more than the monthly dispersal on purpose: most members never reach theirs,
    and the pool is the real ceiling.
    """
    authz.require_admin(db, user_id, org_id)
    _require_org_member(db, org_id, member_id)
    if cap is not None and cap < 0:
        raise ValueError("cap must be >= 0")

    res = db.table("org_members").update({"monthly_cap": cap}).eq("id", member_id).eq("org_id", org_id).execute()
    if not res.data:
        raise ValueError("Member not found")
    return res.data[0]


async def set_org_dispersal(db: Client, org_id: str, monthly_dispersal_credits: int) -> dict:
    """PLATFORM-ADMIN ONLY: how many credits the sweep adds to this org's pool
    each period. Called from subscriptions/admin_router.py, which gates on
    Msanii's own admin dependency — deliberately NOT from /orgs/*, and it takes
    no acting user_id precisely so it cannot be wired to an org-admin route by
    accident.

    WHY: the dispersal is the contract, and nothing in the app collects payment
    for it — there is no org subscription object. If an ORG admin could write
    it, then since any signed-in user can create an org and is auto-made its
    admin, they could set their own dispersal to any number and the sweep would
    grant it monthly for free. It would also self-activate the org, because
    `wallets.cumulative_paid_in` counts 'dispersal' toward the activation floor
    (correct once an operator sets it — an operator setting it IS the commercial
    agreement — and a hole the moment the customer can).

    A raise does NOT top the pool up now: the sweep delivers it once per period,
    so a mid-month change takes effect at the next boundary. That keeps ONE path
    writing dispersal credits, which is what makes its monthly idempotency hold.
    """
    if monthly_dispersal_credits < 0:
        raise ValueError("monthly_dispersal_credits must be >= 0")

    res = (
        db.table("organizations")
        .update({"monthly_dispersal_credits": monthly_dispersal_credits})
        .eq("id", org_id)
        .execute()
    )
    return res.data[0] if res.data else {"id": org_id}


# ============================================================================
# Cap-raise requests — member asks, admin approves. This replaces overage for
# org members: there is no pay-as-you-go on a pool, so the escape hatch is a
# higher ceiling. Approving one moves NO money, which is why it needs none of
# the transfer machinery it used to: it writes org_members.monthly_cap.
# ============================================================================


async def submit_credit_request(
    db: Client, user_id: str, org_id: str, requested_cap: int | None, note: str | None
) -> dict:
    """Any ACTIVE member may ask for a higher monthly cap. Member-level authz
    (NOT admin) — authz.require_member 404s for a non-member OR a
    suspended/removed member, since is_org_member only counts status='active'
    rows (same gate get_org uses).

    The DB partial unique index (org_member_id, WHERE status='pending') is
    the actual anti-spam enforcement — a second pending request for the same
    seat raises a 23505 here, caught and re-raised as
    DuplicatePendingRequestError (409 "You already have a pending request.").
    """
    authz.require_member(db, user_id, org_id)

    member_row = (
        db.table("org_members")
        .select("id")
        .eq("org_id", org_id)
        .eq("user_id", user_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    member = member_row.data if member_row else None
    if not member:
        # Defensive: require_member just confirmed an active seat exists —
        # this should be unreachable outside of a race with a concurrent
        # offboard.
        raise ValueError("Member not found")

    payload: dict = {"org_id": org_id, "org_member_id": member["id"]}
    if requested_cap is not None:
        payload["requested_cap"] = requested_cap
    if note is not None:
        payload["note"] = note

    try:
        created = db.table("credit_requests").insert(payload).execute()
    except Exception as exc:
        if "23505" in str(exc) or "duplicate key" in str(exc).lower():
            raise DuplicatePendingRequestError("You already have a pending request.") from exc
        raise

    request = created.data[0] if created.data else None
    return {"request": request, "org_member_id": member["id"]}


async def list_credit_requests(db: Client, user_id: str, org_id: str) -> list[dict]:
    """Admins see every request for the org (newest first); a non-admin
    member sees only their own."""
    authz.require_member(db, user_id, org_id)

    query = db.table("credit_requests").select("*").eq("org_id", org_id)
    if not authz.is_org_admin(db, user_id, org_id):
        member_row = (
            db.table("org_members")
            .select("id")
            .eq("org_id", org_id)
            .eq("user_id", user_id)
            .eq("status", "active")
            .maybe_single()
            .execute()
        )
        member = member_row.data if member_row else None
        member_id = member["id"] if member else None
        query = query.eq("org_member_id", member_id)

    res = query.order("created_at", desc=True).execute()
    return res.data or []


async def approve_credit_request(db: Client, user_id: str, org_id: str, request_id: str, cap: int) -> dict:
    """Admin-only: approve a cap-raise request by writing the member's new cap.

    1. Fetch the request; 404 if unknown, 409 if already resolved.
    2. Write `org_members.monthly_cap = cap` for the requesting member.
    3. Mark the row approved with `resolved_cap`.

    No money moves, so this needs none of the transfer-replay machinery the
    allocation version carried: setting a cap is idempotent by nature. A retry
    after a partial failure writes the same cap and resolves the same row, and
    the worst case of a double-apply is the cap the admin chose, twice.
    """
    authz.require_admin(db, user_id, org_id)

    current = db.table("credit_requests").select("*").eq("id", request_id).eq("org_id", org_id).maybe_single().execute()
    request = current.data if current else None
    if not request:
        raise CreditRequestNotFoundError("Credit request not found")
    if request["status"] != "pending":
        raise CreditRequestAlreadyResolvedError("This request has already been resolved")
    if cap < 0:
        raise ValueError("cap must be >= 0")

    db.table("org_members").update({"monthly_cap": cap}).eq("id", request["org_member_id"]).eq(
        "org_id", org_id
    ).execute()

    updated = (
        db.table("credit_requests")
        .update(
            {
                "status": "approved",
                "resolved_cap": cap,
                "resolved_by": user_id,
                "resolved_at": _now_iso(),
            }
        )
        .eq("id", request_id)
        .execute()
    )
    if updated.data:
        return updated.data[0]
    return {**request, "status": "approved", "resolved_cap": cap, "resolved_by": user_id}


async def deny_credit_request(db: Client, user_id: str, org_id: str, request_id: str, note: str | None) -> dict:
    """Admin-only. Status-only transition — no money moves, no RPC call."""
    authz.require_admin(db, user_id, org_id)

    current = db.table("credit_requests").select("*").eq("id", request_id).eq("org_id", org_id).maybe_single().execute()
    request = current.data if current else None
    if not request:
        raise CreditRequestNotFoundError("Credit request not found")
    if request["status"] != "pending":
        raise CreditRequestAlreadyResolvedError("This request has already been resolved")

    fields = {"status": "denied", "resolved_by": user_id, "resolved_at": _now_iso()}
    if note is not None:
        fields["note"] = note

    updated = db.table("credit_requests").update(fields).eq("id", request_id).execute()
    if updated.data:
        return updated.data[0]
    return {**request, **fields}
