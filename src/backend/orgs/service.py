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


class SeatBalanceNotZeroError(Exception):
    """Archive blocked — a seat wallet of this org still holds a nonzero balance."""


class DuplicateInviteError(Exception):
    """Already an active member, or a duplicate pending invite for (org, email)."""


class LastAdminError(Exception):
    """The DB last-admin guard (org_members_admin_guard_trigger) rejected a
    role change / suspend / remove that would leave the org with no active
    admin."""


class InviteInvalidError(Exception):
    """Invite is expired, declined, or otherwise no longer actionable."""


class ReclaimFailedError(Exception):
    """The seat -> pool transfer_credits RPC raised during offboarding.

    The org_members status/revoked_at transition (rule 5 — money-first) has
    ALREADY landed by the time this is raised, and is deliberately NOT rolled
    back: a retry of the same offboard recomputes the identical request_id
    from the stored revoked_at, so transfer_credits' own idempotency either
    no-ops (the first attempt actually succeeded downstream of where we
    observed the error) or raises again (genuinely still short) — either way
    safe to retry.
    """


class PoolBalanceInsufficientError(Exception):
    """transfer_credits reported the org pool wallet doesn't have enough
    reserve balance to cover a requested allocation (spec rule 2 — transfers
    never overdraw). Mapped by the router to 409 "The pool doesn't have
    enough credits."."""


class SeatBalanceChangedError(Exception):
    """transfer_credits reported the seat wallet's balance changed between
    being read and the reclaim transfer landing (a concurrent debit/reclaim
    raced this one) — a stale-balance race, not a caller error. Mapped by
    the router to 409 "Balance changed — refresh and retry."."""


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

    cumulative_purchased = wallets.cumulative_purchased(db, wallet["id"]) if wallet else 0

    effective_min = org.get("min_initial_purchase_credits") or _default_min_initial_purchase_credits()
    remaining_to_activate = max(0, effective_min - cumulative_purchased)

    member_count_res = (
        db.table("org_members").select("id", count="exact").eq("org_id", org_id).eq("status", "active").execute()
    )
    member_count = member_count_res.count or 0

    return {
        **org,
        "my_role": my_role,
        "pool_balance": pool_balance,
        "cumulative_purchased": cumulative_purchased,
        "remaining_to_activate": remaining_to_activate,
        "member_count": member_count,
    }


async def update_org(db: Client, user_id: str, org_id: str, fields: dict) -> dict:
    """Update org name/default_seat_allowance. Admin only.

    `fields` is expected to come from `OrgUpdate.model_dump(exclude_unset=True)`
    at the router — keys explicitly present in the request (including an
    explicit `None` for default_seat_allowance, clearing it to manual-only)
    are written; omitted keys are left untouched."""
    authz.require_admin(db, user_id, org_id)
    if not fields:
        res = db.table("organizations").select("*").eq("id", org_id).maybe_single().execute()
        return res.data
    res = db.table("organizations").update(fields).eq("id", org_id).execute()
    return res.data[0] if res.data else None


def _teardown_archived_org_grants(db: Client, org_id: str) -> None:
    """Licensing Phase C, Task 4 (rule 12): `archived_at` is an UPDATE, so
    the `org_project_links.org_id` ON DELETE CASCADE never fires — without
    this, a project stays linked to a tombstone org and rule 8's
    `UNIQUE(project_id)` blocks re-linking to a live org. Runs AFTER
    `archived_at` has already landed (the load-bearing write for
    `archive_org`) and is best-effort/never-raising, mirroring `_offboard`'s
    money-first-then-cleanup posture: a teardown failure here must not undo
    or block an already-committed archive.

    Two independent cleanups, each swallowing its own failure so one doesn't
    block the other:
      1. `revoke_org_granted_memberships` (Task 2's single implementation of
         rule 3, imported lazily here to avoid a module-level import cycle —
         `orgs.projects` imports `_resolve_user_email` from this module at
         its own top level) — org-scoped only (no user_id/project_id
         narrowing), i.e. every `project_members` row THIS org ever granted,
         on every project it touched. Organic rows are untouched by
         construction (the helper's `org_id` filter is the entire mechanism).
      2. Deletes this org's `org_project_links` row(s) directly — there is
         no separate "revoke a link" helper; unlike `project_members` grants,
         a link has no organic/other-org ambiguity to protect (rule 8: at
         most one link per project, and it's unconditionally this org's).
    """
    try:
        from orgs.projects import revoke_org_granted_memberships

        revoke_org_granted_memberships(db, org_id)
    except Exception as exc:
        print(f"archive_org: revoke_org_granted_memberships failed org_id={org_id}: {exc}")
    try:
        db.table("org_project_links").delete().eq("org_id", org_id).execute()
    except Exception as exc:
        print(f"archive_org: deleting org_project_links failed org_id={org_id}: {exc}")


async def archive_org(db: Client, user_id: str, org_id: str) -> dict:
    """Archive an org. Admin only. 409s unless EVERY seat wallet belonging to
    this org's members (any status — active/suspended/removed; offboarding
    should already have zeroed removed/suspended seats, and archiving must
    re-verify rather than trust that) has balance 0. Sets archived_at, then
    tears down every access grant and link this org ever created (Task 4,
    rule 12) — see `_teardown_archived_org_grants`."""
    authz.require_admin(db, user_id, org_id)

    members = db.table("org_members").select("id").eq("org_id", org_id).execute()
    member_ids = [m["id"] for m in (members.data or [])]
    if member_ids:
        wallets = (
            db.table("credit_wallets")
            .select("id, owner_id, bundle_balance, reserve_balance")
            .eq("owner_type", "seat")
            .in_("owner_id", member_ids)
            .execute()
        )
        for w in wallets.data or []:
            if (w.get("bundle_balance", 0) + w.get("reserve_balance", 0)) != 0:
                raise SeatBalanceNotZeroError(
                    "Reclaim all seat credits first — every seat must be at zero before archiving."
                )

    res = db.table("organizations").update({"archived_at": _now_iso()}).eq("id", org_id).execute()
    _teardown_archived_org_grants(db, org_id)
    return res.data[0] if res.data else {"archived": org_id}


async def get_org_usage(db: Client, user_id: str, org_id: str) -> dict:
    """Admin-only per-seat usage rollup for the org admin console (Task 7,
    round-5 requirement: the finite ENTERPRISE_SEAT_STORAGE_BYTES ceiling must
    be visible in the console BEFORE an upload fails, not just at the wall).

    Round-trip shape (deliberately ONE query per table): `organizations.id`
    and `org_members.id` values never collide, so a single `credit_wallets`
    `.in_("owner_id", [org_id, *member_ids])` read returns BOTH the org's pool
    wallet (owner_type='org') AND every member's seat wallet (owner_type=
    'seat') in one round trip; the wallet ids that query yields then drive one
    follow-up `credit_ledger` read the same way (pool 'purchase' rows feed
    cumulativePurchased, seat 'debit' rows feed each seat's spentAllTime).

    Member visibility (round 5): every non-removed member is included. A
    REMOVED member is included ONLY if their seat wallet still holds a
    nonzero balance — offboarding reclaims first (spec rule 5), so this
    should be rare, but an admin needs to see stranded money on an ex-seat's
    wallet rather than have it silently disappear from the console.

    Email resolution (licensing follow-ups Task 4): `org_members.email` is
    captured going forward at invite-accept (see accept_invite), so the
    common case reads it straight off the row with zero auth-admin calls.
    A NULL email (a pre-migration row, or a creator row — the
    auto_create_org_admin trigger never goes through accept_invite) falls
    back to the existing `_resolve_user_email` auth lookup, ONCE, and the
    resolved value is written back onto the row so every subsequent read of
    that same row is free. The write-back is best-effort and deliberately
    non-raising — a failed UPDATE must never take down the usage read; the
    row just gets re-resolved next time.
    """
    authz.require_admin(db, user_id, org_id)

    members_res = db.table("org_members").select("id, user_id, role, status, email").eq("org_id", org_id).execute()
    members = members_res.data or []
    member_ids = [m["id"] for m in members]

    wallets_res = (
        db.table("credit_wallets")
        .select("id, owner_type, owner_id, bundle_balance, reserve_balance")
        .in_("owner_id", [org_id, *member_ids])
        .execute()
    )
    wallet_rows = wallets_res.data or []
    pool_wallet = next((w for w in wallet_rows if w.get("owner_type") == "org"), None)
    seat_wallet_by_member_id = {w["owner_id"]: w for w in wallet_rows if w.get("owner_type") == "seat"}
    pool_balance = (pool_wallet.get("bundle_balance", 0) + pool_wallet.get("reserve_balance", 0)) if pool_wallet else 0

    wallet_ids = [w["id"] for w in wallet_rows]
    ledger_rows: list[dict] = []
    if wallet_ids:
        ledger_res = db.table("credit_ledger").select("wallet_id, delta, kind").in_("wallet_id", wallet_ids).execute()
        ledger_rows = ledger_res.data or []

    cumulative_purchased = sum(
        r.get("delta", 0)
        for r in ledger_rows
        if pool_wallet and r.get("wallet_id") == pool_wallet["id"] and r.get("kind") == "purchase"
    )

    # spentAllTime is sum(|delta|) over kind='debit' rows only — seat wallets
    # have no overage path (rule 8: no personal fallback, no overage for
    # seats), so 'overage_debit' rows never exist here; unlike the personal
    # usage view, there is no second kind to fold in.
    spent_by_wallet_id: dict[str, int] = {}
    for r in ledger_rows:
        if r.get("kind") != "debit":
            continue
        wid = r.get("wallet_id")
        spent_by_wallet_id[wid] = spent_by_wallet_id.get(wid, 0) + abs(r.get("delta", 0))

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
        seat_wallet = seat_wallet_by_member_id.get(m["id"])
        seat_balance = (
            (seat_wallet.get("bundle_balance", 0) + seat_wallet.get("reserve_balance", 0)) if seat_wallet else 0
        )
        if m.get("status") == "removed" and seat_balance == 0:
            continue
        spent_all_time = spent_by_wallet_id.get(seat_wallet["id"], 0) if seat_wallet else 0

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

        seats.append(
            {
                "orgMemberId": m["id"],
                "userId": m["user_id"],
                "email": email,
                "role": m.get("role"),
                "status": m.get("status"),
                "seatBalance": seat_balance,
                "spentAllTime": spent_all_time,
                "storageBytes": storage_by_user.get(m["user_id"], 0),
                "storageCapBytes": storage_cap,
            }
        )

    return {
        "poolBalance": pool_balance,
        "cumulativePurchased": cumulative_purchased,
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
    Admin only. NEVER a hard DELETE — `final_status` lands as a SOFT status
    on the surviving org_members row, which is both the seat wallet's audit
    anchor and the storage-billing exemption marker for an ex-seat member
    (rule 13).

    Money-first ordering:
      1. Transition org_members to `final_status`, stamping `revoked_at` —
         UNLESS the row is ALREADY at `final_status` with a `revoked_at` set
         already (a retry of a prior attempt whose reclaim failed AFTER the
         status write landed): reuse that row as-is rather than re-stamping
         revoked_at, which would mint a fresh request_id and could reclaim
         the same money twice. Otherwise, write revoked_at=now() and REREAD
         the row — revoked_at is the SOURCE OF TRUTH the reclaim key derives
         from, not the locally-computed value, so it's read back from
         storage rather than trusted from the UPDATE call's echo.
      2. Read the seat wallet (owner_type='seat', owner_id=member_id). No
         wallet, or balance (bundle + reserve) <= 0 -> done, no RPC call at
         all (money RPCs raise on non-positive amounts, and there's nothing
         to reclaim).
      3. Nonzero balance -> resolve the org POOL wallet via Task 4's
         `wallets.read_or_create_org_wallet` (create-on-miss). Earlier
         (Task 3) this read the existing row only and raised RuntimeError on
         a miss, reasoning that a funded seat implies a completed pool
         purchase and therefore an existing pool wallet. Task 4 removes that
         stopgap: create-on-miss is now the single load-bearing pool-wallet
         accessor shared with allocate/reclaim, so offboarding gets the same
         forgiving behavior instead of a distinct failure mode for what is,
         in practice, an unreachable edge (a pool wallet with zero purchases
         can't have funded a seat) — and a genuine anomaly here now
         self-heals instead of 500ing.
      4. `transfer_credits(seat -> pool, 'reclaim',
         f"offboard:{member_id}:{epoch(stored revoked_at)}")`. A raise here
         is surfaced as ReclaimFailedError; the status transition from step
         1 is deliberately NOT rolled back (money-first — a retry re-derives
         the identical request_id and converges).
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

    wallet_res = (
        db.table("credit_wallets")
        .select("id, bundle_balance, reserve_balance")
        .eq("owner_type", "seat")
        .eq("owner_id", member_id)
        .execute()
    )
    wallet_rows = wallet_res.data or []
    seat_wallet = wallet_rows[0] if wallet_rows else None
    balance = (seat_wallet.get("bundle_balance", 0) + seat_wallet.get("reserve_balance", 0)) if seat_wallet else 0
    if not seat_wallet or balance <= 0:
        _revoke_offboarded_member_access(db, org_id, row.get("user_id"))
        return row

    pool_wallet = wallets.read_or_create_org_wallet(db, org_id)
    pool_wallet_id = pool_wallet["id"]

    request_id = f"offboard:{member_id}:{_epoch(row['revoked_at'])}"
    try:
        db.rpc(
            "transfer_credits",
            {
                "p_from_wallet": seat_wallet["id"],
                "p_to_wallet": pool_wallet_id,
                "p_amount": balance,
                "p_kind": "reclaim",
                "p_request_id": request_id,
                "p_metadata": {"org_id": org_id, "reason": final_status},
            },
        ).execute()
    except Exception as exc:
        raise ReclaimFailedError(f"Failed to reclaim seat credits for member {member_id}") from exc

    _revoke_offboarded_member_access(db, org_id, row.get("user_id"))
    return row


async def suspend_member(db: Client, user_id: str, org_id: str, member_id: str) -> dict:
    """Suspend a seat: reclaim-then-transition to status='suspended'. Admin
    only (via _offboard's authz.require_admin)."""
    return await _offboard(db, user_id, org_id, member_id, "suspended")


async def remove_member(db: Client, user_id: str, org_id: str, member_id: str) -> dict:
    """Remove a seat: reclaim-then-transition to status='removed' — SOFT,
    NEVER a hard DELETE (rule 13). Admin only (via _offboard's
    authz.require_admin)."""
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
# Allocate / reclaim (spec rule 2, Task 4) — admin-initiated pool<->seat money
# movement via the transfer_credits RPC. Both wallets are resolved through
# Task 4's create-on-miss helpers (orgs.wallets): an admin may allocate to a
# brand-new seat, or reclaim from one, before either wallet has been lazily
# created by any other path.
# ============================================================================


def _is_insufficient_balance_error(exc: Exception) -> bool:
    """transfer_credits' own message is 'insufficient balance on source
    wallet (have %, need %)' — same substring-detection idiom as
    _is_last_admin_error above; the RPC doesn't raise a typed exception the
    Python client can catch structurally."""
    return "insufficient balance" in str(exc).lower()


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


async def allocate_credits(
    db: Client, user_id: str, org_id: str, member_id: str, amount: int, idempotency_key: str
) -> dict:
    """Admin-only pool -> seat allocation. `transfer_credits` never overdraws
    the pool (spec rule 2): an underfunded pool raises, caught here and
    re-raised as PoolBalanceInsufficientError (409 at the router). The
    request_id passed to the RPC is the BASE `alloc:{idempotency_key}` key —
    transfer_credits appends its own :from/:to suffixes; never suffix here.
    """
    authz.require_admin(db, user_id, org_id)
    _require_org_member(db, org_id, member_id)
    seat_wallet = wallets.read_or_create_seat_wallet(db, member_id)
    pool_wallet = wallets.read_or_create_org_wallet(db, org_id)

    try:
        res = db.rpc(
            "transfer_credits",
            {
                "p_from_wallet": pool_wallet["id"],
                "p_to_wallet": seat_wallet["id"],
                "p_amount": amount,
                "p_kind": "allocation",
                "p_request_id": f"alloc:{idempotency_key}",
                "p_metadata": {"org_id": org_id, "member_id": member_id, "actor": user_id},
            },
        ).execute()
    except Exception as exc:
        if _is_insufficient_balance_error(exc):
            raise PoolBalanceInsufficientError("The pool doesn't have enough credits.") from exc
        raise
    return res.data


async def reclaim_credits(
    db: Client, user_id: str, org_id: str, member_id: str, amount: int | None, idempotency_key: str
) -> dict:
    """Admin-only seat -> pool reclaim.

    `amount=None` means reclaim-all: the seat wallet's current balance
    (bundle + reserve) is read from the same create-on-miss lookup used to
    resolve its wallet id. A balance <= 0 is a genuine no-op, returned as
    `{"removed": 0}` WITHOUT calling transfer_credits at all (round 5 guard:
    accepted debit drift can push a seat's bundle negative, and the RPC
    raises on a non-positive amount — surfacing that as an unmapped 500
    instead of a quiet no-op would be a regression, not a fix).

    A stale-balance race (the seat's balance changed between this read and
    the transfer landing — e.g. a concurrent allowance top-up or another
    reclaim) surfaces from the RPC as the same "insufficient balance on
    source" message allocate_credits maps differently — here it becomes
    SeatBalanceChangedError, "Balance changed — refresh and retry."."""
    authz.require_admin(db, user_id, org_id)
    _require_org_member(db, org_id, member_id)
    seat_wallet = wallets.read_or_create_seat_wallet(db, member_id)
    pool_wallet = wallets.read_or_create_org_wallet(db, org_id)

    if amount is None:
        balance = seat_wallet.get("bundle_balance", 0) + seat_wallet.get("reserve_balance", 0)
        if balance <= 0:
            return {"removed": 0}
        transfer_amount = balance
    else:
        transfer_amount = amount

    try:
        res = db.rpc(
            "transfer_credits",
            {
                "p_from_wallet": seat_wallet["id"],
                "p_to_wallet": pool_wallet["id"],
                "p_amount": transfer_amount,
                "p_kind": "reclaim",
                "p_request_id": f"reclaim:{idempotency_key}",
                "p_metadata": {"org_id": org_id, "member_id": member_id, "actor": user_id},
            },
        ).execute()
    except Exception as exc:
        if _is_insufficient_balance_error(exc):
            raise SeatBalanceChangedError("Balance changed — refresh and retry.") from exc
        raise
    return res.data


# ============================================================================
# Credit requests (Task 9) — member ask -> admin approve, replacing overage
# for seats. Money movement mirrors allocate_credits (pool -> seat via
# transfer_credits, both wallets resolved through Task 4's create-on-miss
# helpers) with one addition: the round-4 duplicate-replay read-back (see
# approve_credit_request's docstring).
# ============================================================================


async def submit_credit_request(
    db: Client, user_id: str, org_id: str, requested_credits: int | None, note: str | None
) -> dict:
    """Any ACTIVE member may ask for more credits. Member-level authz (NOT
    admin) — authz.require_member 404s for a non-member OR a
    suspended/removed seat, since is_org_member only counts status='active'
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
    if requested_credits is not None:
        payload["requested_credits"] = requested_credits
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


def _read_back_resolved_credits(db: Client, base_request_id: str) -> int:
    """Round-4 duplicate-replay read-back (plan Task 9, rule 10): when
    transfer_credits reports `{"duplicate": true}`, the amount that ACTUALLY
    moved must be read from the ledger's `:from` leg — never trusted from the
    retry's requested `credits` body, which may legitimately differ from what
    landed on the transfer that succeeded before a later step (the
    credit_requests status UPDATE) failed.

    Example the plan spells out: approve-100 -> transfer_credits lands ->
    the status UPDATE then raises (500) -> the client retries approve with
    credits=100 (or any other value) -> transfer_credits sees the `:from` key
    already exists and returns duplicate=True -> this function reads the
    ORIGINAL -100 delta off that ledger row and returns 100, regardless of
    what the retry asked for.
    """
    res = db.table("credit_ledger").select("delta").eq("request_id", f"{base_request_id}:from").maybe_single().execute()
    row = res.data if res else None
    if not row:
        raise RuntimeError(
            f"transfer_credits reported a duplicate for {base_request_id!r} but no "
            f"ledger row exists at '{base_request_id}:from' to read the amount back from"
        )
    return abs(row["delta"])


async def approve_credit_request(db: Client, user_id: str, org_id: str, request_id: str, credits: int) -> dict:
    """Admin-only. TRANSFER FIRST, then mark resolved (spec rule 10):

    1. Fetch the request; 404 if unknown, 409 if already resolved.
    2. transfer_credits(pool -> seat, 'allocation', request_id=f"credreq:{request_id}")
       for `credits` (the admin-chosen amount — may differ from what the
       member asked for). An underfunded pool raises here, mapped to
       PoolBalanceInsufficientError (409) by the caller of the RPC — and
       because this happens BEFORE any write to credit_requests, the
       request row is left untouched at status='pending' (rule 10: a
       pool-insufficient error must not resolve the request).
    3. Resolve `resolved_credits`: a FRESH transfer (duplicate=False) used
       exactly `credits`; a duplicate replay reads the actually-moved
       amount back from the ledger (_read_back_resolved_credits) instead of
       trusting this call's `credits` argument, which is the retry's own
       input and may not match what landed the first time.
    4. Mark the row status='approved', resolved_credits, resolved_by,
       resolved_at.
    """
    authz.require_admin(db, user_id, org_id)

    current = db.table("credit_requests").select("*").eq("id", request_id).eq("org_id", org_id).maybe_single().execute()
    request = current.data if current else None
    if not request:
        raise CreditRequestNotFoundError("Credit request not found")
    if request["status"] != "pending":
        raise CreditRequestAlreadyResolvedError("This request has already been resolved")

    seat_wallet = wallets.read_or_create_seat_wallet(db, request["org_member_id"])
    pool_wallet = wallets.read_or_create_org_wallet(db, org_id)

    rpc_request_id = f"credreq:{request_id}"
    try:
        result = db.rpc(
            "transfer_credits",
            {
                "p_from_wallet": pool_wallet["id"],
                "p_to_wallet": seat_wallet["id"],
                "p_amount": credits,
                "p_kind": "allocation",
                "p_request_id": rpc_request_id,
                "p_metadata": {"org_id": org_id, "credit_request_id": request_id, "actor": user_id},
            },
        ).execute()
    except Exception as exc:
        if _is_insufficient_balance_error(exc):
            raise PoolBalanceInsufficientError("The pool doesn't have enough credits.") from exc
        raise

    transfer_data = result.data or {}
    if transfer_data.get("duplicate"):
        resolved_credits = _read_back_resolved_credits(db, rpc_request_id)
    else:
        resolved_credits = credits

    updated = (
        db.table("credit_requests")
        .update(
            {
                "status": "approved",
                "resolved_credits": resolved_credits,
                "resolved_by": user_id,
                "resolved_at": _now_iso(),
            }
        )
        .eq("id", request_id)
        .execute()
    )
    if updated.data:
        return updated.data[0]
    return {**request, "status": "approved", "resolved_credits": resolved_credits, "resolved_by": user_id}


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
