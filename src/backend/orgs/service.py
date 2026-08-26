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

import logging
import math
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import HTTPException
from supabase import Client

import artist_access
from analytics import capture as analytics_capture
from orgs import authz, wallets

logger = logging.getLogger(__name__)


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


SEATS_PER_PRO = 5
"""Pro seat-unlock block size (owner decision 2026-08-16, superseding the
flat 5-member Pro cap from the previous same-day decision):
`effective_limit = min(dials.max_team_members, SEATS_PER_PRO * (1 +
pro_member_count))`, where `pro_member_count` is the org's ACTIVE members
(excluding the coverer) whose resolved tier is Pro. The covering owner
counts as the first Pro seat for free, so a lone Pro coverer unlocks 5; one
Pro member joining unlocks 10 (== tier_entitlements.pro.max_team_members,
the hard ceiling — Basic's 3-member cap is always below SEATS_PER_PRO so it
never engages this formula). This gates ADDING a member only, never
holding one: if the Pro member who unlocked a seat later leaves or
downgrades, existing members are NOT removed — only new invites/accepts
are refused until the team is back under the (re-shrunk) ceiling. No
per-org overrides, no paid seat add-ons past the tier's max."""


class TeamFullError(Exception):
    """Self-serve org already holds `effective_limit` ACTIVE seats (excluding
    the covering owner) — invite/accept refused until a seat frees up, the
    ceiling grows (another Pro member joins), or the coverer upgrades
    (Task 7, spec §3; SEATS_PER_PRO formula owner decision 2026-08-16).
    Mapped by the router to 402, with the limit baked into the message.

    `limit` carries the same number separately, for callers (accept_invite)
    that need to build their own message rather than surface this one
    verbatim.

    `next_step` is "contact" when the coverer is on Pro and already at
    `max_team_members` (no per-org overrides / paid seat add-ons — bigger
    teams are an Enterprise conversation), else "upgrade" (a Pro coverer
    with unlockable seats left also gets "contact", pointed at Enterprise as
    an alternative to waiting on another Pro member). The router forwards it
    verbatim in the 402 detail for the frontend CTA."""

    def __init__(self, message: str, limit: int | None = None, next_step: str | None = None):
        super().__init__(message)
        self.limit = limit
        self.next_step = next_step


class TeamLapsedError(Exception):
    """Self-serve org is status='lapsed' — no new members while paused; the
    covering admin must reactivate first (Task 7, spec §3). Mapped by the
    router to 409."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_last_admin_error(exc: Exception) -> bool:
    return "only admin" in str(exc).lower()


def _is_insufficient_reserve_error(exc: Exception) -> bool:
    """True for the `transfer_credits` RPC's RAISE on an under-funded personal
    reserve (20260816000002: 'insufficient reserve: have %, need %'). Mapped
    to 409 by transfer_credits_to_pool — a silent partial transfer would be
    worse than a rejected one, so the RPC raises rather than clamps."""
    return "insufficient reserve" in str(exc).lower()


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


def _member_email(db: Client, member: dict) -> str | None:
    """Email for an `org_members` row, healing a NULL onto the row once.

    `org_members.email` is captured going forward at invite-accept, so the
    common case reads it straight off the row with zero auth-admin calls. A NULL
    (a pre-migration row, or a creator row — the auto_create_org_admin trigger
    never goes through accept_invite) falls back to `_resolve_user_email` ONCE
    and is written back, best-effort and deliberately non-raising: a failed heal
    must never break the read that triggered it. Shared by `get_org` (member-
    visible admin contacts) and `get_org_usage` (the admin seat rollup).
    """
    email = member.get("email")
    if email:
        return email
    email = _resolve_user_email(db, member["user_id"])
    if email:
        try:
            db.table("org_members").update({"email": email}).eq("id", member["id"]).execute()
        except Exception as exc:
            print(f"Failed to heal org_members.email for id={member['id']}: {exc}")
    return email


def _org_name(db: Client, org_id: str, default: str) -> str:
    """The org's display name for user-facing copy, or `default` if it can't be
    read. Callers differ on the fallback wording ("your organization" when
    addressing a member, "an organization" when describing one)."""
    res = db.table("organizations").select("name").eq("id", org_id).maybe_single().execute()
    return ((res.data or {}) if res else {}).get("name") or default


# Stored value meaning "no monthly ceiling at all", as opposed to NULL, which
# means "inherit the level above". Needed because every org now carries a
# default_member_cap (2,000 — 20260814000001), so clearing a member's own cap
# inherits that default instead of removing the limit. Same "-1 is unlimited"
# idiom tier_entitlements uses for its caps. Normalized to None on every read
# (EntitlementsService._member_cap, effective_member_cap below) and inside
# debit_credits, so it never escapes the storage layer.
UNLIMITED_CAP = -1

# How long an org invite stays acceptable. Mirrors the column default set in
# 20260814000002 — this constant is what a RE-invite stamps, the column default
# is what a first invite gets, and they must agree.
INVITE_TTL = timedelta(hours=48)

# Shared 402 detail for both reactivation guards (claim_coverage, unarchive_org)
# that require reactivation_allowed(): a lapsed/archived team may not wake up
# over-pool. Duplicated literally before this constant existed.
REACTIVATION_BLOCKED_DETAIL = (
    "Your teams hold more storage than your plan includes — free up space or upgrade before reactivating."
)


def effective_member_cap(monthly_cap: int | None, default_cap: int | None) -> int | None:
    """Resolve the cap chain to a display value: None = no ceiling.

    The Python mirror of debit_credits' CASE — keep the two in step.
    """
    cap = monthly_cap if monthly_cap is not None else default_cap
    return None if cap is not None and cap < 0 else cap


def _default_min_initial_purchase_credits() -> int:
    """Platform default activation floor when an org's own
    min_initial_purchase_credits is NULL (spec §4)."""
    return int(os.getenv("ENTERPRISE_MIN_INITIAL_CREDITS", "10000"))


async def create_org(db: Client, user_id: str, name: str) -> dict:
    """Create an org. Flag matrix (spec §2 rev 2, review finding 5):
      - LICENSING_ENABLED + CREDITS_ENABLED both on -> SELF-SERVE: kind=
        'self_serve', slot-gated (NoSlotError -> 402 at the router), covered
        by its creator, born ACTIVE (no activation floor — the slot IS the
        activation).
      - LICENSING_ENABLED on, CREDITS_ENABLED off -> 503. This half-flag
        window must NOT fall through to the legacy branch below: that would
        silently mint a permanent kind='enterprise' row nobody governs (no
        dispersal, no activation flow, no admin visibility) every time a
        user hits this endpoint. Refuse outright instead.
      - LICENSING_ENABLED off (CREDITS_ENABLED irrelevant) -> legacy insert,
        byte-identical to pre-licensing (status defaults to pending in the
        DB; kind defaults to enterprise).
    Enterprise orgs are otherwise created ONLY via the Msanii admin
    endpoints; the open-creation default-'enterprise' path was a standing
    bypass. The auto_create_org_admin trigger still adds the creator's admin
    row in every branch — this function must NOT insert org_members (see
    module docstring)."""
    from orgs import standing
    from subscriptions.service import credits_enabled, licensing_enabled

    if licensing_enabled() and not credits_enabled():
        raise HTTPException(status_code=503, detail="Team creation is temporarily unavailable — please try again soon.")

    payload = {"name": name, "created_by": user_id}
    if standing.self_serve_enabled():
        standing.require_free_slot(db, user_id)  # raises NoSlotError -> 402
        payload.update(
            kind="self_serve",
            status="active",
            covered_by=user_id,
            covered_at=_now_iso(),
        )
    res = db.table("organizations").insert(payload).execute()
    org = res.data[0] if res.data else None
    if not org:
        raise RuntimeError("Failed to create organization")
    # The creator is always the admin (auto_create_org_admin), so annotate
    # my_role like list_my_orgs — no follow-up GET needed to learn the role.
    return {**org, "my_role": "admin"}


# Fields on an org payload that only an ACTIVE ADMIN of that org may see. All
# of them are commercial facts about the ORG — what it negotiated, what it has
# left, what it still owes to activate, how much storage it is burning — rather
# than facts about the reader's own access. A plain member's own ceiling
# reaches them through the credits block (`memberCap`), never from here.
#
# `default_member_cap` is deliberately NOT on this list: it IS the member's own
# effective cap whenever they have no personal override.
_ADMIN_ONLY_ORG_FIELDS = frozenset(
    {
        "monthly_dispersal_credits",
        "min_initial_purchase_credits",
        "storage_bytes",
        "pool_balance",
        "cumulative_paid_in",
        "remaining_to_activate",
        "topup_stripe_subscription_id",
        "topup_admin_id",
        "grace_started_at",
        # Team storage pool snapshot (Task 15, spec §6) — a commercial fact
        # (what the pool holds, what it costs the covering admin to overflow
        # it) exactly like the credit fields above it. The KEY is only added
        # to the payload at all for a self_serve org with an active coverer
        # (get_org) — this entry is what strips it for a non-admin when it
        # IS present, same redaction posture as pool_balance etc.
        "teamStorage",
    }
)


def redact_org_for_role(org: dict, role: str | None) -> dict:
    """Strip admin-only financial fields from an org payload for a non-admin.

    REMOVES the keys rather than zeroing them: a 0 pool balance reads as "the
    organization is out of credits", which is a lie a member would act on (they
    would go chase an admin who has plenty). An absent key is unambiguous, and
    the member UI renders "pulling from the org credits pool" instead of a
    number it was never given.

    Defaults CLOSED — any role that is not exactly "admin", including None or
    an unrecognised value, gets the redacted payload. Counterpart to
    subscriptions.service._pool_visible_to, which does the same job for the
    entitlements/credits payload.
    """
    if role == "admin":
        return org
    return {k: v for k, v in org.items() if k not in _ADMIN_ONLY_ORG_FIELDS}


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
        # select("*") pulls the dispersal / activation-floor / storage columns —
        # admin-only. Membership here is status != 'removed', so a SUSPENDED
        # admin appears with role='admin'; collapse them to non-admin so this
        # matches get_org (whose my_role read is filtered to active seats) and
        # authz.require_admin (is_org_admin counts ACTIVE rows only). One
        # definition of "admin" across all three, and it fails closed.
        effective_role = info.get("role") if info.get("status") == "active" else None
        out.append(redact_org_for_role(o, effective_role))
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

    wallet = wallets.read_wallet(db, "org", org_id)
    pool_balance = (wallet.get("bundle_balance", 0) + wallet.get("reserve_balance", 0)) if wallet else 0

    cumulative_paid_in = wallets.cumulative_paid_in(db, wallet["id"]) if wallet else 0

    effective_min = org.get("min_initial_purchase_credits") or _default_min_initial_purchase_credits()
    remaining_to_activate = max(0, effective_min - cumulative_paid_in)

    member_count_res = (
        db.table("org_members").select("id", count="exact").eq("org_id", org_id).eq("status", "active").execute()
    )
    member_count = member_count_res.count or 0

    # Admin contacts, visible to EVERY member (this function is member-gated).
    # A member's only remedy for a reached cap or a dry pool is "ask an admin",
    # which isn't actionable if they can't see who that is — and the seat
    # rollup that carries the rest of the roster (get_org_usage) is admin-only
    # by design. Deliberately admins only, and deliberately identity + email
    # only: no caps, no spend, no non-admin members.
    admin_rows = (
        db.table("org_members")
        .select("id, user_id, email")
        .eq("org_id", org_id)
        .eq("role", "admin")
        .eq("status", "active")
        .execute()
        .data
        or []
    )
    names: dict[str, str] = {}
    if admin_rows:
        profile_rows = (
            db.table("profiles").select("id, full_name").in_("id", [a["user_id"] for a in admin_rows]).execute().data
            or []
        )
        names = {p["id"]: p["full_name"] for p in profile_rows if p.get("full_name")}
    admins = [
        {"userId": a["user_id"], "email": _member_email(db, a), "fullName": names.get(a["user_id"])} for a in admin_rows
    ]

    from orgs import standing

    extra: dict = {
        "my_role": my_role,
        "pool_balance": pool_balance,
        "cumulative_paid_in": cumulative_paid_in,
        "remaining_to_activate": remaining_to_activate,
        "member_count": member_count,
        "admins": admins,
        # Configured grace-window length (spec §3/§6, Task 15) — a global
        # constant (env-configured, not per-org), so unlike the fields above
        # it carries nothing to redact; the lifecycle banner needs it to
        # render "loses access on {date}" instead of a vague "soon".
        "graceDays": standing.grace_days(),
    }

    # Team storage pool snapshot (Task 15, spec §6): the org billing console's
    # storage meter. Self-serve orgs only, and only once a coverer exists to
    # size the pool against — an enterprise org has no per-owner storage pool
    # at all, and a released self-serve org (covered_by NULL) has no owner to
    # bill overage to. The KEY is entirely ABSENT in both cases (not present-
    # but-zero) so the frontend can tell "no team storage pool" from "empty
    # pool". Admin-only via `_ADMIN_ONLY_ORG_FIELDS`, same redaction posture
    # as pool_balance/cumulative_paid_in above.
    if org.get("kind") == "self_serve" and org.get("covered_by"):
        from orgs import storage_guard

        used_bytes, pool_bytes, _ = storage_guard.pool_state(db, org["covered_by"])
        overage_gb = math.ceil(max(0, used_bytes - pool_bytes) / 2**30)
        extra["teamStorage"] = {
            "usedBytes": used_bytes,
            "poolBytes": pool_bytes,
            "overageGb": overage_gb,
            "ratePerGb": float(os.getenv("TEAM_STORAGE_OVERAGE_USD_PER_GB", "0.025")),
        }

    # Pool/dispersal/activation figures are stripped for non-admins (my_role is
    # read from ACTIVE seats only, so a suspended admin gets the member view).
    # Everything a member legitimately needs survives: the org's identity and
    # status, their own role, the member count, and who to ask for more.
    return redact_org_for_role({**org, **extra}, my_role)


async def update_org(db: Client, user_id: str, org_id: str, fields: dict) -> dict:
    """Update org name / default_member_cap. Admin only.

    `fields` is expected to come from `OrgUpdate.model_dump(exclude_unset=True)`
    at the router — keys explicitly present in the request (including an
    explicit `None` for default_member_cap, clearing members to uncapped) are
    written; omitted keys are left untouched. The dispersal has its own endpoint
    (set_org_dispersal) because it is the contract, not a display preference.
    Task 17: refuses on an archived/dissolved org when actually writing
    (empty `fields` is a plain read-back, not a mutation — no guard needed)."""
    authz.require_admin(db, user_id, org_id)
    if not fields:
        res = db.table("organizations").select("*").eq("id", org_id).maybe_single().execute()
        return res.data
    _require_live_org(_first_org(db, org_id))
    res = db.table("organizations").update(fields).eq("id", org_id).execute()
    return res.data[0] if res.data else None


def _teardown_archived_org_grants(db: Client, org_id: str) -> None:
    """Licensing Phase C, Task 4 (rule 12): drop every `project_members` row
    THIS org granted, once the org stops conferring access. TWO callers, which
    place it differently: `archive_org` runs it AFTER `archived_at` has landed
    (its load-bearing write), while `dissolve_org` runs it as part of its
    seat/invite teardown, BEFORE the terminal `dissolved_at` patch. Either way
    it is best-effort/never-raising, mirroring `_offboard`'s
    money-first-then-cleanup posture: a teardown failure must not undo or block
    the lifecycle transition that called it.

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
    except Exception:
        # The daily sweep's org-grant reconciliation (subscriptions/sweep.py)
        # cleans up any grant this best-effort pass missed.
        logger.exception("org grant teardown: revoke_org_granted_memberships failed org_id=%s", org_id)


async def archive_org(db: Client, user_id: str, org_id: str) -> dict:
    """Archive an org. Admin only. Sets archived_at, then tears down every access
    grant and link this org ever created (Task 4, rule 12) — see
    `_teardown_archived_org_grants`.

    No balance precondition: members hold no credits, so there is nothing
    stranded to reclaim first. Whatever is left in the POOL stays in the pool —
    an archived org's money is a support/refund decision (see the admin
    clawback endpoint), never something archiving silently disposes of.

    Task 17 (F5): dissolved is refused (allow_archived=True — re-archiving an
    already-archived org is fine and just refreshes its own archived_at), a
    blind write here on a DISSOLVED org would clobber the original archive
    date that dissolve_org deliberately preserves (`archived_at:
    org.get("archived_at") or now`).
    """
    authz.require_admin(db, user_id, org_id)
    _require_live_org(_first_org(db, org_id), allow_archived=True)

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

    Per-seat storage is deliberately NOT reported. Post-20260803000003,
    `usage_counters.total_storage_bytes` counts only PERSONAL (team_id IS NULL)
    artists — surfacing it here was both semantically stale (team bytes live on
    `organizations.storage_bytes`, kept by the storage triggers) and a privacy
    leak of members' non-org activity to org admins. Org-level storage stays
    available via the org row itself (get_org spreads `organizations.*`).

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

    pool_wallet = wallets.read_wallet(db, "org", org_id)
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
    spent_by_member: Counter[str] = Counter()
    for r in ledger_rows:
        member_id = (r.get("metadata") or {}).get("org_member_id")
        if r.get("kind") == "debit" and member_id:
            spent_by_member[member_id] += abs(r.get("delta", 0))

    seats = []
    for m in members:
        spent = spent_by_member.get(m["id"], 0)
        if m.get("status") == "removed" and spent == 0:
            continue

        email = _member_email(db, m)

        # None = no ceiling, whether that's an unset chain or an explicit -1.
        effective_cap = effective_member_cap(m.get("monthly_cap"), default_cap)
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


async def get_org_ledger(db: Client, user_id: str, org_id: str) -> list[dict]:
    """Admin-only: the org pool's newest 50 ledger rows (Task 15, spec §6) —
    the billing console's "Pool activity" feed. Reads straight off the pool
    wallet (`wallets.read_or_create_org_wallet` — create-on-miss, so a
    brand-new org with no ledger activity yet still resolves a wallet id and
    returns an empty list rather than 404ing).

    kind/action/delta/metadata/created_at only: no balance_after, no
    request_id — the console renders a feed, not a reconciliation tool, and
    those two are internal plumbing (idempotency key, running balance) that
    would need explaining rather than helping.
    """
    authz.require_admin(db, user_id, org_id)
    wallet = wallets.read_or_create_org_wallet(db, org_id)
    res = (
        db.table("credit_ledger")
        .select("kind, action, delta, metadata, created_at")
        .eq("wallet_id", wallet["id"])
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return res.data or []


# ============================================================================
# Invite flow (mirrors teams/service.py's invite_member/accept_invite/
# decline_invite; deltas noted inline where org semantics diverge)
# ============================================================================


def _self_serve_seat_room(db: Client, org_id: str) -> None:
    """Self-serve team-size gate (Task 7, spec §3), called from BOTH ends of
    the invite flow (invite_member and accept_invite) so a limit can't be
    dodged by inviting past it and accepting later.

    Enterprise orgs return immediately — no tier_entitlements read, so
    behavior for them is byte-identical to before this gate existed.

    The member count this gates on EXCLUDES the covering owner (`.neq
    ("user_id", covered_by)` below) — a coverer's own seat never eats into
    the limit they're paying for.

    covered_by is whose team_dials_for_user the limit binds to. It reads
    NULL only for a released org (release_coverage clears covered_at, never
    covered_by) or a pre-migration row that never got a coverer — either
    way there is no dials owner left to resolve a limit against, so the
    gate is skipped rather than guessed. A released org is already on the
    sweep's path to 'lapsed', which is the real backstop for that case.

    The effective ceiling is SEATS_PER_PRO's formula (owner decision
    2026-08-16): `min(dials.max_team_members, SEATS_PER_PRO * (1 +
    pro_member_count))`, pro_member_count being this org's ACTIVE members
    (excluding the coverer, who counts as the first Pro seat for free)
    whose resolved tier is Pro. This gates ADDING only — a member who
    unlocked a seat leaving later does NOT evict anyone, it just lowers the
    ceiling new invites/accepts are checked against.
    """
    org = _first_org(db, org_id)
    if not org or org.get("kind") != "self_serve":
        return
    if org.get("status") == "lapsed":
        raise TeamLapsedError("This team is paused. Reactivate it before inviting more members.")
    covered_by = org.get("covered_by")
    if covered_by is None:
        return

    from orgs import standing  # lazy import, mirrors claim_coverage above

    # ACCEPTED consequence (review r2): a covering owner who has downgraded
    # to Free reads max_team_members=0 here even DURING their grace window,
    # so invites (and accepts) are refused while grace is still running,
    # ahead of the sweep actually lapsing the org. That is intentional — a
    # team already heading toward lapse should not be allowed to grow — and
    # narrows spec §3's "invites still allowed during grace" for this case.
    dials = standing.team_dials_for_user(db, covered_by)
    members_res = (
        db.table("org_members")
        .select("user_id", count="exact")
        .eq("org_id", org_id)
        .eq("status", "active")
        .neq("user_id", covered_by)
        .execute()
    )
    active_count = members_res.count if members_res.count is not None else len(members_res.data or [])
    member_ids = [m["user_id"] for m in (members_res.data or [])]
    # Per-member reads: acceptable here — at most SEATS_PER_PRO*(1+few) ~ 10
    # members, and only on the invite/accept path, never a hot loop.
    pro_member_count = sum(1 for uid in member_ids if standing.resolve_tier_for_user(db, uid) == "pro")
    effective_limit = min(dials.max_team_members, SEATS_PER_PRO * (1 + pro_member_count))

    if active_count >= effective_limit:
        is_pro = dials.tier == "pro"
        next_step = "contact" if is_pro else "upgrade"
        tier_label = {"pro": "Pro", "basic": "Basic", "free": "Free"}.get(dials.tier) or "your plan"
        if is_pro and effective_limit < dials.max_team_members:
            message = (
                f"This team has {effective_limit} seats on Pro. Another Pro member joining unlocks "
                f"{SEATS_PER_PRO} more (up to {dials.max_team_members}) — or talk to us about Enterprise."
            )
        elif is_pro:
            message = (
                f"This team is at its {dials.max_team_members}-member limit on Pro. "
                "For bigger teams, talk to us about Enterprise."
            )
        else:
            message = (
                f"This team is at its {effective_limit}-member limit on {tier_label}. "
                f"Upgrade to Pro for teams of up to {SEATS_PER_PRO}."
            )
        raise TeamFullError(message, effective_limit, next_step)


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
    _require_live_org(_first_org(db, org_id))
    _self_serve_seat_room(db, org_id)
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
                    # Re-inviting RESTARTS the 48h window (and revives an
                    # expired row) — that is the whole point of resending.
                    "expires_at": (datetime.now(UTC) + INVITE_TTL).isoformat(),
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

    ALSO filtered on expires_at, not just status: the sweep that flips a lapsed
    invite to 'expired' runs once a day, so between lapsing and being swept a
    row still reads 'pending' while accept_invite would reject it. Listing it
    would invite the admin to wait on something already dead.
    """
    authz.require_admin(db, user_id, org_id)
    res = (
        db.table("pending_org_invites")
        .select("*")
        .eq("org_id", org_id)
        .eq("status", "pending")
        .gt("expires_at", datetime.now(UTC).isoformat())
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def expire_stale_invites(db: Client, limit: int = 1000) -> int:
    """Flip lapsed pending invites to 'expired' and tell the inviter. Returns
    the number expired. Called once a day from the billing sweep.

    This changes NO permissions: accept_invite has always refused an invite past
    its expires_at, so nothing here gates anything. What it adds is a one-time
    edge to hang the "they never accepted" notification off — a row that merely
    sits at status='pending' past its date looks identical on every subsequent
    pass, so without a terminal state the admin would be told every day forever.

    Ordering: the status UPDATE lands FIRST, and it is filtered on
    status='pending', so a concurrent sweep (or a retry after a crash) matches
    zero rows and sends nothing. A notification that fails afterwards is lost
    rather than duplicated — the right trade for a courtesy message, since the
    admin can always see the invite is gone from the pending list.

    NOT batched per admin: an admin who let three invites lapse gets three
    notifications, each naming its invitee, because "chase this person" is the
    action and a merged row would bury it.

    ponytail: LATENCY IS THE SWEEP'S CADENCE. On the current once-a-day
    schedule an invite that lapses just after a run is reported up to ~24h
    late. Point Cloud Scheduler at /internal/billing-sweep more often to
    tighten it — this function is idempotent and cheap (one indexed scan).
    """
    now = datetime.now(UTC)
    try:
        stale = (
            db.table("pending_org_invites")
            .select("id, org_id, email, invited_by")
            .eq("status", "pending")
            .lt("expires_at", now.isoformat())
            .limit(limit)
            .execute()
        ).data or []
    except Exception:
        logger.exception("expire_stale_invites: scan failed")
        return 0
    if not stale:
        return 0

    ids = [r["id"] for r in stale]
    try:
        updated = (
            db.table("pending_org_invites")
            .update({"status": "expired"})
            .in_("id", ids)
            .eq("status", "pending")
            .execute()
        ).data or []
    except Exception:
        logger.exception("expire_stale_invites: status update failed for %d invites", len(ids))
        return 0

    # Notify only for rows THIS call actually transitioned.
    claimed = {r["id"] for r in updated}
    for invite in stale:
        if invite["id"] not in claimed or not invite.get("invited_by"):
            continue
        try:
            _notify_invite_expired(db, invite)
        except Exception:
            logger.exception("expire_stale_invites: notify failed for invite %s", invite["id"])
    return len(claimed)


def _notify_invite_expired(db: Client, invite: dict) -> None:
    """In-app row for the admin who sent an invite that lapsed unanswered.

    Deliberately NOT type='invitation' — that pair (invitation + entity_type
    'org') is what NotificationRow keys the Accept/Decline buttons off, and this
    row is a report, not something to action. 'status_change' is already
    CHECK-allowed (notifications_type_check, 20260629000004) and renders
    button-less, so this needs no migration.
    """
    org_name = _org_name(db, invite["org_id"], "your organization")
    db.table("notifications").insert(
        {
            "user_id": invite["invited_by"],
            "type": "status_change",
            "title": "Invite expired",
            "message": (
                f"{invite['email']} didn't accept your invitation to {org_name} within 48 hours, "
                "so it has expired. You can send a new one from the organization page."
            ),
            "entity_type": "org",
            "entity_id": invite["org_id"],
            "metadata": {"org_id": invite["org_id"], "email": invite["email"], "reason": "invite_expired"},
        }
    ).execute()


async def cancel_invite(db: Client, user_id: str, org_id: str, invite_id: str) -> dict:
    """Delete a pending invite. Admin only."""
    authz.require_admin(db, user_id, org_id)
    _require_live_org(_first_org(db, org_id))
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

    # Task 17: a dissolved/archived org can't gain a new seat. Invite-validity
    # checks above run first (a bad token must 404, never leak lifecycle
    # state), so this can raise its own 409 straight through — nothing below
    # catches HTTPException specially, same as transfer_credits_to_pool.
    _require_live_org(_first_org(db, invite["org_id"]))

    # Team-size gate (Task 7, spec §3): re-checked here, not just at invite
    # time, so a team that fills up (or lapses) between invite and accept
    # can't be grown by an invitee accepting late. Translated to
    # InviteInvalidError (already mapped to 410 by the router) rather than
    # surfaced as its own status — from the invitee's side this reads the
    # same as any other "this invite can't be actioned" outcome, and the
    # invite row is deliberately left untouched (still 'pending') so the
    # 48h expiry — not this rejection — is what eventually collects it.
    try:
        _self_serve_seat_room(db, invite["org_id"])
    except TeamFullError as exc:
        raise InviteInvalidError(
            f"This team is at its member limit ({exc.limit}). Ask the covering admin to upgrade before accepting."
        ) from None
    except TeamLapsedError:
        raise InviteInvalidError(
            "This team is paused. Ask the covering admin to reactivate it before accepting."
        ) from None

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
    _close_invite_notification(db, user_id, token)

    return {"type": "accepted", "org_id": invite["org_id"]}


def create_org_invite_notification(
    db: Client, target_user_id: str, org_id: str, inviter_user_id: str, token: str
) -> None:
    """In-app Accept/Decline row for an invited EXISTING user.

    type='invitation' + entity_type='org' is the pair NotificationRow keys the
    org Accept/Decline buttons off. (Until 2026-08-16 there was a second,
    board-teams invite type on this table; it was removed with that module and
    must not be reintroduced here.) 'invitation' is already
    CHECK-allowed (20260629000004), so this needs no migration — registry's
    'invitation' rows carry entity_type 'work'/NULL and keep rendering as before.

    `metadata.token` is what the buttons submit, same as the team row.
    """
    org_name = _org_name(db, org_id, "an organization")
    inviter = db.table("profiles").select("full_name").eq("id", inviter_user_id).maybe_single().execute()
    inviter_name = ((inviter.data or {}) if inviter else {}).get("full_name") or "Someone"

    db.table("notifications").insert(
        {
            "user_id": target_user_id,
            "type": "invitation",
            "title": f"Invited to {org_name}",
            "message": f'{inviter_name} invited you to join the organization "{org_name}".',
            "entity_type": "org",
            "entity_id": org_id,
            "metadata": {"org_id": org_id, "token": token},
        }
    ).execute()


def create_org_join_notifications(db: Client, org_id: str, member_user_id: str, member_email: str) -> None:
    """In-app notifications for a fresh seat acceptance, on the unified
    `notifications` table (entity_type='org').

    Two recipients, because acceptance is the one org event both sides act on:
      - the new member, whose `billing_context_org_id` accept_invite just
        repointed at the org — the mirror of the billing-reverted notice
        offboarding sends on the way OUT (orgs/router._notify_billing_reverted_background)
      - every ACTIVE admin, who now has a roster change and a cap to consider

    NOT an actionable 'invitation' row: NotificationRow renders Accept/Decline
    for that type, and these are after-the-fact confirmations with no actions,
    so 'confirmation' is correct.

    Best-effort and never raises — the seat is already active and committed by
    the time this runs; a notification failure must not fail the acceptance.
    """
    org_name = _org_name(db, org_id, "your organization")

    rows = [
        {
            "user_id": member_user_id,
            "type": "confirmation",
            "title": f"You joined {org_name}",
            "message": (
                f"Your Msanii usage now bills to {org_name}'s shared credit pool. "
                "You can switch back to personal billing from your Profile."
            ),
            "entity_type": "org",
            "entity_id": org_id,
            "metadata": {"org_id": org_id},
        }
    ]

    admins = (
        db.table("org_members")
        .select("user_id")
        .eq("org_id", org_id)
        .eq("role", "admin")
        .eq("status", "active")
        .execute()
        .data
        or []
    )
    rows += [
        {
            "user_id": a["user_id"],
            "type": "confirmation",
            "title": "Invite accepted",
            "message": f"{member_email} joined {org_name}.",
            "entity_type": "org",
            "entity_id": org_id,
            "metadata": {"org_id": org_id, "member_user_id": member_user_id},
        }
        for a in admins
        if a.get("user_id") and a["user_id"] != member_user_id
    ]

    db.table("notifications").insert(rows).execute()


def _close_invite_notification(db: Client, user_id: str, token: str) -> None:
    """Retire the bell's copy of an invite once it has been actioned.

    The invite can be answered from the emailed claim page OR from the in-app
    notification; only the latter marks its own row read. Without this, the
    bell keeps showing live Accept/Decline buttons for an invite that is
    already settled. Best-effort by construction — see the registry helper."""
    from registry.service import mark_invite_notifications_read

    mark_invite_notifications_read(db, user_id, token)


async def decline_invite(db: Client, user_id: str, user_email: str, token: str) -> dict:
    """Decline an invite by token. The caller's email must match the invite.

    Task 17: deliberately NOT gated by _require_live_org — declining is
    harmless on any org state (it writes nothing an org lifecycle protects),
    and blocking someone from saying no to a defunct team would be user-
    hostile for no protective benefit.
    """
    invite = await get_invite_by_token(db, token)
    if not invite:
        raise ValueError("Invite not found")
    if invite["email"].lower() != user_email.lower():
        raise PermissionError("This invite was sent to a different email")
    db.table("pending_org_invites").update({"status": "declined"}).eq("id", invite["id"]).execute()
    _close_invite_notification(db, user_id, token)
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
    _require_live_org(_first_org(db, org_id))
    try:
        res = db.table("org_members").update({"role": role}).eq("id", member_id).eq("org_id", org_id).execute()
    except Exception as exc:
        if _is_last_admin_error(exc):
            raise LastAdminError("You are the only admin of this organization — promote another member first") from exc
        raise
    if not res.data:
        raise ValueError("Member not found")
    return res.data[0]


async def list_org_roster(db: Client, user_id: str, org_id: str) -> list[dict]:
    """Member-visible roster (spec 2026-08-16 §3): ACTIVE seats with profile
    name/avatar, NO emails (those stay on the admin-only /usage seats). Feeds
    the board assignee picker / "Created by" filter / board-members picker."""
    authz.require_member(db, user_id, org_id)
    # ...and LIVENESS, like every other predicate this feature added
    # (boards.authz.is_live_org_member, require_org_admin, _live_org_ids).
    # is_org_member alone ignores archived_at/'lapsed', so without this a
    # lapsed org — whose boards and artists have all gone inert — would still
    # hand out its full roster with names and avatars.
    if org_id not in artist_access.live_org_ids(db, user_id):
        raise HTTPException(status_code=404, detail="Organization not found")
    rows = (
        db.table("org_members").select("user_id, role").eq("org_id", org_id).eq("status", "active").execute().data or []
    )
    ids = [r["user_id"] for r in rows if r.get("user_id")]
    profiles = {}
    if ids:
        res = db.table("profiles").select("id, full_name, avatar_url").in_("id", ids).execute()
        profiles = {p["id"]: p for p in (res.data or [])}
    return [
        {
            "user_id": r["user_id"],
            "role": r["role"],
            "full_name": profiles.get(r["user_id"], {}).get("full_name"),
            "avatar_url": profiles.get(r["user_id"], {}).get("avatar_url"),
        }
        for r in rows
        if r.get("user_id")
    ]


def _purge_member_from_org_boards(db: Client, org_id: str, member_user_id: str) -> None:
    """REMOVAL only: drop the member's per-board visibility rows on this org's
    boards, so a later re-invite doesn't silently hand back access to boards
    that were narrowed to them.

    Task ASSIGNMENTS are deliberately left alone (unlike the old
    team_member_removal_cleanup trigger, which hard-deleted them): an
    offboarded seat is not `active`, so live_org_ids/can_access_board already
    deny the board and the board_task_assignees RLS policy already hides the
    row. The UI renders such an assignee as "(no longer in team)". Deleting
    history to enforce access that is already denied buys nothing and cannot
    be undone.
    """
    board_ids = [b["id"] for b in (db.table("boards").select("id").eq("team_id", org_id).execute().data or [])]
    if not board_ids:
        return
    db.table("board_members").delete().eq("user_id", member_user_id).in_("board_id", board_ids).execute()


def _revert_org_boards(db: Client, org_id: str, fallback_user: str) -> None:
    """Dissolve: every team board becomes a personal board of a PERSON, picked
    exactly the way _dissolve_recipients picks one for an artist — the creator
    if they still hold an ACTIVE seat, else the dissolving admin. A board left
    owned by an already-removed member would be reachable by nobody once the
    org is archived.

    Not best-effort: a board still pointing at a dissolved org is invisible to
    everyone, so a failure here must abort dissolve (which is retryable —
    reverted boards no longer match team_id=org_id).
    """
    boards = db.table("boards").select("id, owner_id").eq("team_id", org_id).execute().data or []
    if not boards:
        return
    active = {
        m["user_id"]
        for m in (
            db.table("org_members").select("user_id").eq("org_id", org_id).eq("status", "active").execute().data or []
        )
    }
    db.table("board_members").delete().in_("board_id", [b["id"] for b in boards]).execute()
    for b in boards:
        creator = b.get("owner_id")
        recipient = creator if (creator and creator in active) else fallback_user
        # artist_id is cleared too: uq_boards_personal_artist is UNIQUE
        # (owner_id, artist_id) WHERE team_id IS NULL AND artist_id IS NOT NULL,
        # so a team board carrying an artist_id whose recipient already owns a
        # personal board for that artist would raise 23505 here — and every
        # retry would hit the same row, wedging dissolve forever with the
        # artists already handed back. The team's artist alias means nothing
        # once the board belongs to a person.
        db.table("boards").update({"team_id": None, "restricted": False, "owner_id": recipient, "artist_id": None}).eq(
            "id", b["id"]
        ).execute()


def _revoke_offboarded_member_access(db: Client, org_id: str, member_user_id: str | None, final_status: str) -> None:
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
    from this module at its own top level.

    `final_status` distinguishes the two offboards: only 'removed' purges
    board membership (see `_purge_member_from_org_boards`); a 'suspended'
    seat destroys nothing, because reactivate_member can bring it back and
    nothing would restore deleted rows."""
    if not member_user_id:
        return
    try:
        from orgs.projects import revoke_org_granted_memberships

        revoke_org_granted_memberships(db, org_id, user_id=member_user_id)
    except Exception:
        # The daily sweep's org-grant reconciliation (subscriptions/sweep.py)
        # cleans up any grant this best-effort pass missed.
        logger.exception(
            "_offboard: revoke_org_granted_memberships failed org_id=%s member_user_id=%s", org_id, member_user_id
        )

    # Board visibility rows: REMOVAL only. Suspend is reversible
    # (reactivate_member restores the seat) and nothing would restore these,
    # so a suspend must not delete anything. Its OWN try/except, separate from
    # the grant revoke above, so a board failure can't mask a grant failure.
    if final_status == "removed":
        try:
            _purge_member_from_org_boards(db, org_id, member_user_id)
        except Exception:
            logger.exception(
                "_offboard: board membership purge failed org_id=%s member_user_id=%s", org_id, member_user_id
            )


def cancel_topup(db: Client, org_id: str) -> bool:
    """Cancel this org's recurring credit top-up at Stripe and release the two
    columns that point at it. Returns False when there was nothing to cancel.

    ONE implementation, three callers: POST /orgs/{id}/cancel-topup,
    `_cancel_topup_if_purchaser` below (the paying admin being offboarded)
    and dissolve_org step 2.

    Order matters: Stripe first, columns second. A failed cancel PROPAGATES
    with the columns intact — the pointer is what lets an admin retry, and
    clearing it would leave a card silently billing every month with nothing
    in the DB naming it. (customer.subscription.deleted clears the same two
    columns when the subscription actually goes away, so the state converges
    either way.)
    """
    res = db.table("organizations").select("topup_stripe_subscription_id").eq("id", org_id).maybe_single().execute()
    sub_id = ((res.data if res else None) or {}).get("topup_stripe_subscription_id")
    if not sub_id:
        return False

    from subscriptions.stripe_client import get_stripe

    get_stripe().Subscription.delete(sub_id)
    db.table("organizations").update({"topup_stripe_subscription_id": None, "topup_admin_id": None}).eq(
        "id", org_id
    ).execute()
    return True


def _cancel_topup_if_purchaser(db: Client, org_id: str, member_user_id: str | None) -> None:
    """The recurring top-up is billed to the PURCHASING ADMIN's own card
    (spec §4.3), so offboarding that admin must stop the charge — they no
    longer run this team — and the remaining admins have to learn the pool
    stopped refilling, since only an admin can start a new one.

    Best-effort and never raises, exactly like `_revoke_offboarded_member_access`
    beside it: the status transition has already landed by the time this runs.
    A failed cancel leaves the org's columns pointing at the subscription, so
    POST /orgs/{id}/cancel-topup can retry it.
    """
    if not member_user_id:
        return
    try:
        res = db.table("organizations").select("topup_admin_id").eq("id", org_id).maybe_single().execute()
        if ((res.data if res else None) or {}).get("topup_admin_id") != member_user_id:
            return
        if not cancel_topup(db, org_id):
            return
        analytics_capture(member_user_id, "org_topup_canceled", {"org_id": org_id, "trigger": "offboard"})
        org_name = _org_name(db, org_id, "your organization")
        admins = (
            db.table("org_members")
            .select("user_id")
            .eq("org_id", org_id)
            .eq("role", "admin")
            .eq("status", "active")
            .execute()
            .data
            or []
        )
        rows = [
            {
                "user_id": a["user_id"],
                "type": "confirmation",
                "title": "Monthly credit top-up canceled",
                "message": (
                    f"The admin who paid for {org_name}'s monthly credit top-up is no longer on the team, "
                    "so it was canceled. Any admin can start a new one."
                ),
                "entity_type": "org",
                "entity_id": org_id,
                "metadata": {"org_id": org_id},
            }
            for a in admins
            if a.get("user_id") and a["user_id"] != member_user_id
        ]
        if rows:
            db.table("notifications").insert(rows).execute()
    except Exception:
        logger.exception("_offboard: top-up cancel failed org_id=%s member_user_id=%s", org_id, member_user_id)


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
    _require_live_org(_first_org(db, org_id))

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
        row = updated.data[0]

    _revoke_offboarded_member_access(db, org_id, row.get("user_id"), final_status)
    _cancel_topup_if_purchaser(db, org_id, row.get("user_id"))
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
    _require_live_org(_first_org(db, org_id))
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

    THREE distinct settings, which is why None and -1 are not the same thing:
      cap=N     -> this member's own ceiling.
      cap=None  -> INHERIT: fall through to the org's `default_member_cap`
                   (2,000 unless an admin changed it).
      cap=-1    -> NO LIMIT for this member, whatever the org default says.
                   Since every org now carries a default, clearing to None can
                   no longer express "unlimited" — hence the sentinel (the same
                   "-1 is unlimited" idiom tier_entitlements uses). It is
                   normalized back to None on every read (_member_cap) and
                   inside debit_credits, so nothing downstream sees a -1.

    Caps may sum to more than the monthly dispersal on purpose: most members
    never reach theirs, and the pool is the real ceiling.
    """
    authz.require_admin(db, user_id, org_id)
    _require_live_org(_first_org(db, org_id))
    _require_org_member(db, org_id, member_id)
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


def _active_seat_id(db: Client, org_id: str, user_id: str) -> str | None:
    """org_members.id of the caller's ACTIVE seat in this org, or None."""
    res = (
        db.table("org_members")
        .select("id")
        .eq("org_id", org_id)
        .eq("user_id", user_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    return ((res.data if res else None) or {}).get("id")


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
    _require_live_org(_first_org(db, org_id))

    member_id = _active_seat_id(db, org_id, user_id)
    if not member_id:
        # Defensive: require_member just confirmed an active seat exists —
        # this should be unreachable outside of a race with a concurrent
        # offboard.
        raise ValueError("Member not found")

    payload: dict = {"org_id": org_id, "org_member_id": member_id}
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
    return {"request": request, "org_member_id": member_id}


async def list_credit_requests(db: Client, user_id: str, org_id: str) -> list[dict]:
    """Admins see every request for the org (newest first); a non-admin
    member sees only their own."""
    authz.require_member(db, user_id, org_id)

    query = db.table("credit_requests").select("*").eq("org_id", org_id)
    if not authz.is_org_admin(db, user_id, org_id):
        query = query.eq("org_member_id", _active_seat_id(db, org_id, user_id))

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
    _require_live_org(_first_org(db, org_id))

    current = db.table("credit_requests").select("*").eq("id", request_id).eq("org_id", org_id).maybe_single().execute()
    request = current.data if current else None
    if not request:
        raise CreditRequestNotFoundError("Credit request not found")
    if request["status"] != "pending":
        raise CreditRequestAlreadyResolvedError("This request has already been resolved")

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
    _require_live_org(_first_org(db, org_id))

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


# ============================================================================
# Coverage claim / release (spec 2026-08-15 §2 rev 2, Task 5) — coverage is
# CLAIMED by an admin who wants it, never assigned to someone else.
# ============================================================================


def _first_org(db: Client, org_id: str) -> dict | None:
    """Lightweight org read: kind/status/coverage/lifecycle fields only — what
    a lifecycle guard needs to decide, not a full row. Shared org-row fetch
    for guards/helpers (this module's coverage/dissolve/transfer endpoints
    and `orgs/projects.py`'s access guards), not an enumerated call list."""
    res = (
        db.table("organizations")
        .select("kind, status, covered_by, covered_at, archived_at, dissolved_at")
        .eq("id", org_id)
        .maybe_single()
        .execute()
    )
    return res.data if res else None


def _require_live_org(org: dict | None, *, allow_archived: bool = False) -> None:
    """Task 17: refuse a mutation on an archived or dissolved org. Call AFTER
    membership/admin authz (a non-member of a dissolved org must still get
    403/404, never a 409 that leaks the org's lifecycle state) with the dict
    `_first_org(db, org_id)` returned.

    Dissolved is always terminal and blocks everything. Archived blocks
    everything EXCEPT the ops that legitimately act on an archived org —
    archive_org (re-archiving an already-archived org is a harmless refresh)
    and unarchive_org (the whole point) both pass allow_archived=True to skip
    just that check while still refusing a (terminal) dissolved org. Every
    other mutating caller uses the default.

    `org` failing to be a dict (row not found, or any other unexpected shape)
    fails OPEN — nothing is enforced. That's deliberate: this is a refinement
    on top of the real access gate (authz, called first by every caller),
    never the only thing standing between a caller and the org, so an
    unreadable row must not itself become a way to block a legitimate call.
    """
    if not isinstance(org, dict):
        return
    if org.get("dissolved_at"):
        # dissolved_at is written ONLY by dissolve_org, which is gated to
        # kind == "self_serve" — this branch is unreachable for an enterprise
        # org, so "team" (the self-serve UI vocabulary) is correct here even
        # though this guard is shared with the (kind-agnostic) archived check
        # below.
        raise HTTPException(status_code=409, detail="This team has been dissolved")
    if not allow_archived and org.get("archived_at"):
        raise HTTPException(status_code=409, detail="This organization is archived.")


def _require_self_serve_org(db: Client, org_id: str) -> dict:
    """Load the org and refuse (409) unless it's self-serve — the shared
    preamble for every op that only makes sense on a self-serve team (claim/
    release coverage, unarchive, dissolve preview/execute, pool transfer).
    Call AFTER admin authz. HTTPException inline, not an exception class —
    every call site maps it identically, so there's nothing a `raise` site
    elsewhere would customize."""
    org = _first_org(db, org_id)
    if not org or org.get("kind") != "self_serve":
        raise HTTPException(status_code=409, detail="This organization is managed by Msanii")
    return org


async def claim_coverage(db: Client, user_id: str, org_id: str) -> dict:
    """Spec §2 rev 2: coverage is CLAIMED, never assigned — slot, storage and
    top-up liability move only onto someone who asked for them."""
    from orgs import standing

    authz.require_admin(db, user_id, org_id)
    org = _require_self_serve_org(db, org_id)
    try:
        _require_live_org(org)
    except HTTPException as exc:
        # claim_coverage's router maps ValueError -> 400 (not HTTPException ->
        # its own status), so re-raise as ValueError to keep that contract
        # while still routing the actual check through the one shared guard.
        raise ValueError(str(exc.detail)) from exc
    if org.get("covered_by") == user_id and org.get("covered_at"):
        return org  # idempotent — already actively covering
    # NOTE covered_by == user_id with covered_at NULL is a RELEASED org being
    # re-claimed by its last coverer — that must fall through to the claim
    # (review r2 hole 3: the old guard made release-then-reclaim impossible).
    standing.require_free_slot(db, user_id)
    patch = {"covered_by": user_id, "covered_at": _now_iso(), "grace_started_at": None}
    if org.get("status") == "lapsed":
        # Storage guard (Task 12): a lapsed team may not wake up over-pool.
        from orgs.storage_guard import reactivation_allowed

        if not reactivation_allowed(db, user_id):
            raise HTTPException(status_code=402, detail=REACTIVATION_BLOCKED_DETAIL)
        patch["status"] = "active"
    res = db.table("organizations").update(patch).eq("id", org_id).execute()
    return res.data[0]


async def release_coverage(db: Client, user_id: str, org_id: str) -> dict:
    """Current coverer steps away. covered_by DELIBERATELY stays (last-coverer
    storage attribution + the sweep's keep-rule read it); the SWEEP starts
    grace — one writer for grace state, so a release+reclaim inside a day
    never emits a scare notification.

    The write is conditioned on covered_by=user_id, not just the read above:
    between that read and this update, a rival admin's claim could already
    have landed, and a blind write would null THEIR fresh covered_at out from
    under them (covered_by=B + covered_at=None reads as "released" to the
    sweep, wedging B's org in grace it never earned). Zero rows matched means
    exactly that race happened — 403, nothing written.
    """
    authz.require_admin(db, user_id, org_id)
    org = _require_self_serve_org(db, org_id)
    # Task 17 judgment call: deliberately NOT gated by _require_live_org.
    # Releasing frees no additional resource — an archived (or dissolved) org
    # already holds no slot and confers nothing (unarchive_org's docstring) —
    # so refusing it here would only strand the covering admin's own
    # free-slot bookkeeping on a dead org for no protective benefit.
    if org.get("covered_by") != user_id:
        raise HTTPException(status_code=403, detail="Only the covering admin can release coverage")
    # Marker the sweep reads: covered_at=None means "released, evaluate me".
    res = db.table("organizations").update({"covered_at": None}).eq("id", org_id).eq("covered_by", user_id).execute()
    if not res.data:
        raise HTTPException(
            status_code=403,
            detail="Coverage changed while you were releasing — refresh to see the current coverer.",
        )
    return res.data[0]


async def unarchive_org(db: Client, user_id: str, org_id: str) -> dict:
    """Self-serve reactivation of an archived org (spec §3, Task 8). The
    unarchiving admin re-claims coverage in the same call: an archived org
    holds no slot (`standing.count_covered_orgs` excludes it), so coming back
    needs a free slot the same way claiming a released org does, PLUS the
    storage guard a lapsed reactivation needs — an archived team's bytes
    still count against the owner's pool even while it holds no slot
    (`storage_guard.pool_state`).

    Standalone rather than sharing claim_coverage's guard block (a
    `_require_claimable` extraction was considered and dropped): the two
    diverge on almost every branch — claim tolerates an active org and folds
    archived/dissolved into one 400, this REQUIRES archived_at and splits
    dissolved into its own 409 — so a shared helper would carry as many
    branches as it would save lines.

    LIMITATION (pre-existing, not fixed here): org-granted project
    memberships and members' billing contexts torn down at archive time
    (`_teardown_archived_org_grants`, called from `archive_org`) are NOT
    restored by unarchiving. A member who lost project access via this org
    must be re-added, and anyone whose `billing_context_org_id` was reverted
    off the org must re-select it from Profile.
    """
    from orgs import standing
    from orgs.storage_guard import reactivation_allowed

    authz.require_admin(db, user_id, org_id)
    org = _require_self_serve_org(db, org_id)
    _require_live_org(org, allow_archived=True)  # dissolved still blocks; archived is the whole point
    if not org.get("archived_at"):
        raise HTTPException(status_code=400, detail="This organization is not archived")

    # NOT atomic with the write below: two concurrent unarchive calls can both
    # pass this slot check before either lands, oversubscribing the slot by
    # one. Accepted — this endpoint is admin-only and low-frequency, not worth
    # a lock for.
    standing.require_free_slot(db, user_id)  # raises NoSlotError -> 402 (router)
    if not reactivation_allowed(db, user_id):
        raise HTTPException(status_code=402, detail=REACTIVATION_BLOCKED_DETAIL)

    patch = {
        "archived_at": None,
        "covered_by": user_id,
        "covered_at": _now_iso(),
        "status": "active",
        "grace_started_at": None,
    }
    res = db.table("organizations").update(patch).eq("id", org_id).execute()
    return res.data[0]


# ============================================================================
# Dissolve (Task 9, spec §3) — the terminal self-serve lifecycle op, and the
# only one that hands artists back to people. SOFT: the org row, its pool
# wallet and every ledger line are RETAINED (support still needs to read
# them); what actually moves is the artists, the pool's purchased reserve,
# the top-up subscription and the seats. Nothing here deletes anything.
# ============================================================================


def _dissolve_recipients(db: Client, org_id: str, fallback_user: str) -> list[dict]:
    """Spec §3: creator if they still hold an ACTIVE seat; else the dissolving
    admin. artists.user_id is the CREATOR (never 'owner' semantics) — after
    team_id goes NULL it becomes the owner again, which is exactly why the
    fallback must REASSIGN user_id, not just detach."""
    artists = (db.table("artists").select("id, name, user_id").eq("team_id", org_id).execute()).data or []
    active = {
        m["user_id"]
        for m in (db.table("org_members").select("user_id").eq("org_id", org_id).eq("status", "active").execute()).data
        or []
    }
    out = []
    for a in artists:
        creator = a.get("user_id")
        keep = creator is not None and creator in active
        out.append(
            {
                "artist_id": a["id"],
                "artist_name": a.get("name"),
                "recipient": creator if keep else fallback_user,
                "fallback": not keep,
            }
        )
    return out


async def dissolve_preview(db: Client, user_id: str, org_id: str) -> dict:
    """Exactly what POST /orgs/{id}/dissolve will do, for the confirm dialog.
    Same AUTHZ as the execute (active admin, self_serve) — a preview any
    member could read would hand out the roster and the pool balance.

    Unlike the execute, this does NOT check dissolved_at at all (Task 17
    deliberately leaves it that way) — dissolve_org's own dissolved_at check
    is an idempotent short-circuit (`{"already": True}`), not a refusal, and
    it checks ONLY dissolved_at, never archived_at: archived orgs can be
    dissolved (that's deliberate — dissolve is the terminal step, archived is
    not a barrier to it), so this preview can't gate on archived_at either
    without lying about what the execute will do. It ISN'T pure read-only,
    either: `wallets.read_or_create_org_wallet` inserts a wallet row on a
    miss, and `_member_email` backfills a NULL `org_members.email`. Both are
    benign, idempotent housekeeping — they touch no lifecycle-protected
    resource (no artist, no seat, no money moves) — which is the actual
    reason this stays unguarded, not "read-only."

    TWO credit numbers, because the pool's buckets die differently:
    `forfeitReserve` is purchased/comped credits, which the execute claws back
    (debit_credits' clawback branch is reserve-only, by design). `inertBundle`
    is whatever monthly dispersal is still sitting in the EXPIRING bucket —
    nothing reclaims it and no new expiry machinery is added; it is simply left
    on the wallet of an archived org, unspendable because billing-context
    resolution refuses an archived org. Stated rather than silently zeroed so
    the number the admin is shown matches the ledger afterwards.
    """
    authz.require_admin(db, user_id, org_id)
    _require_self_serve_org(db, org_id)

    members = (
        db.table("org_members").select("id, user_id, email").eq("org_id", org_id).eq("status", "active").execute()
    ).data or []
    by_user = {m["user_id"]: m for m in members}
    wallet = wallets.read_or_create_org_wallet(db, org_id)
    rows = _dissolve_recipients(db, org_id, user_id)

    # Every recipient is an ACTIVE member by construction (creator-with-a-seat,
    # or the calling admin), so the seat row is the email source; _member_email
    # heals a NULL off auth.users once, and None is fine — the dialog can name
    # the artist without it.
    emails: dict[str, str | None] = {}
    for r in rows:
        uid = r["recipient"]
        if uid not in emails:
            member = by_user.get(uid)
            emails[uid] = _member_email(db, member) if member else None

    return {
        "recipients": [
            {
                "artistId": r["artist_id"],
                "artistName": r["artist_name"],
                "userId": r["recipient"],
                "email": emails.get(r["recipient"]),
                "fallback": r["fallback"],
            }
            for r in rows
        ],
        "forfeitReserve": wallet.get("reserve_balance") or 0,
        "inertBundle": wallet.get("bundle_balance") or 0,
        "memberCount": len(members),
    }


async def dissolve_org(db: Client, user_id: str, org_id: str, confirm_name: str) -> dict:
    """Dissolve a self-serve team. Active admin only, typed-name confirmed, and
    IDEMPOTENT — a second call returns `{"already": True}` and writes nothing.

    MONEY FIRST, then the work that is worse half-done than not done:
      1. forfeit the pool's purchased RESERVE (one fixed request_id, so a retry
         can never claw back twice); the expiring bundle is left inert.
      2. cancel the top-up subscription at Stripe.
      3. revert every team artist to a person (creator if they still hold a
         seat, else the dissolving admin).
      3b. revert every team board to a person, same recipient rule.
      4. re-derive the storage totals the revert just invalidated.
      5. soft-remove the seats, expire the open invites, drop org-granted
         project access.
      6. stamp dissolved_at (+ archived_at, unless it was already archived).

    Steps 1, 2, 4 and 5 are best-effort: a stale byte total, an uncanceled
    subscription or a surviving grant is a support ticket, not a reason to
    strand a team mid-dissolve (and the daily sweep re-reconciles 4 and 5).
    Steps 3, 3b and 6 ABORT with a 500 — the artist and board reverts are the
    entire point of the operation, and they deliberately run BEFORE the seats
    are removed, so a crash anywhere leaves an intact admin who can retry. A
    retry is safe and convergent: the forfeit's request_id dedupes, an
    already-reverted artist or board no longer matches team_id=org_id, and a
    second call after step 6 lands returns {"already": True}.
    """
    authz.require_admin(db, user_id, org_id)
    org = _require_self_serve_org(db, org_id)
    if org.get("dissolved_at"):
        return {"already": True}

    # The confirmation needs the name verbatim (not _org_name's "your
    # organization" fallback, which a user could type).
    res = db.table("organizations").select("name").eq("id", org_id).maybe_single().execute()
    row = (res.data if res else None) or {}
    # Both sides trimmed: a stored name that picked up a trailing space is
    # otherwise IMPOSSIBLE to type back, i.e. impossible to dissolve.
    name = (row.get("name") or "").strip()
    if not name or confirm_name.strip() != name:
        # Nothing written yet, and nothing after this point is conditional.
        raise HTTPException(status_code=400, detail="Type the team name exactly as it appears to confirm")

    # 1. MONEY FIRST — forfeit the purchased reserve. Reserve-only and clamped
    # inside the RPC; p_member_id is deliberately absent (debit_credits rejects
    # a clawback that names a member — a forfeit is not member spend and must
    # not touch anyone's cap counter).
    try:
        wallet = wallets.read_or_create_org_wallet(db, org_id)
        reserve = wallet.get("reserve_balance") or 0
        if reserve > 0:
            db.rpc(
                "debit_credits",
                {
                    "p_wallet_id": wallet["id"],
                    "p_amount": reserve,
                    "p_action": "dissolve_forfeit",
                    "p_request_id": f"dissolve:{org_id}",
                    "p_kind": "clawback",
                    "p_metadata": {"org_id": org_id, "dissolved_by": user_id, "reason": "team_dissolved"},
                },
            ).execute()
    except Exception:
        logger.exception("dissolve_org: pool forfeit failed org_id=%s", org_id)

    # 2. Stop the meter (cancel_topup: Stripe first, columns second; a failed
    # cancel LEAVES topup_stripe_subscription_id on the row — the pointer
    # support needs to finish the cancellation by hand).
    #
    # LOAD-BEARING: the top-up subscription must carry Task 11's org_topup
    # metadata contract (kind='org_topup' and NO user_id). This delete fires
    # customer.subscription.deleted back at us, and stripe_events resolves that
    # webhook by metadata.user_id — a top-up that carried one would downgrade
    # the covering admin's PERSONAL plan to free the moment a team dissolves.
    try:
        if cancel_topup(db, org_id):
            analytics_capture(user_id, "org_topup_canceled", {"org_id": org_id, "trigger": "dissolve"})
    except Exception:
        logger.exception("dissolve_org: top-up cancel failed org_id=%s", org_id)

    # 3. Artists back to people. NOT best-effort: a team_id left pointing at a
    # dissolved org locks its creator out of their own subtree
    # (can_access_artist denies on archived_at), so a failure here must abort
    # the whole call with the org still intact and retryable.
    recipients = _dissolve_recipients(db, org_id, user_id)
    try:
        for r in recipients:
            db.table("artists").update({"team_id": None, "user_id": r["recipient"]}).eq("id", r["artist_id"]).execute()
    except Exception as exc:
        # Say what actually happened: the reverts already committed STAY
        # committed (each is its own statement), and re-running skips them
        # because they no longer match team_id=org_id.
        raise HTTPException(
            status_code=500,
            detail=(
                "Some artists were handed back but the team could not be dissolved. "
                "Nothing was lost — run dissolve again to finish."
            ),
        ) from exc

    # 3b. Boards back to people — same recipient rule and same "not
    # best-effort" reasoning as the artists above.
    try:
        _revert_org_boards(db, org_id, user_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Artists were handed back but boards could not be — nothing was lost; run dissolve again.",
        ) from exc

    # 4. Re-derive both sides' byte totals — same idiom (and same reason) as
    # the transfer path in orgs/artists.py: a stale total must not fail the op,
    # but an overstated personal total would wrongly block a real upload.
    distinct_recipients = dict.fromkeys(r["recipient"] for r in recipients)  # DISTINCT, order-stable
    recalcs = [("recalc_user_storage", {"p_user_id": uid}) for uid in distinct_recipients]
    recalcs.append(("recalc_team_storage", {"p_org_id": org_id}))
    for rpc, params in recalcs:
        try:
            db.rpc(rpc, params).execute()
        except Exception as exc:  # noqa: BLE001 - a stale total must not fail the dissolve
            print(f"dissolve_org: {rpc} failed org={org_id}: {exc}")

    # 5. Seats, invites, grants. SOFT throughout (rule 13).
    #
    # The dissolving admin's OWN seat is deliberately EXCLUDED: the
    # org_members_admin_guard trigger (20260721000001) rejects any update that
    # would leave a still-existing org with no active admin, and one rejected
    # row aborts the whole statement — including it would strand every other
    # member's seat. The surviving seat confers nothing once step 6 lands:
    # billing-context resolution, the team-slot count and the storage pool all
    # refuse an archived/dissolved org.
    # `.neq("status", "removed")` is about the FIRST run, not a retry: members
    # offboarded before the dissolve already carry the revoked_at that dated
    # their real removal, and that STORED timestamp is what the offboard
    # reclaim's request_id grammar (offboard:{member_id}:{epoch(revoked_at)})
    # keys off. Restamping it would mint a new idempotency key for money that
    # already moved, so a replay would stop converging.
    now = _now_iso()
    try:
        db.table("org_members").update({"status": "removed", "revoked_at": now}).eq("org_id", org_id).neq(
            "user_id", user_id
        ).neq("status", "removed").execute()
        db.table("pending_org_invites").update({"status": "expired"}).eq("org_id", org_id).eq(
            "status", "pending"
        ).execute()
    except Exception:
        logger.exception("dissolve_org: seat/invite teardown failed org_id=%s", org_id)
    # Same revocation archiving does (rule 12) — without it, ex-members keep
    # org-granted access to projects that now belong to an individual until the
    # sweep's next reconciliation pass.
    _teardown_archived_org_grants(db, org_id)

    # 6. Terminal state. archived_at is preserved if it was already set (an
    # org dissolved out of archive keeps its original archive date).
    patch = {"dissolved_at": now, "archived_at": org.get("archived_at") or now}
    res = db.table("organizations").update(patch).eq("id", org_id).execute()
    return res.data[0] if res.data else {"dissolved": org_id}


# ============================================================================
# Transfer credits (Task 10, spec §4.1) — the owner-requested funding inlet.
# An ACTIVE admin moves credits OUT OF THEIR OWN personal reserve and INTO
# this org's pool. This is the only inlet an org admin drives themselves;
# everything else (dispersal, packs, admin grants) is set by a Msanii admin
# or an operator-negotiated contract.
# ============================================================================


async def transfer_credits_to_pool(db: Client, user_id: str, org_id: str, amount: int) -> dict:
    """Move `amount` credits from the caller's personal RESERVE into org_id's
    pool via the `transfer_credits` RPC (20260816000002). Active admin,
    self_serve org only (an enterprise org's pool is filled by its contract
    and packs, never by a member's own credits) and refuses on an archived or
    dissolved org — same guard shape as dissolve_org/unarchive_org, reusing
    `_require_self_serve_org`.

    The RPC is reserve-only on the source side (bundle expires monthly;
    moving it would silently end that expiry the moment the credits land in
    a pool) and RAISEs rather than clamps when the reserve can't cover the
    amount. The accept/reject decision itself does NOT race: the RPC takes
    `FOR UPDATE` on the source wallet row and re-checks reserve_balance under
    that lock before raising, so two concurrent transfers can't both slip
    past the check. What's separately re-read (not the pre-call snapshot) is
    the `reserveBalance` figure in the 409 body below — that's a plain SELECT
    issued AFTER the RPC's transaction has already rolled back, so it can, in
    principle, still differ from the value the RPC itself rejected against if
    something else moves the balance in the gap between the RPC's rollback
    and this read.

    IDEMPOTENT: a request_id collision (retry of the same client action)
    returns `duplicate: true` from the RPC, treated here as a normal success
    — the transfer already landed, so a retry must read as "done", not fail.
    """
    authz.require_admin(db, user_id, org_id)
    org = _require_self_serve_org(db, org_id)
    _require_live_org(org)

    personal_wallet = wallets.read_or_create_user_wallet(db, user_id)
    pool_wallet = wallets.read_or_create_org_wallet(db, org_id)

    try:
        res = db.rpc(
            "transfer_credits",
            {
                "p_from_wallet": personal_wallet["id"],
                "p_to_wallet": pool_wallet["id"],
                "p_amount": amount,
                "p_request_id": f"xfer:{uuid4()}",
                "p_metadata": {"org_id": org_id, "admin_user_id": user_id},
            },
        ).execute()
    except Exception as exc:
        if _is_insufficient_reserve_error(exc):
            reserve = wallets.read_or_create_user_wallet(db, user_id).get("reserve_balance") or 0
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "You don't have enough reserve credits for this transfer.",
                    "reserveBalance": reserve,
                },
            ) from exc
        raise

    return res.data if isinstance(res.data, dict) else {}
