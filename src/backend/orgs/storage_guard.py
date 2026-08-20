"""Per-owner team storage pool checks (spec 2026-08-15 §5). pool_state and
reactivation_allowed are Task 5's; Task 12 adds the two other byte inlets on
top — upload_allowed (uploads) and artist_subtree_bytes (artist transfer).
Task 13 adds the PAYG sweep step — bill_team_storage_overage — that turns
pool overage into a Stripe InvoiceItem on the owner's PERSONAL customer.
"""

import logging
import math
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# v1 single-page cap on the owner scan below, mirroring orgs/standing.py's own
# ROW_CAP norm (that module's org scan uses the same log-and-press-on shape).
# Not imported from there / subscriptions.sweep's _capped to avoid coupling
# this module's one query to either helper's signature — if the cap is ever
# hit we log an error and press on rather than silently dropping the tail.
ROW_CAP = 10000

# One copy, shared by upload_allowed and subscriptions/service.py's
# _team_storage self-serve branch — the member sees this exact string
# regardless of which call site denied them.
TEAM_STORAGE_FULL_MSG = "Your team's storage is full — ask your admin to free up space or upgrade."


def pool_state(db, owner_id: str):
    """(used_bytes, pool_bytes, is_pro_like) — used sums storage_bytes over
    self_serve orgs covered_by=owner incl. archived/lapsed, EXCLUDING dissolved
    (anti archive-cycling, spec §5)."""
    from orgs.standing import team_dials_for_user

    rows = (
        db.table("organizations")
        .select("storage_bytes")
        .eq("kind", "self_serve")
        .eq("covered_by", owner_id)
        .is_("dissolved_at", "null")
        .execute()
    ).data or []
    used = sum(r.get("storage_bytes") or 0 for r in rows)
    dials = team_dials_for_user(db, owner_id)
    # pro-like = the Pro tier (Msanii admins resolve as pro): overflow goes to
    # PAYG. Keyed on the resolved tier, NOT the pool size — Basic and Pro pools
    # are 100/250 GiB (20260817000001), so size can't tell them apart.
    is_pro_like = dials.tier == "pro"
    return used, dials.team_storage_bytes, is_pro_like


def reactivation_allowed(db, owner_id: str) -> bool:
    """A lapsed/archived team may not wake up while the owner is over-pool
    (spec §3 reactivation guard). Pro-like owners pass — PAYG absorbs."""
    used, pool, is_pro_like = pool_state(db, owner_id)
    return is_pro_like or used <= pool


def upload_allowed(db, owner_id: str, size: int) -> tuple[bool, str | None]:
    """Gate an upload against the owner's self-serve pool (spec §5 byte
    inlet #1). Pro-like owners always pass — PAYG absorbs the overage, same
    as reactivation_allowed. Everyone else must fit: used + size <= pool."""
    used, pool, is_pro_like = pool_state(db, owner_id)
    if is_pro_like or used + size <= pool:
        return True, None
    return False, TEAM_STORAGE_FULL_MSG


def artist_subtree_bytes(db, artist_id: str) -> int:
    """Bytes an artist's subtree would add to a pool (spec §5 byte inlet #2:
    artist transfer). Mirrors recalc_team_storage's two joins
    (20260803000003_team_storage.sql) but scoped to ONE artist instead of a
    whole team, and read-only — sizing a transfer before it happens, not
    repairing a cached total."""
    pf_rows = (
        db.table("project_files")
        .select("file_size, projects!inner(artist_id)")
        .eq("projects.artist_id", artist_id)
        .execute()
        .data
        or []
    )
    af_rows = (
        db.table("audio_files")
        .select("file_size, audio_folders!inner(artist_id)")
        .eq("audio_folders.artist_id", artist_id)
        .execute()
        .data
        or []
    )
    return sum(r.get("file_size") or 0 for r in pf_rows + af_rows)


def bill_team_storage_overage(sb) -> int:
    """Sweep step (spec §5 PAYG, review finding 2): once per owner-period,
    turn pool overage into a Stripe InvoiceItem on the owner's PERSONAL
    customer (`bill_pending_overage` is the pattern — Stripe as the only
    record, no local ledger row).

    One row per DISTINCT `covered_by` owner across live (non-dissolved)
    self_serve orgs — pool_state already sums bytes across every org that
    owner covers, so per-org billing would double-count; billing is per
    OWNER, exactly like the pool itself. The owner scan is ONE query;
    pool_state's own per-owner query does the real per-owner filtering.

    Returns the count of owners actually invoiced (stamp-only skips don't
    count — they're not billing events).
    """
    org_rows = (
        sb.table("organizations")
        .select("covered_by")
        .eq("kind", "self_serve")
        .is_("dissolved_at", "null")
        .limit(ROW_CAP)
        .execute()
    ).data or []
    if len(org_rows) >= ROW_CAP:
        logger.error("bill_team_storage_overage: org scan hit the %d-row cap — some owners skipped this run", ROW_CAP)
    owner_ids = sorted({r["covered_by"] for r in org_rows if r.get("covered_by")})

    billed = 0
    for owner_id in owner_ids:
        try:
            if _bill_one_owner(sb, owner_id):
                billed += 1
        except Exception:
            logger.exception("bill_team_storage_overage failed owner=%s", owner_id)
    return billed


def _stamp(sb, owner_id: str, period_start: str) -> None:
    """Mark this owner-period handled — stamping IS the once-per-period
    idempotency for _bill_one_owner."""
    sb.table("subscriptions").update({"last_team_storage_invoiced_period": period_start}).eq(
        "user_id", owner_id
    ).execute()


def _bill_one_owner(sb, owner_id: str) -> bool:
    """Bill (or stamp-skip) one owner's period. Returns True iff an
    InvoiceItem was actually created for them this call."""
    used, pool, is_pro_like = pool_state(sb, owner_id)
    if not is_pro_like:
        return False  # basic/free owners have no PAYG lane — the pool is a hard cap for them

    sub_rows = (
        sb.table("subscriptions")
        .select("stripe_customer_id, last_team_storage_invoiced_period")
        .eq("user_id", owner_id)
        .execute()
    ).data
    if not sub_rows:
        return False  # no subscriptions row — nothing to stamp or bill against
    sub_row = sub_rows[0]

    wallet_rows = (
        sb.table("credit_wallets").select("period_start").eq("owner_type", "user").eq("owner_id", owner_id).execute()
    ).data
    period_start = wallet_rows[0].get("period_start") if wallet_rows else None
    if not period_start:
        return False  # no personal wallet — nothing to key the once-per-period stamp on

    # Parsed comparison, not raw-string (mirrors the "never raw-string sort"
    # rule used elsewhere for TIMESTAMPTZ columns) — bill only when the
    # wallet has rolled into a period strictly newer than the last stamp.
    stamp_raw = sub_row.get("last_team_storage_invoiced_period")
    stamp = datetime.fromisoformat(stamp_raw) if stamp_raw else None
    period = datetime.fromisoformat(period_start)
    if stamp is not None and stamp >= period:
        return False  # already handled this owner-period

    overage_gb = math.ceil(max(0, used - pool) / 2**30)
    if overage_gb <= 0:
        # Zero overage still stamps the period — without it a fully-covered
        # owner gets re-evaluated by every single sweep, forever.
        _stamp(sb, owner_id, period_start)
        return False

    customer = sub_row.get("stripe_customer_id")
    if not customer:
        # Msanii admins resolve pro-like via team_dials_for_user but rarely
        # carry a real Stripe customer — stamp so they don't re-evaluate
        # forever, but never attempt to invoice a nonexistent customer.
        logger.warning(
            "bill_team_storage_overage: owner %s is pro-like and over-pool but has no stripe customer "
            "(admin?) — stamping without billing",
            owner_id,
        )
        _stamp(sb, owner_id, period_start)
        return False

    amount_cents = round(overage_gb * float(os.getenv("TEAM_STORAGE_OVERAGE_USD_PER_GB", "0.025")) * 100)
    import subscriptions.stripe_client as stripe_client_module

    stripe = stripe_client_module.get_stripe()
    try:
        # No `invoice` kwarg → a floating PENDING item, same as
        # bill_overage_row's default: it folds into the owner's next
        # subscription invoice automatically.
        stripe.InvoiceItem.create(
            customer=customer,
            amount=amount_cents,
            currency="usd",
            description=f"Team storage: {overage_gb} GB over your included 100 GB",
            idempotency_key=f"teamstorage:{owner_id}:{period_start}",
            # Metadata parity with bill_overage_row's credit-overage InvoiceItem
            # (owner identifier + a key that pins this to one specific charge
            # instance, there ledger_row_id, here owner_id+period_start).
            metadata={"owner_id": owner_id, "period_start": period_start},
        )
    except Exception:
        # Stripe failure → do NOT stamp; retried next sweep (check-then-act,
        # same shape as bill_overage_row — no partial state to unwind).
        logger.exception(
            "bill_team_storage_overage: Stripe InvoiceItem.create failed owner=%s — not stamped, retried next sweep",
            owner_id,
        )
        return False

    _stamp(sb, owner_id, period_start)

    try:
        from analytics import capture as analytics_capture

        analytics_capture(owner_id, "team_storage_overage_billed", {"gb": overage_gb, "cents": amount_cents})
    except Exception:
        pass  # analytics must never break billing

    return True
