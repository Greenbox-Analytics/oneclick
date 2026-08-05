"""Org pool wallet helper (Licensing Phase B).

ONE wallet per organization: `credit_wallets` with owner_type='org',
owner_id=organizations.id, introduced by
supabase/migrations/20260721000001_licensing_core.sql. Members hold no wallet —
they spend from this pool against a monthly cap enforced inside `debit_credits`.

The pool carries BOTH buckets, with different lifetimes:
  - `bundle_balance` — the monthly contract dispersal. EXPIRES at each period
    end (the sweep rolls it), which is what stops an org banking a year of
    credits and burning them in one month.
  - `reserve_balance` — purchased packs. Never expires.
`debit_credits` drains bundle before reserve, so the expiring money is always
spent first.

LOUD WARNING — do NOT reuse `subscriptions.service.EntitlementsService.
_read_or_create_wallet` for the org pool, and do NOT copy its upsert pattern
here. That helper seeds `period_end=now()` so the caller's next
`_maybe_rollover_wallet` fires immediately and grants the *tier's*
`monthly_credits` — on an org pool that would overwrite the dispersal with a
personal-plan grant. The pool's period is written by the dispersal sweep and
nothing else.

INSERT, not upsert: an upsert's `ON CONFLICT ... DO UPDATE` could reset
bundle_balance/reserve_balance back to their insert defaults if two
create-on-miss callers race — a plain INSERT either wins (first writer) or
raises a unique_violation (second writer), which is caught below and treated
as "someone else already created it, re-read what they wrote."
"""

import os

from supabase import Client

# Ledger kinds that represent the org PAYING us for credits. All count toward
# the activation floor: a contract dispersal is money in exactly as much as a
# one-time pack is, and an org whose only funding is its monthly dispersal would
# otherwise sit pending forever with its seats conferring nothing.
#
# 'monthly_grant' is here because the dispersal sweep is implemented via
# rollover_wallet, which writes kind='monthly_grant' — and on ORG wallets the
# sweep is the ONLY monthly_grant writer, so on this helper's (org-only)
# wallets it is exactly the dispersal component. Kind 'dispersal' is kept in
# the list for completeness but nothing has ever written it: no code path
# emits that kind (rollover_wallet emits 'monthly_grant' instead).
PAID_IN_KINDS = ("purchase", "dispersal", "monthly_grant")


def _read_wallet(sb: Client, owner_type: str, owner_id: str) -> dict | None:
    res = sb.table("credit_wallets").select("*").eq("owner_type", owner_type).eq("owner_id", owner_id).execute()
    rows = res.data or []
    return rows[0] if rows else None


def read_or_create_org_wallet(sb: Client, org_id: str) -> dict:
    """The org's pool wallet: owner_type='org', owner_id=org_id. Create-on-miss.

    NEVER call this with a user id — user wallets are seeded exclusively by
    `EntitlementsService._read_or_create_wallet`, which arms a rollover-triggering
    tier grant the pool must never receive (see module docstring).
    """
    existing = _read_wallet(sb, "org", org_id)
    if existing:
        return existing

    try:
        # No period fields: the dispersal sweep writes the pool's first period
        # when it makes the first top-up. bundle_balance / reserve_balance /
        # overage_this_period all default to 0 at the DB.
        inserted = sb.table("credit_wallets").insert({"owner_type": "org", "owner_id": org_id}).execute()
    except Exception:
        # Duplicate-race: another create-on-miss caller's INSERT won the
        # UNIQUE (owner_type, owner_id) constraint between our SELECT and our
        # INSERT. The wallet exists now — re-select rather than raise.
        row = _read_wallet(sb, "org", org_id)
        if row:
            return row
        raise
    else:
        rows = inserted.data or []
        if rows:
            return rows[0]
        # Some client/mocking configurations don't echo the inserted row even
        # though it landed — fall back to a re-select before giving up.
        row = _read_wallet(sb, "org", org_id)
        if row:
            return row
        raise RuntimeError(f"failed to read or create org wallet for org_id={org_id}")


def cumulative_paid_in(sb: Client, org_wallet_id: str) -> int:
    """SUM of paid-in ledger deltas on an org's pool wallet (see PAID_IN_KINDS).

    Shared by the callers that must agree DEFINITIONALLY on "how much has this
    org paid us":
      - `subscriptions.stripe_events._handle_org_topup_grant`'s activation check
        (has cumulative payment crossed the pending -> active floor)
      - the dispersal sweep, which activates an org once its contract has
        dispersed enough
      - `subscriptions.admin_service.get_org_pool`'s support-visibility snapshot
    Extracted here specifically so none of them can drift from the others.
    """
    res = (
        sb.table("credit_ledger")
        .select("delta, kind")
        .eq("wallet_id", org_wallet_id)
        .in_("kind", list(PAID_IN_KINDS))
        .execute()
    )
    return sum(row.get("delta", 0) for row in (res.data or []))


def maybe_activate_org(sb: Client, org_id: str, org_wallet_id: str) -> bool:
    """Flip a PENDING org to 'active' once its cumulative paid-in credits cross
    the activation floor (the org's own `min_initial_purchase_credits`, else the
    ENTERPRISE_MIN_INITIAL_CREDITS env default). The ONE implementation shared by
    both activation paths — `stripe_events._handle_org_topup_grant` (pack
    purchase) and the dispersal sweep (`subscriptions/sweep.py` step 4) — so
    "has this org paid enough to activate" can never drift between them.

    Activation only ever moves pending -> active: already-active / suspended /
    archived orgs are never touched (that would be a status regression, not an
    activation). Idempotent — re-running after activation is a no-op. Returns
    True only when this call performed the flip.
    """
    org_res = sb.table("organizations").select("status, min_initial_purchase_credits").eq("id", org_id).execute()
    org_row = org_res.data[0] if org_res.data else None
    if not org_row or org_row.get("status") != "pending":
        return False

    total_paid_in = cumulative_paid_in(sb, org_wallet_id)
    effective_min = org_row.get("min_initial_purchase_credits") or int(
        os.getenv("ENTERPRISE_MIN_INITIAL_CREDITS", "10000")
    )
    if total_paid_in >= effective_min:
        sb.table("organizations").update({"status": "active"}).eq("id", org_id).execute()
        return True
    return False
