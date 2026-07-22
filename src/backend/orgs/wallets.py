"""Seat/org wallet helpers (Licensing Phase B, Task 4).

Dedicated create-on-miss helpers for the two NEW `credit_wallets.owner_type`
values introduced by supabase/migrations/20260721000001_licensing_core.sql:
'org' (a pool wallet, owner_id=organizations.id) and 'seat' (a per-member
wallet, owner_id=org_members.id). Both are NULL-period, reserve-only, BY
CONSTRUCTION (spec rule 1): period_start/period_end are never written here —
that is the structural rollover exemption. `rollover_wallet` also carries a
DB-level RAISE on any non-'user' wallet (belt-and-suspenders), but these
helpers are the first line of defense: they simply never produce a wallet
with a period to roll.

LOUD WARNING — do NOT reuse `subscriptions.service.EntitlementsService.
_read_or_create_wallet` for seat/org wallets, and do NOT copy its upsert
pattern here. That helper deliberately seeds `period_end=now()` so the
caller's very next `_maybe_rollover_wallet` call fires immediately and grants
the tier's `monthly_credits` — exactly the rollover behavior rule 1 forbids
for seat/org wallets. For reserve-only money that must never expire, that
"seed grant" would zero out and silently overwrite real allocated/purchased
credits with a bogus zero-credit monthly grant the first time anything reads
the wallet. These helpers exist specifically so that code path is never
reachable for 'org'/'seat' wallets.

INSERT, not upsert: an upsert's `ON CONFLICT ... DO UPDATE` could reset
bundle_balance/reserve_balance back to their insert defaults if two
create-on-miss callers race — a plain INSERT either wins (first writer) or
raises a unique_violation (second writer), which is caught below and treated
as "someone else already created it, re-read what they wrote."
"""

from supabase import Client


def _read_wallet(sb: Client, owner_type: str, owner_id: str) -> dict | None:
    res = sb.table("credit_wallets").select("*").eq("owner_type", owner_type).eq("owner_id", owner_id).execute()
    rows = res.data or []
    return rows[0] if rows else None


def _read_or_create_wallet(sb: Client, owner_type: str, owner_id: str) -> dict:
    """Shared implementation behind the two public helpers below.

    NEVER call this (or the public wrappers) with owner_type='user' — user
    wallets are seeded exclusively by
    `EntitlementsService._read_or_create_wallet`, which deliberately arms a
    rollover-triggering grant these NULL-period wallets must never receive
    (see module docstring).
    """
    existing = _read_wallet(sb, owner_type, owner_id)
    if existing:
        return existing

    try:
        # NO period fields, ever — see module docstring (rule 1). bundle_balance
        # / reserve_balance / overage_this_period all default to 0 at the DB;
        # period_start/period_end default to NULL.
        inserted = sb.table("credit_wallets").insert({"owner_type": owner_type, "owner_id": owner_id}).execute()
    except Exception:
        # Duplicate-race: another create-on-miss caller's INSERT won the
        # UNIQUE (owner_type, owner_id) constraint between our SELECT and our
        # INSERT. The wallet exists now — re-select rather than raise.
        row = _read_wallet(sb, owner_type, owner_id)
        if row:
            return row
        raise
    else:
        rows = inserted.data or []
        if rows:
            return rows[0]
        # Some client/mocking configurations don't echo the inserted row even
        # though it landed — fall back to a re-select before giving up.
        row = _read_wallet(sb, owner_type, owner_id)
        if row:
            return row
        raise RuntimeError(f"failed to read or create {owner_type} wallet for owner_id={owner_id}")


def read_or_create_org_wallet(sb: Client, org_id: str) -> dict:
    """The org's pool wallet: owner_type='org', owner_id=org_id. NULL periods
    forever (rule 1). Create-on-miss."""
    return _read_or_create_wallet(sb, "org", org_id)


def read_or_create_seat_wallet(sb: Client, org_member_id: str) -> dict:
    """A member's seat wallet: owner_type='seat', owner_id=org_members.id.
    NULL periods forever (rule 1). Create-on-miss."""
    return _read_or_create_wallet(sb, "seat", org_member_id)


def cumulative_purchased(sb: Client, org_wallet_id: str) -> int:
    """SUM of 'purchase'-kind ledger deltas on an org's pool wallet.

    Shared by two callers that must agree DEFINITIONALLY on "how much has
    this org ever bought" (follow-ups plan Task 2, review round 2):
      - `subscriptions.stripe_events._handle_org_topup_grant`'s activation
        check (does cumulative purchase cross the pending -> active floor)
      - `subscriptions.admin_service.get_org_pool`'s support-visibility
        snapshot (does cumulative purchase cross the floor, for a support
        agent deciding how to dispose of the pool)
    Extracted here specifically so neither can drift from the other.
    """
    res = sb.table("credit_ledger").select("delta").eq("wallet_id", org_wallet_id).eq("kind", "purchase").execute()
    return sum(row.get("delta", 0) for row in (res.data or []))
