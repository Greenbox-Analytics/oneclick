# src/backend/subscriptions/sweep.py
"""Daily billing sweep (spec §3 clock, §5 storage, §7 annual overage).

Cloud Scheduler hits POST /internal/billing-sweep once a day with the
X-Sweep-Token header. Idempotent: every step no-ops on re-run. Lazy per-request
rollover stays the fast path; this catches inactive users and period-end
billing events that must fire without user activity.
"""

import hmac
import logging
import os
import uuid
from datetime import UTC, datetime

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Header, HTTPException

import subscriptions.stripe_client as stripe_client_module
from subscriptions.overage_billing import bill_pending_overage, invoice_unswept_items
from subscriptions.service import _parse_iso, credits_enabled, licensing_enabled

logger = logging.getLogger(__name__)
router = APIRouter(tags=["internal"])

PAID_TIERS = ("pro", "pro_max")

# v1 single-page cap on the unbounded scans below. Full pagination via the
# repo's pagination helper (pagination.py) is a follow-up; at current scale one
# page is plenty. If a scan hits the cap we log an error and press on rather
# than silently dropping the tail.
ROW_CAP = 10000

# Annual standalone invoices fire at most once per ~month per wallet. 27 days
# (< the shortest month) guarantees ≤1 invoice/month and ≤~1 month of unbilled
# liability, while tolerating the daily sweep landing a few hours early/late.
ANNUAL_INVOICE_MIN_DAYS = 27


def _require_token(token: str | None) -> None:
    expected = os.getenv("SWEEP_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Sweep not configured")
    # Timing-safe compare on a bearer secret (compare_digest short-circuits on
    # length, so pad the None case to an empty string first).
    if not hmac.compare_digest(token or "", expected):
        raise HTTPException(status_code=403, detail="Forbidden")


def _capped(builder, name: str) -> list:
    """Execute a select builder with a hard row cap; log if the cap is hit."""
    res = builder.limit(ROW_CAP).execute()
    data = res.data or []
    if len(data) >= ROW_CAP:
        logger.error("sweep: query %s hit the %d-row cap — some users skipped this run", name, ROW_CAP)
    return data


@router.post("/internal/billing-sweep")
async def billing_sweep(x_sweep_token: str | None = Header(None)):
    _require_token(x_sweep_token)
    if not credits_enabled():
        return {"walletsRolled": 0, "storageBilled": 0, "overageBilled": 0, "annualInvoiced": 0, "disabled": True}

    from main import get_supabase_client

    # NOTE: the supabase client is synchronous, so every call below blocks the
    # event loop. ACCEPTED for v1 — this is a once-daily scheduled endpoint off
    # the hot path at current (small) scale. Revisit (thread offload / async
    # client) if the user base or per-user work grows.
    sb = get_supabase_client()
    now = datetime.now(UTC)
    rolled = storage_billed = overage_billed = annual_invoiced = 0
    storage_grandfathered = seats_topped_up = pool_low = 0

    tier_data = _capped(
        sb.table("tier_entitlements").select("tier, monthly_credits, included_storage_bytes"),
        "tier_entitlements",
    )
    grants = {r["tier"]: r["monthly_credits"] for r in tier_data}
    included = {r["tier"]: r.get("included_storage_bytes", -1) for r in tier_data}
    paid_subs = _capped(
        sb.table("subscriptions")
        .select("user_id, tier, stripe_customer_id, stripe_price_id, storage_overage_enabled")
        .in_("tier", list(PAID_TIERS)),
        "subscriptions",
    )
    # Single lookup table for the per-user rows so steps 2 & 4 don't re-query
    # `subscriptions` once per wallet/user.
    subs_by_uid = {s["user_id"]: s for s in paid_subs}

    # Per-user monthly_credits overrides (VIPs; NOT the tester path, which uses
    # one-time reserve grants): the lazy path merges these in _merge(), so the
    # sweep must agree — rollover fires once per period, so a wrong grant here
    # sticks for a whole month. Expiry/revoked semantics mirror _read_override.
    override_rows = _capped(
        sb.table("tier_overrides").select("user_id, monthly_credits, reason, expires_at"),
        "tier_overrides",
    )
    override_grants: dict = {}
    for r in override_rows:
        if r.get("monthly_credits") is None or r.get("reason") == "tester_revoked":
            continue
        exp = _parse_iso(r.get("expires_at"))
        if exp is not None and exp < now:
            continue
        override_grants[r["user_id"]] = r["monthly_credits"]

    # Licensing rule 13 (storage-billing grandfather): storage is a single
    # per-user counter that follows a member out of an org seat — an
    # ex-seat member who accrued hundreds of GB under org context must never
    # be auto-billed for it in personal context (block-don't-bill; the wall
    # copy points at support, never "ask your admin"). Any org_members row —
    # ANY status, including 'removed' — exempts the user, because removal is
    # a SOFT-remove and the surviving row is exactly the durability marker
    # rule 13 relies on. Unconditional (not gated on licensing_enabled()):
    # once the org tables exist, historical org-accrued storage must stay
    # exempt even if the flag is later toggled off.
    try:
        org_member_rows = _capped(
            sb.table("org_members").select("user_id"),
            "org_members(storage-grandfather)",
        )
        org_member_user_ids = {r["user_id"] for r in org_member_rows if r.get("user_id")}
    except Exception as exc:
        # Expected pre-migration state (Phase B review finding 2): deploys are
        # automatic but the org_members migration
        # (20260721000001_licensing_core.sql) is applied manually, so a
        # backend deploy can land before the table exists. This scan is
        # unconditional (not gated on licensing_enabled()/CREDITS_ENABLED
        # beyond the early-return above) and was the ONLY step-level DB
        # access in this sweep with no try/except — an unguarded failure
        # here aborted the ENTIRE sweep (rollover, storage billing, overage
        # billing all silently stopped). Fail OPEN on an EMPTY grandfather
        # set: correct pre-migration, since nobody can have org history yet.
        # logger.warning (not .exception) — this is anticipated, not a bug.
        logger.warning(
            "sweep: org_members scan failed (expected if the licensing migration "
            "hasn't run yet) — proceeding with an empty storage-grandfather set: %s",
            exc,
        )
        org_member_user_ids: set[str] = set()

    # --- 1. Storage overage snapshot — BEFORE rollover, so it's keyed on the
    # period that is about to close. Semantics (spec §5 refined): a monthly
    # charge for current bytes above included, billed once per wallet period.
    rate = float(os.getenv("STORAGE_OVERAGE_USD_PER_GB", "0.05"))
    for sub in paid_subs:
        try:
            if not sub.get("storage_overage_enabled") or not sub.get("stripe_customer_id"):
                continue
            if sub["user_id"] in org_member_user_ids:
                storage_grandfathered += 1
                logger.info(
                    "sweep storage billing: user %s grandfathered (holds an org_members row) — never auto-billed",
                    sub["user_id"],
                )
                continue
            usage = sb.table("usage_counters").select("total_storage_bytes").eq("user_id", sub["user_id"]).execute()
            used = usage.data[0]["total_storage_bytes"] if usage.data else 0
            inc = included.get(sub["tier"], -1)
            if inc == -1 or used <= inc:
                continue
            wallet_res = (
                sb.table("credit_wallets")
                .select("id, period_end")
                .eq("owner_type", "user")
                .eq("owner_id", sub["user_id"])
                .execute()
            )
            if not wallet_res.data:
                continue
            wallet = wallet_res.data[0]
            # Idempotency: one storage_bill ledger row per wallet period. The
            # check counts ANY storage_bill row for the period (even one whose
            # invoice_item_id backfill never landed — see safe-direction note
            # below), so a crashed prior run skips the period rather than
            # retrying it into a double charge.
            existing = (
                sb.table("credit_ledger")
                .select("metadata")
                .eq("wallet_id", wallet["id"])
                .eq("kind", "storage_bill")
                .execute()
            )
            already = any(
                (r.get("metadata") or {}).get("period_end") == wallet["period_end"] for r in (existing.data or [])
            )
            if already:
                continue
            over_gb = (used - inc) / 1_073_741_824
            amount_cents = round(over_gb * rate * 100)
            if amount_cents <= 0:
                continue
            # Safe-direction ordering (spec §5): INSERT the ledger row first,
            # THEN create the Stripe InvoiceItem, THEN backfill invoice_item_id.
            # A crash between create and insert — combined with cron drift past
            # Stripe's 24h idempotency-key window — would double-charge. Insert-
            # first turns that rare race into a rare UNDER-charge instead: the
            # `already` check above sees the row and skips the period next run.
            ledger_id = str(uuid.uuid4())
            base_metadata = {
                "period_end": wallet["period_end"],
                "gb": round(over_gb, 2),
                "usd": amount_cents / 100,
            }
            sb.table("credit_ledger").insert(
                {
                    "id": ledger_id,
                    "wallet_id": wallet["id"],
                    "delta": 0,
                    "kind": "storage_bill",
                    "balance_after": 0,
                    "metadata": base_metadata,
                }
            ).execute()
            stripe = stripe_client_module.get_stripe()
            # idempotency_key defends the check-then-act gap if two sweeps race.
            item = stripe.InvoiceItem.create(
                idempotency_key=f"storage:{wallet['id']}:{wallet['period_end']}",
                customer=sub["stripe_customer_id"],
                amount=amount_cents,
                currency="usd",
                description=f"Storage overage: {over_gb:.1f} GB × ${rate:.2f}/GB/mo",
                metadata={"user_id": sub["user_id"], "period_end": wallet["period_end"]},
            )
            sb.table("credit_ledger").update({"metadata": {**base_metadata, "invoice_item_id": item.id}}).eq(
                "id", ledger_id
            ).execute()
            storage_billed += 1
        except Exception:
            logger.exception("sweep storage billing failed user=%s", sub.get("user_id"))

    # --- 2. Roll over stale wallets at their tier's grant --------------------
    stale = _capped(
        sb.table("credit_wallets").select("*").eq("owner_type", "user").lt("period_end", now.isoformat()),
        "credit_wallets(stale)",
    )
    for wallet in stale:
        try:
            tier = subs_by_uid.get(wallet["owner_id"], {}).get("tier", "free")
            new_end = datetime.fromisoformat(wallet["period_end"])
            while new_end < now:
                new_end = new_end + relativedelta(months=1)
            res = sb.rpc(
                "rollover_wallet",
                {
                    "p_wallet_id": wallet["id"],
                    "p_monthly_grant": override_grants.get(wallet["owner_id"], grants.get(tier, 0)),
                    "p_new_period_start": (new_end - relativedelta(months=1)).isoformat(),
                    "p_new_period_end": new_end.isoformat(),
                },
            ).execute()
            if res.data:
                rolled += 1
        except Exception:
            logger.exception("sweep rollover failed wallet=%s", wallet.get("id"))

    # --- 3. Bill unbilled overage rows for ALL paid users (daily, cheap) ----
    # Creates pending InvoiceItems only. Monthly plans: items ride the next
    # renewal invoice automatically. This is the ONLY place (besides the
    # invoice.created safety net) that talks to Stripe about credit overage —
    # never the request path.
    for sub in paid_subs:
        try:
            if not sub.get("stripe_customer_id"):
                continue
            overage_billed += bill_pending_overage(sb, sub["user_id"])
        except Exception:
            logger.exception("sweep overage billing failed user=%s", sub.get("user_id"))

    # --- 4. Annual plans: standalone invoice on a MONTHLY cadence ------------
    # CRITICAL: gate on a per-wallet cadence timestamp (last_standalone_invoice_at),
    # NOT on "did THIS sweep roll the wallet". The lazy get_for_user path
    # (_maybe_rollover_wallet in service.py) advances the period on the user's
    # first read each month, so for any ACTIVE annual user the sweep's rollover
    # RPC returns false — they'd never appear in a "rolled this sweep" set and
    # their overage would sit unbilled until the ~12-month Stripe renewal
    # (violating spec §7's ≤1-month unbilled-liability guarantee). Gating on the
    # cadence timestamp makes the invoice fire monthly regardless of which path
    # advanced the wallet. auto_advance pulls EVERY floating pending item into
    # the one invoice, so this sweeps in both credit overage (overage_debit) and
    # storage (storage_bill) — the latter also otherwise starves on annual plans.
    cadence_floor = now - relativedelta(days=ANNUAL_INVOICE_MIN_DAYS)
    for sub in paid_subs:
        try:
            is_annual = bool(sub.get("stripe_price_id") and "annual" in sub["stripe_price_id"])
            if not is_annual or not sub.get("stripe_customer_id"):
                continue
            wallet_res = (
                sb.table("credit_wallets")
                .select("id, last_standalone_invoice_at")
                .eq("owner_type", "user")
                .eq("owner_id", sub["user_id"])
                .execute()
            )
            if not wallet_res.data:
                continue
            wallet = wallet_res.data[0]
            last_at = _parse_iso(wallet.get("last_standalone_invoice_at"))
            if last_at is not None and last_at > cadence_floor:
                continue  # already invoiced within the cadence window this month
            result = invoice_unswept_items(
                sb,
                wallet["id"],
                sub["stripe_customer_id"],
                idempotency_key=f"annual:{wallet['id']}:{now.date().isoformat()}",
            )
            # Cadence recorded whenever rows were handled — including the
            # consumed-elsewhere case — so a doomed invoice never retries daily.
            if result["stamped"]:
                sb.table("credit_wallets").update({"last_standalone_invoice_at": now.isoformat()}).eq(
                    "id", wallet["id"]
                ).execute()
            if result["invoiced"]:
                annual_invoiced += 1
        except Exception:
            logger.exception("sweep annual invoicing failed user=%s", sub.get("user_id"))

    # --- 5. Default seat allowance (licensing Phase B, spec §4 flow, rule 6):
    # for every ACTIVE, non-archived org with default_seat_allowance > 0, top
    # each ACTIVE seat up to the allowance — FULL amount or SKIP. LICENSING_
    # ENABLED only; the credits-disabled early-return above already covers
    # the credits gate, so this step only needs its own flag check.
    if licensing_enabled():
        month_key = now.strftime("%Y-%m")
        orgs = _capped(
            sb.table("organizations")
            .select("id, default_seat_allowance")
            .eq("status", "active")
            .is_("archived_at", "null")
            .gt("default_seat_allowance", 0),
            "organizations(allowance)",
        )
        for org in orgs:
            org_id = org["id"]
            allowance = org.get("default_seat_allowance") or 0
            if allowance <= 0:
                continue  # manual-only org (NULL/0) — defensive, query already filters this

            try:
                pool_rows = (
                    sb.table("credit_wallets")
                    .select("id, reserve_balance")
                    .eq("owner_type", "org")
                    .eq("owner_id", org_id)
                    .execute()
                    .data
                    or []
                )
            except Exception:
                logger.exception("sweep allowance: pool wallet read failed org=%s", org_id)
                continue
            if not pool_rows:
                # No purchases yet -> no pool wallet. Not an error; there is
                # nothing to allocate from.
                logger.info("sweep allowance: org %s has no pool wallet yet — skipping", org_id)
                continue
            pool_wallet_id = pool_rows[0]["id"]
            # Tracked LOCALLY and decremented after each successful transfer so
            # one sweep run can't overdraw the pool off a stale read across
            # several seats (rule: track in-loop pool balance locally).
            pool_reserve = pool_rows[0].get("reserve_balance") or 0

            members = _capped(
                sb.table("org_members").select("id").eq("org_id", org_id).eq("status", "active"),
                "org_members(allowance)",
            )
            for member in members:
                member_id = member["id"]
                try:
                    seat_rows = (
                        sb.table("credit_wallets")
                        .select("id, bundle_balance, reserve_balance")
                        .eq("owner_type", "seat")
                        .eq("owner_id", member_id)
                        .execute()
                        .data
                        or []
                    )
                    if not seat_rows:
                        # Wallet creation is the app layer's job (lazy on first
                        # org-context read/allocation), not the sweep's — the
                        # next org-context read creates it and next month's
                        # sweep tops it up.
                        logger.info(
                            "sweep allowance: seat wallet missing for member %s — skipping this month", member_id
                        )
                        continue
                    seat_wallet = seat_rows[0]
                    seat_balance = (seat_wallet.get("bundle_balance") or 0) + (seat_wallet.get("reserve_balance") or 0)
                    top_up = allowance - seat_balance
                    if top_up <= 0:
                        continue  # already at/above allowance — no RPC call (money RPCs raise on non-positive)

                    if pool_reserve < top_up:
                        pool_low += 1
                        logger.info(
                            "sweep allowance: pool for org %s can't cover top-up for member %s (have %d, need %d)",
                            org_id,
                            member_id,
                            pool_reserve,
                            top_up,
                        )
                        continue  # skip WITHOUT consuming the month key (rule 6)

                    request_id = f"allowance:{member_id}:{month_key}"
                    res = sb.rpc(
                        "transfer_credits",
                        {
                            "p_from_wallet": pool_wallet_id,
                            "p_to_wallet": seat_wallet["id"],
                            "p_amount": top_up,
                            "p_kind": "allocation",
                            "p_request_id": request_id,
                            "p_metadata": {"org_id": org_id, "source": "allowance"},
                        },
                    ).execute()
                    if (res.data or {}).get("duplicate"):
                        continue  # already topped up this month — no-op
                    seats_topped_up += 1
                    pool_reserve -= top_up
                except Exception:
                    logger.exception("sweep allowance failed org=%s member=%s", org_id, member_id)

    return {
        "walletsRolled": rolled,
        "storageBilled": storage_billed,
        "storageGrandfathered": storage_grandfathered,
        "overageBilled": overage_billed,
        "annualInvoiced": annual_invoiced,
        "seatsToppedUp": seats_topped_up,
        "poolLow": pool_low,
    }
