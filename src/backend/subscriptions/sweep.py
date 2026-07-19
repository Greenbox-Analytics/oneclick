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
from subscriptions.overage_billing import bill_pending_overage
from subscriptions.service import _parse_iso, credits_enabled

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

    # --- 1. Storage overage snapshot — BEFORE rollover, so it's keyed on the
    # period that is about to close. Semantics (spec §5 refined): a monthly
    # charge for current bytes above included, billed once per wallet period.
    rate = float(os.getenv("STORAGE_OVERAGE_USD_PER_GB", "0.05"))
    for sub in paid_subs:
        try:
            if not sub.get("storage_overage_enabled") or not sub.get("stripe_customer_id"):
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
                    "p_monthly_grant": grants.get(tier, 0),
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
            rows_res = (
                sb.table("credit_ledger")
                .select("id, kind, metadata")
                .eq("wallet_id", wallet["id"])
                .in_("kind", ["overage_debit", "storage_bill"])
                .execute()
            )
            # Unswept billable items: overage_debit rows already turned into a
            # pending InvoiceItem (invoice_item_id set) PLUS storage_bill rows.
            # The `swept` stamp — not invoice_item_id, which persists forever —
            # is what stops us re-invoicing the same items every month, so an
            # empty invoice (invoice_no_customer_line_items) never fires.
            unswept = []
            for r in rows_res.data or []:
                meta = r.get("metadata") or {}
                if meta.get("swept"):
                    continue
                # Both kinds require a backfilled invoice_item_id — it proves a
                # real pending Stripe item exists. A storage_bill row whose
                # step-1 InvoiceItem.create never landed (insert-first crash)
                # would otherwise trigger an empty Invoice.create
                # (invoice_no_customer_line_items) that retries every daily
                # sweep and sticks the cadence until the next period.
                if meta.get("invoice_item_id"):
                    unswept.append(r)
            if not unswept:
                continue
            stripe = stripe_client_module.get_stripe()
            stripe.Invoice.create(customer=sub["stripe_customer_id"], auto_advance=True)
            # Stamp rows swept, then record the cadence timestamp. Both happen
            # only after the invoice succeeds, so a failed create retries next
            # sweep (last_standalone_invoice_at stays put).
            for r in unswept:
                sb.table("credit_ledger").update({"metadata": {**(r.get("metadata") or {}), "swept": True}}).eq(
                    "id", r["id"]
                ).execute()
            sb.table("credit_wallets").update({"last_standalone_invoice_at": now.isoformat()}).eq(
                "id", wallet["id"]
            ).execute()
            annual_invoiced += 1
        except Exception:
            logger.exception("sweep annual invoicing failed user=%s", sub.get("user_id"))

    return {
        "walletsRolled": rolled,
        "storageBilled": storage_billed,
        "overageBilled": overage_billed,
        "annualInvoiced": annual_invoiced,
    }
