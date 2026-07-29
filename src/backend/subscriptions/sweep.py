# src/backend/subscriptions/sweep.py
"""Daily billing sweep (spec §3 clock, §7 annual overage).

Cloud Scheduler hits POST /internal/billing-sweep once a day with the
X-Sweep-Token header. Idempotent: every step no-ops on re-run. Lazy per-request
rollover stays the fast path; this catches inactive users and period-end
billing events that must fire without user activity.
"""

import hmac
import logging
import os
from datetime import UTC, datetime

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Header, HTTPException

from subscriptions.overage_billing import bill_pending_overage, invoice_unswept_items
from subscriptions.service import _parse_iso, credits_enabled, licensing_enabled

logger = logging.getLogger(__name__)
router = APIRouter(tags=["internal"])

PAID_TIERS = ("basic", "pro")

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
        return {"walletsRolled": 0, "overageBilled": 0, "annualInvoiced": 0, "disabled": True}

    from main import get_supabase_client

    # NOTE: the supabase client is synchronous, so every call below blocks the
    # event loop. ACCEPTED for v1 — this is a once-daily scheduled endpoint off
    # the hot path at current (small) scale. Revisit (thread offload / async
    # client) if the user base or per-user work grows.
    sb = get_supabase_client()
    now = datetime.now(UTC)
    rolled = overage_billed = annual_invoiced = 0
    seats_topped_up = pool_low = 0

    tier_data = _capped(
        sb.table("tier_entitlements").select("tier, monthly_credits"),
        "tier_entitlements",
    )
    grants = {r["tier"]: r["monthly_credits"] for r in tier_data}
    paid_subs = _capped(
        sb.table("subscriptions")
        .select("user_id, tier, stripe_customer_id, stripe_price_id")
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

    # --- 1. Roll over stale wallets at their tier's grant --------------------
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

    # --- 2. Bill unbilled overage rows for ALL paid users (daily, cheap) ----
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

    # --- 3. Annual plans: standalone invoice on a MONTHLY cadence ------------
    # CRITICAL: gate on a per-wallet cadence timestamp (last_standalone_invoice_at),
    # NOT on "did THIS sweep roll the wallet". The lazy get_for_user path
    # (_maybe_rollover_wallet in service.py) advances the period on the user's
    # first read each month, so for any ACTIVE annual user the sweep's rollover
    # RPC returns false — they'd never appear in a "rolled this sweep" set and
    # their overage would sit unbilled until the ~12-month Stripe renewal
    # (violating spec §7's ≤1-month unbilled-liability guarantee). Gating on the
    # cadence timestamp makes the invoice fire monthly regardless of which path
    # advanced the wallet. auto_advance pulls EVERY floating pending item into
    # the one invoice, so credit overage (overage_debit) can't starve on an
    # annual plan waiting for its ~12-month renewal.
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

    # --- 4. Default seat allowance (licensing Phase B, spec §4 flow, rule 6):
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
        "overageBilled": overage_billed,
        "annualInvoiced": annual_invoiced,
        "seatsToppedUp": seats_topped_up,
        "poolLow": pool_low,
    }
