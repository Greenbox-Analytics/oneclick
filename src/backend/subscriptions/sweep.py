# src/backend/subscriptions/sweep.py
"""Daily billing sweep (spec §3 clock, §7 annual overage, org dispersal).

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
    rolled = overage_billed = annual_invoiced = dispersed = 0

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

    # --- 4. Monthly credit dispersal (licensing Phase B): for every ACTIVE,
    # non-archived org on a contract, roll the POOL wallet — which zeroes last
    # month's unspent dispersal and grants this month's. LICENSING_ENABLED only;
    # the credits-disabled early-return above already covers the credits gate.
    #
    # rollover_wallet is the whole implementation, and deliberately so: it is
    # already the tested primitive for "expire the bundle, grant the new period,
    # write the compensating ledger rows", and it no-ops when the period hasn't
    # ended — which IS the once-per-month idempotency. A bespoke dispersal
    # grant would need to reimplement all of that, including the expiry row that
    # keeps sum(delta) reconciling to the balance.
    #
    # Only the BUNDLE bucket is touched, so purchased packs (reserve) survive
    # untouched: a contract that lapses doesn't confiscate credits the org bought.
    if licensing_enabled():
        orgs = _capped(
            sb.table("organizations")
            .select("id, monthly_dispersal_credits")
            .eq("status", "active")
            .is_("archived_at", "null")
            .gt("monthly_dispersal_credits", 0),
            "organizations(dispersal)",
        )
        for org in orgs:
            org_id = org["id"]
            dispersal = org.get("monthly_dispersal_credits") or 0
            if dispersal <= 0:
                continue  # defensive — the query already filters this
            try:
                pool_rows = (
                    sb.table("credit_wallets")
                    .select("id, period_end")
                    .eq("owner_type", "org")
                    .eq("owner_id", org_id)
                    .execute()
                    .data
                    or []
                )
                if not pool_rows:
                    # No pool wallet yet: the app layer creates it on the first
                    # org-context read or purchase. Nothing to roll into.
                    logger.info("sweep dispersal: org %s has no pool wallet yet — skipping", org_id)
                    continue
                pool = pool_rows[0]
                # First-ever dispersal: a NULL period_end means the pool has
                # never been on a cycle, so anchor it to the start of this month
                # rather than "now", keeping every org's period aligned to
                # calendar months (which is what member caps reset against).
                current_end = _parse_iso(pool.get("period_end"))
                if current_end is None:
                    new_end = (now.replace(day=1) + relativedelta(months=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    if current_end > now:
                        continue  # period still open — nothing due
                    new_end = current_end
                    while new_end < now:
                        new_end = new_end + relativedelta(months=1)
                res = sb.rpc(
                    "rollover_wallet",
                    {
                        "p_wallet_id": pool["id"],
                        "p_monthly_grant": dispersal,
                        "p_new_period_start": (new_end - relativedelta(months=1)).isoformat(),
                        "p_new_period_end": new_end.isoformat(),
                    },
                ).execute()
                if res.data:
                    dispersed += 1
            except Exception:
                logger.exception("sweep dispersal failed org=%s", org_id)

    return {
        "walletsRolled": rolled,
        "overageBilled": overage_billed,
        "annualInvoiced": annual_invoiced,
        "orgsDispersed": dispersed,
    }
