"""Stripe webhook event handlers.

Idempotency is handled at the router level (via stripe_events table). Each
handler is safe to invoke multiple times for the same event_id: we use upsert
or update operations that converge to the same final state.

Note on `invoice.payment_succeeded`: intentionally NOT handled — the parallel
customer.subscription.updated event carries the same period_end info, and
handling both would create redundant DB writes.
"""

import logging
import os
from datetime import UTC, datetime

import stripe
from dateutil.relativedelta import relativedelta

from analytics import capture as analytics_capture
from analytics import identify as analytics_identify

logger = logging.getLogger(__name__)


def _ts(epoch: int | None) -> str | None:
    """Convert Stripe's UNIX timestamps to ISO strings for Postgres TIMESTAMPTZ."""
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, UTC).isoformat()


TIER_RANK = {"free": 0, "pro": 1, "pro_max": 2}


def _tier_for_price(price_id: str | None) -> str:
    """Map a Stripe price id to a tier. Unknown/legacy prices default to 'pro'."""
    pro_max_prices = {
        os.getenv("STRIPE_PRICE_PRO_MAX_MONTHLY"),
        os.getenv("STRIPE_PRICE_PRO_MAX_ANNUAL"),
    }
    return "pro_max" if price_id and price_id in pro_max_prices else "pro"


def _align_wallet_to_checkout(supabase, user_id: str, tier: str, event_id: str) -> None:
    """Checkout: top the bundle up to the tier's grant + re-anchor the period.

    MUST NOT use rollover_wallet here: its self-defense predicate
    (period_end <= now()) makes it silently return false for a wallet whose
    period_end is still in the future — exactly a mid-period upgrader's state.
    grant_credits does the idempotent top-up (keyed on the Stripe event id);
    the period is re-anchored with a direct service-role UPDATE (checkout is
    rare enough that the unguarded period write is acceptable).

    No try/except: failures must 500 so Stripe retries; the grant is
    request-id idempotent, so retries are safe.
    """
    tier_res = supabase.table("tier_entitlements").select("monthly_credits").eq("tier", tier).execute()
    if not tier_res.data:
        logger.warning(
            "checkout wallet alignment: no tier_entitlements row for tier=%s — paying user %s gets no grant",
            tier,
            user_id,
        )
    grant = tier_res.data[0]["monthly_credits"] if tier_res.data else 0
    wallet_res = (
        supabase.table("credit_wallets")
        .select("id, bundle_balance, period_start")
        .eq("owner_type", "user")
        .eq("owner_id", user_id)
        .execute()
    )
    if not wallet_res.data:
        return
    wallet = wallet_res.data[0]
    # Anti-farming: cap by what's already been granted this period, not
    # just the unspent bundle — a spend→downgrade→re-upgrade loop must not
    # refill. (The spec assumes portal downgrades apply at period end; this
    # guard holds even if that configuration drifts.)
    granted_rows = (
        supabase.table("credit_ledger")
        .select("delta, created_at")
        .eq("wallet_id", wallet["id"])
        .eq("kind", "monthly_grant")
        .gte("created_at", wallet.get("period_start") or "1970-01-01")
        .execute()
    )
    granted_this_period = sum(r["delta"] for r in (granted_rows.data or []))
    top_up = min(
        max(grant - max(wallet.get("bundle_balance", 0), 0), 0),
        max(grant - granted_this_period, 0),
    )
    if top_up > 0:
        supabase.rpc(
            "grant_credits",
            {
                "p_wallet_id": wallet["id"],
                "p_amount": top_up,
                "p_kind": "monthly_grant",
                "p_bucket": "bundle",
                "p_metadata": {"reason": "checkout", "tier": tier},
                "p_request_id": f"checkout:{event_id}",
            },
        ).execute()
    now = datetime.now(UTC)
    supabase.table("credit_wallets").update(
        {
            "period_start": now.isoformat(),
            "period_end": (now + relativedelta(months=1)).isoformat(),
            "overage_this_period": 0,
        }
    ).eq("id", wallet["id"]).execute()


def handle_checkout_session_completed(event, supabase) -> None:
    """User finished Checkout → upsert subscriptions row with tier resolved from price."""
    session = event.data.object
    user_id = session.metadata.get("user_id") if session.metadata else None
    if not user_id:
        return  # Shouldn't happen; metadata is set at session creation

    subscription_id = session.subscription
    customer_id = session.customer

    # Fetch the Subscription to get price + period info
    sub = stripe.Subscription.retrieve(subscription_id)
    price_id = sub["items"]["data"][0]["price"]["id"]
    tier = _tier_for_price(price_id)

    supabase.table("subscriptions").upsert(
        {
            "user_id": user_id,
            "tier": tier,
            "status": sub.status,  # 'active' or 'trialing'
            "stripe_customer_id": customer_id,
            "stripe_subscription_id": subscription_id,
            "stripe_price_id": price_id,
            "current_period_start": _ts(sub.current_period_start),
            "current_period_end": _ts(sub.current_period_end),
            "cancel_at_period_end": sub.cancel_at_period_end,
            "canceled_at": _ts(sub.canceled_at) if sub.canceled_at else None,
        },
        on_conflict="user_id",
    ).execute()
    analytics_capture(
        user_id, "subscription_activated", {"stripe_price_id": price_id, "status": sub.status, "tier": tier}
    )
    try:
        analytics_identify(user_id, {"plan": tier})
    except Exception as e:
        logger.warning("analytics identify on subscription_activated failed: %s", e)

    from subscriptions.service import credits_enabled

    if credits_enabled():
        _align_wallet_to_checkout(supabase, user_id, tier, event.id)


def handle_subscription_updated(event, supabase) -> None:
    """Plan change / status update / cancel scheduled — sync fields INCLUDING tier.

    Tier IS now synced here (this deliberately extends the old "never touch
    tier" isolation): portal-driven Pro<->Pro Max switches only surface via
    this event, never checkout.session.completed. Downgrades leave existing
    wallet balances alone — the bundle clamps down at the next rollover
    (never a mid-period claw-back); the reserve balance survives every
    transition, upgrade or downgrade.
    """
    sub = event.data.object
    user_id = sub.metadata.get("user_id") if sub.metadata else None
    if not user_id:
        return

    price_id = sub["items"]["data"][0]["price"]["id"]
    new_tier = _tier_for_price(price_id)

    prev_res = supabase.table("subscriptions").select("tier").eq("user_id", user_id).execute()
    prev_tier = prev_res.data[0]["tier"] if prev_res.data else "free"

    from subscriptions.service import credits_enabled

    update = {
        "status": sub.status,
        "stripe_price_id": price_id,
        "current_period_start": _ts(sub.current_period_start),
        "current_period_end": _ts(sub.current_period_end),
        "cancel_at_period_end": sub.cancel_at_period_end,
        "canceled_at": _ts(sub.canceled_at) if sub.canceled_at else None,
    }
    # Pre-credits, this handler never touched tier — gate the write so
    # flipping CREDITS_ENABLED off is a clean rollback to that behavior.
    if credits_enabled():
        update["tier"] = new_tier

    supabase.table("subscriptions").update(update).eq("user_id", user_id).execute()

    if credits_enabled() and TIER_RANK.get(new_tier, 0) > TIER_RANK.get(prev_tier, 0):
        # Upgrade: TOP UP the bundle to the new tier's grant — never additive.
        # No try/except: failures must 500 so Stripe retries; the grant is
        # request-id idempotent, so retries are safe.
        tiers = supabase.table("tier_entitlements").select("tier, monthly_credits").eq("tier", new_tier).execute()
        if not tiers.data:
            logger.warning(
                "upgrade top-up: no tier_entitlements row for tier=%s — paying user %s gets no grant",
                new_tier,
                user_id,
            )
        new_grant = tiers.data[0]["monthly_credits"] if tiers.data else 0
        wallet_res = (
            supabase.table("credit_wallets")
            .select("id, bundle_balance, period_start")
            .eq("owner_type", "user")
            .eq("owner_id", user_id)
            .execute()
        )
        if wallet_res.data:
            wallet = wallet_res.data[0]
            # Anti-farming: cap by what's already been granted this period, not
            # just the unspent bundle — a spend→downgrade→re-upgrade loop must not
            # refill. (The spec assumes portal downgrades apply at period end; this
            # guard holds even if that configuration drifts.)
            granted_rows = (
                supabase.table("credit_ledger")
                .select("delta, created_at")
                .eq("wallet_id", wallet["id"])
                .eq("kind", "monthly_grant")
                .gte("created_at", wallet.get("period_start") or "1970-01-01")
                .execute()
            )
            granted_this_period = sum(r["delta"] for r in (granted_rows.data or []))
            # TOCTOU residual (accepted): concurrent subscription.updated
            # deliveries can both read a stale bundle and over-grant; bounded
            # by the period-sum cap on the next event and clamped at next
            # rollover.
            top_up = min(
                max(new_grant - max(wallet.get("bundle_balance", 0), 0), 0),
                max(new_grant - granted_this_period, 0),
            )
            if top_up > 0:
                supabase.rpc(
                    "grant_credits",
                    {
                        "p_wallet_id": wallet["id"],
                        "p_amount": top_up,
                        "p_kind": "monthly_grant",
                        "p_bucket": "bundle",
                        "p_metadata": {"reason": "tier_upgrade_topup", "from": prev_tier, "to": new_tier},
                        # Stripe redelivers events; a handler that failed AFTER
                        # granting would re-grant on retry without this key.
                        "p_request_id": f"tier-upgrade:{event.id}",
                    },
                ).execute()


def handle_subscription_deleted(event, supabase) -> None:
    """Subscription truly ended (cancel-at-period-end fired, or hard delete).
    User loses Pro access; SP3 gates re-engage on next request."""
    sub = event.data.object
    user_id = sub.metadata.get("user_id") if sub.metadata else None
    if not user_id:
        return

    supabase.table("subscriptions").update(
        {
            "tier": "free",
            "status": "canceled",
            "canceled_at": _ts(sub.canceled_at) if sub.canceled_at else _ts(int(datetime.now(UTC).timestamp())),
            # Keep stripe_customer_id for re-subscribe convenience
            "stripe_subscription_id": None,
            "stripe_price_id": None,
            "current_period_end": None,
            "cancel_at_period_end": False,
        }
    ).eq("user_id", user_id).execute()
    analytics_capture(user_id, "subscription_canceled", {})
    try:
        analytics_identify(user_id, {"plan": "free"})
    except Exception as e:
        logger.warning("analytics identify on subscription_canceled failed: %s", e)

    from subscriptions.service import credits_enabled

    if credits_enabled():
        # Final billing (spec §7 gap): after deletion this user leaves the
        # paid-tier sweep population, so any unbilled overage would be orphaned
        # forever, and already-created pending InvoiceItems would float until a
        # future re-subscribe surprise-bills them. Convert stragglers to pending
        # items, then collect EVERYTHING floating onto one final auto-advancing
        # invoice. No try/except: a failure 500s the webhook so Stripe retries;
        # every step is idempotent (invoice_item_id checks, Stripe idempotency
        # keys, swept stamps).
        from subscriptions.overage_billing import bill_pending_overage, invoice_unswept_items

        customer_id = getattr(sub, "customer", None)
        if not customer_id:
            row = supabase.table("subscriptions").select("stripe_customer_id").eq("user_id", user_id).execute()
            customer_id = row.data[0].get("stripe_customer_id") if row.data else None
        if customer_id:
            bill_pending_overage(supabase, user_id)
            wallet_res = (
                supabase.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", user_id).execute()
            )
            if wallet_res.data:
                invoice_unswept_items(
                    supabase, wallet_res.data[0]["id"], customer_id, idempotency_key=f"final:{event.id}"
                )


def handle_invoice_payment_failed(event, supabase) -> None:
    """Failed renewal charge → status=past_due. Tier stays 'pro' during retries.
    Stripe retries automatically; if retries exhaust, customer.subscription.deleted
    fires and tier drops to 'free'."""
    invoice = event.data.object
    subscription_id = invoice.subscription
    if not subscription_id:
        return  # one-off invoice, not subscription-related

    sub = stripe.Subscription.retrieve(subscription_id)
    user_id = sub.metadata.get("user_id") if sub.metadata else None
    if not user_id:
        return

    supabase.table("subscriptions").update(
        {
            "status": "past_due",
        }
    ).eq("user_id", user_id).execute()
    analytics_capture(user_id, "payment_failed", {})
    try:
        analytics_identify(user_id, {"plan": "free"})
    except Exception as e:
        logger.warning("analytics identify on payment_failed failed: %s", e)


def handle_invoice_created(event, supabase) -> None:
    """Safety net (spec §7): before a renewal invoice finalizes, ensure every
    unbilled overage row has a pending InvoiceItem ATTACHED TO IT (pending
    items created after a draft exists don't auto-attach)."""
    invoice = event.data.object
    if getattr(invoice, "billing_reason", None) != "subscription_cycle":
        return
    customer = invoice.customer
    if not customer:
        return

    from subscriptions.service import credits_enabled

    if not credits_enabled():
        return

    sub_res = supabase.table("subscriptions").select("user_id").eq("stripe_customer_id", customer).execute()
    if not sub_res.data:
        return

    from subscriptions.overage_billing import bill_pending_overage

    bill_pending_overage(supabase, sub_res.data[0]["user_id"], invoice_id=invoice.id)


# Dispatcher: maps Stripe event types to handler functions.
# Keys must match Stripe's exact event type strings.
HANDLERS = {
    "checkout.session.completed": handle_checkout_session_completed,
    "customer.subscription.updated": handle_subscription_updated,
    "customer.subscription.deleted": handle_subscription_deleted,
    "invoice.payment_failed": handle_invoice_payment_failed,
    "invoice.created": handle_invoice_created,
}
