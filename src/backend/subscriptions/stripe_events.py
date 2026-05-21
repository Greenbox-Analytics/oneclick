"""Stripe webhook event handlers.

Idempotency is handled at the router level (via stripe_events table). Each
handler is safe to invoke multiple times for the same event_id: we use upsert
or update operations that converge to the same final state.

Note on `invoice.payment_succeeded`: intentionally NOT handled — the parallel
customer.subscription.updated event carries the same period_end info, and
handling both would create redundant DB writes.
"""

import logging
from datetime import UTC, datetime

import stripe

from analytics import capture as analytics_capture
from analytics import identify as analytics_identify

logger = logging.getLogger(__name__)


def _ts(epoch: int | None) -> str | None:
    """Convert Stripe's UNIX timestamps to ISO strings for Postgres TIMESTAMPTZ."""
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def handle_checkout_session_completed(event, supabase) -> None:
    """User finished Checkout → upsert subscriptions row with tier='pro'."""
    session = event.data.object
    user_id = session.metadata.get("user_id") if session.metadata else None
    if not user_id:
        return  # Shouldn't happen; metadata is set at session creation

    subscription_id = session.subscription
    customer_id = session.customer

    # Fetch the Subscription to get price + period info
    sub = stripe.Subscription.retrieve(subscription_id)
    price_id = sub["items"]["data"][0]["price"]["id"]

    supabase.table("subscriptions").upsert(
        {
            "user_id": user_id,
            "tier": "pro",
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
    analytics_capture(user_id, "subscription_activated", {"stripe_price_id": price_id, "status": sub.status})
    try:
        analytics_identify(user_id, {"plan": "pro"})
    except Exception as e:
        logger.warning("analytics identify on subscription_activated failed: %s", e)


def handle_subscription_updated(event, supabase) -> None:
    """Plan change / status update / cancel scheduled — sync fields BUT NOT tier."""
    sub = event.data.object
    user_id = sub.metadata.get("user_id") if sub.metadata else None
    if not user_id:
        return

    price_id = sub["items"]["data"][0]["price"]["id"]

    supabase.table("subscriptions").update(
        {
            "status": sub.status,
            "stripe_price_id": price_id,
            "current_period_start": _ts(sub.current_period_start),
            "current_period_end": _ts(sub.current_period_end),
            "cancel_at_period_end": sub.cancel_at_period_end,
            "canceled_at": _ts(sub.canceled_at) if sub.canceled_at else None,
            # tier intentionally NOT updated — only subscription.deleted changes tier
        }
    ).eq("user_id", user_id).execute()


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


# Dispatcher: maps Stripe event types to handler functions.
# Keys must match Stripe's exact event type strings.
HANDLERS = {
    "checkout.session.completed": handle_checkout_session_completed,
    "customer.subscription.updated": handle_subscription_updated,
    "customer.subscription.deleted": handle_subscription_deleted,
    "invoice.payment_failed": handle_invoice_payment_failed,
}
