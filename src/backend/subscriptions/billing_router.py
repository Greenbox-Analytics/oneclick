"""Billing router: Stripe Checkout, Portal, and webhook endpoints."""

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import subscriptions.stripe_client as stripe_client_module
import subscriptions.stripe_events as stripe_events_module
from analytics import capture as analytics_capture
from auth import get_current_user_email, get_current_user_id

router = APIRouter(prefix="/billing", tags=["billing"])

class CreateCheckoutRequest(BaseModel):
    plan: str  # "monthly" | "annual"
    # Optional return paths so different flows (onboarding, pricing page, etc.)
    # can land users back where they were when they cancel. Must be relative
    # paths starting with "/" — absolute URLs are rejected to prevent open
    # redirects to phishing domains. None = use the default /pricing or
    # /subscription page.
    cancel_path: str | None = None
    success_path: str | None = None


def _safe_return_path(path: str | None, default: str) -> str:
    """Whitelist relative paths to prevent open-redirect via cancel_url."""
    if not path:
        return default
    if not path.startswith("/") or path.startswith("//"):
        return default
    return path


@router.post("/create-checkout-session")
async def create_checkout_session(
    body: CreateCheckoutRequest,
    user_id: str = Depends(get_current_user_id),
    email: str = Depends(get_current_user_email),
):
    """Create a Stripe Checkout session for the requested plan; return redirect URL."""
    if body.plan == "monthly":
        price_id = os.environ["STRIPE_PRICE_MONTHLY"]
    elif body.plan == "annual":
        price_id = os.environ["STRIPE_PRICE_ANNUAL"]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {body.plan}")

    frontend_url = os.environ["FRONTEND_URL"]
    success_path = _safe_return_path(
        body.success_path, "/subscription?stripe_session_id={CHECKOUT_SESSION_ID}&welcome=true"
    )
    cancel_path = _safe_return_path(body.cancel_path, "/pricing?canceled=true")
    # Stripe replaces {CHECKOUT_SESSION_ID} server-side; preserve the literal
    # placeholder if it's in the path. The default already includes it.
    success_url = f"{frontend_url}{success_path}"
    cancel_url = f"{frontend_url}{cancel_path}"

    stripe = stripe_client_module.get_stripe()
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        customer_email=email,
        metadata={"user_id": user_id},
        subscription_data={"metadata": {"user_id": user_id}},
        success_url=success_url,
        cancel_url=cancel_url,
    )
    analytics_capture(user_id, "checkout_started", {"plan": body.plan})
    return {"url": session.url}


@router.post("/create-portal-session")
async def create_portal_session(
    user_id: str = Depends(get_current_user_id),
):
    """Create a Stripe Customer Portal session; return redirect URL.

    Returns 404 if the user has no stripe_customer_id (e.g., manually-granted Pro
    users with only a tier_overrides row).
    """
    from main import get_supabase_client

    sb = get_supabase_client()
    sub_res = sb.table("subscriptions").select("stripe_customer_id").eq("user_id", user_id).execute()
    if not sub_res.data or not sub_res.data[0].get("stripe_customer_id"):
        raise HTTPException(
            status_code=404,
            detail="No Stripe subscription on file. If you believe this is an error, contact support.",
        )

    frontend_url = os.environ["FRONTEND_URL"]
    portal = stripe_client_module.get_stripe().billing_portal.Session.create(
        customer=sub_res.data[0]["stripe_customer_id"],
        return_url=f"{frontend_url}/subscription",
    )
    analytics_capture(user_id, "billing_portal_opened", {})
    return {"url": portal.url}


@router.post("/webhook")
async def webhook(request: Request):
    """Receive Stripe webhook events.

    Flow:
    1. Verify signature (400 on failure) to ensure request is from Stripe
    2. INSERT event_id into stripe_events (idempotency; conflict → {duplicate: true})
    3. Dispatch to handler (500 on failure + delete idempotency row so Stripe retries)
    4. Return 200 on success
    """
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        event = stripe_client_module.verify_webhook(payload, sig)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid signature: {e}")

    from main import get_supabase_client

    sb = get_supabase_client()

    # Idempotency: insert event_id; if conflict, we've already processed this event
    try:
        sb.table("stripe_events").insert(
            {
                "event_id": event.id,
                "event_type": event.type,
                "payload": event.to_dict(),
            }
        ).execute()
    except Exception:
        # Duplicate key (or other insert failure) → ack so Stripe stops retrying
        return {"received": True, "duplicate": True}

    handler = stripe_events_module.HANDLERS.get(event.type)
    if handler is None:
        # Unknown event type — ack so Stripe stops retrying
        return {"received": True, "handled": False}

    try:
        handler(event, sb)
    except Exception as e:
        # Handler failed — delete idempotency row so Stripe will retry
        try:
            sb.table("stripe_events").delete().eq("event_id", event.id).execute()
        except Exception:
            pass  # best-effort cleanup
        raise HTTPException(status_code=500, detail=f"Handler error: {e}")

    return {"received": True, "handled": True}
