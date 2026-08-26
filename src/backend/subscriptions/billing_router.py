"""Billing router: Stripe Checkout, Portal, and webhook endpoints."""

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, model_validator

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
    # /profile (Account & Billing) page.
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
    # Plan params are <tier>_<period> (the FE's CheckoutPlan type). Env var
    # names are unchanged on purpose (they are deploy secrets): STRIPE_PRICE_*
    # is the basic plan, PRO_MAX is the pro one.
    PLAN_TO_ENV = {
        "basic_monthly": "STRIPE_PRICE_MONTHLY",
        "basic_annual": "STRIPE_PRICE_ANNUAL",
        "pro_monthly": "STRIPE_PRICE_PRO_MAX_MONTHLY",
        "pro_annual": "STRIPE_PRICE_PRO_MAX_ANNUAL",
    }
    env_key = PLAN_TO_ENV.get(body.plan)
    if env_key is None:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {body.plan}")
    price_id = os.environ[env_key]

    frontend_url = os.environ["FRONTEND_URL"]
    success_path = _safe_return_path(body.success_path, "/profile?stripe_session_id={CHECKOUT_SESSION_ID}&welcome=true")
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


class CreateTopupRequest(BaseModel):
    """Buy credits: EITHER a preset bundle OR a custom credit count.

    Note what is absent: a price. The client sends a credit COUNT and the
    server derives the amount (subscriptions.credit_purchase), so a tampered
    request can only ask for a different quantity, never a different price.
    """

    pack_key: str | None = None
    # Custom amount, in credits. Bounded and priced server-side by
    # credit_purchase.validate_custom_credits / price_cents_for_credits.
    credits: int | None = None
    # Licensing Phase B: when set, the purchase (bundle OR custom) targets
    # that org's credit pool instead of the caller's personal wallet. Caller
    # must be an ACTIVE ADMIN of a NON-ARCHIVED org (checked below via
    # orgs.authz); None (default) preserves the personal-wallet flow.
    org_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_item(self):
        if (self.pack_key is None) == (self.credits is None):
            raise ValueError("Provide either pack_key or credits, not both.")
        return self


@router.get("/credit-packs")
async def list_credit_packs():
    """Bundles, custom-amount bounds, and tool prices for the pack picker.

    DELIBERATELY unauthenticated (stated decision, not an omission): this is
    public pricing data — no user state — and the pricing page must render it
    logged-out. Every sibling route stays user-authed.

    A pack is listed on `active` ALONE. It used to also require a configured
    stripe_price_id, which left the whole ladder unsellable until an operator
    hand-created Prices in the Stripe dashboard; create_topup_session now
    builds the line item ad-hoc from price_cents when that column is NULL, so
    the extra filter would only hide sellable packs.

    Doubles as the RECURRING top-up catalog (spec 2026-08-15 §4.3): a pack is
    also buyable monthly exactly when the operator has set a recurring Stripe
    price on it, surfaced as `recurringPriceId` (null on the rest). No second
    endpoint and no second table — a top-up is the same pack, billed monthly.
    Recurring genuinely DOES still need a real Price (Stripe cannot bill an
    ad-hoc one-time amount on a subscription), which is why that column stays.

    `prices` is the live per-action credit table, shipped so the picker can say
    what a bundle typically buys ("about 40 OneClick runs") without hardcoding
    numbers that would drift from the DB the next time base rates move. It is
    OMITTED entirely when credit_prices reads empty, so the UI drops the
    subtitle rather than quoting "0 runs".
    """
    from main import get_supabase_client
    from subscriptions.credit_purchase import custom_config

    sb = get_supabase_client()
    res = (
        sb.table("credit_packs")
        .select("key, label, credits, price_cents, sort_order, recurring_stripe_price_id")
        .eq("active", True)
        .order("sort_order")
        .execute()
    )
    packs = [
        {
            "key": p["key"],
            "label": p.get("label"),
            "credits": p["credits"],
            "price_cents": p["price_cents"],
            "sort_order": p["sort_order"],
            "recurringPriceId": p.get("recurring_stripe_price_id"),
        }
        for p in (res.data or [])
    ]

    payload: dict = {"packs": packs, "custom": custom_config()}

    # Same public-read table EntitlementsService._get_credit_prices() reads;
    # queried directly here because this route is unauthenticated and holds no
    # service instance. Keys mirror Entitlements.to_dict()'s `prices` block
    # exactly, so one frontend type serves both payloads.
    price_res = sb.table("credit_prices").select("action, credits").execute()
    prices = {row["action"]: row["credits"] for row in (price_res.data or [])}
    if prices:
        payload["prices"] = {
            "zoeMessage": prices.get("zoe_message"),
            "oneclickRun": prices.get("oneclick_run"),
            "registryParse": prices.get("registry_parse"),
            "splitSheet": prices.get("split_sheet"),
        }
    return payload


@router.post("/create-topup-session")
async def create_topup_session(
    body: CreateTopupRequest,
    user_id: str = Depends(get_current_user_id),
    email: str = Depends(get_current_user_email),
):
    """One-time credit purchase — a preset bundle, or a custom credit amount.

    Both target the personal wallet by default (bundles: spec 2026-07-19 §3,
    Phase A byte-identical); `org_id` routes the SAME purchase — bundle or
    custom — into that org's pool instead (Phase B, admin-gated). The org
    gates (licensing flag, require_admin, archived check) run BEFORE the
    bundle/custom fork below, so both products sit behind identical authz.
    """
    from subscriptions.credit_purchase import (
        credits_line_item,
        price_cents_for_credits,
        validate_custom_credits,
    )
    from subscriptions.service import credits_enabled

    if not credits_enabled():
        raise HTTPException(status_code=409, detail="Credit top-ups aren't available yet.")

    from main import get_supabase_client

    sb = get_supabase_client()

    # Licensing Phase B: org-pool target. Checked BEFORE any pack lookup so
    # an unauthorized/disallowed request never leaks pack availability.
    # 404 (not 409) when the flag is off — same "don't reveal the feature"
    # stance as orgs/router.py's require_licensing dependency.
    target = "user"
    if body.org_id:
        from subscriptions.service import licensing_enabled

        if not licensing_enabled():
            raise HTTPException(status_code=404, detail="Not found")

        from orgs.authz import require_admin

        require_admin(sb, user_id, body.org_id)  # raises 403 if not an active admin

        org_res = sb.table("organizations").select("archived_at").eq("id", body.org_id).execute()
        org_row = org_res.data[0] if org_res.data else None
        if org_row and org_row.get("archived_at"):
            raise HTTPException(status_code=409, detail="This organization is archived.")
        target = body.org_id

    # Resolve the line item + metadata for whichever of the two products this
    # is. The metadata written here is the ONLY thing the webhook trusts to
    # decide what to grant, and this endpoint is its only writer.
    if body.pack_key is not None:
        res = (
            sb.table("credit_packs")
            .select("key, credits, price_cents, stripe_price_id, active")
            .eq("key", body.pack_key)
            .execute()
        )
        pack = res.data[0] if res.data else None
        if not pack or not pack.get("active"):
            raise HTTPException(status_code=400, detail="That credit pack isn't available.")
        # An operator-set Price wins when present (keeps Stripe-native
        # per-price reporting for anyone who wants it); otherwise bill the
        # catalog's price_cents ad-hoc, which is what makes the ladder
        # sellable with no Stripe dashboard setup at all.
        if pack.get("stripe_price_id"):
            line_item = {"price": pack["stripe_price_id"], "quantity": 1}
        else:
            line_item = credits_line_item(
                price_cents=pack["price_cents"],
                name=f"{pack['credits']:,} Msanii credits",
            )
        metadata = {"user_id": user_id, "pack_key": pack["key"], "target": target}
        analytics_plan = pack["key"]
        analytics_credits = pack["credits"]
    else:
        try:
            credits = validate_custom_credits(body.credits)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        line_item = credits_line_item(
            price_cents=price_cents_for_credits(credits),
            name=f"{credits:,} Msanii credits",
        )
        # No `pack_key` (there is no catalog row), and deliberately NO `kind`
        # key: `kind` is the org_topup discriminator that
        # handle_checkout_session_completed branches on FIRST, and overloading
        # it would route a credit purchase into the org top-up handler.
        metadata = {"user_id": user_id, "credits": str(credits), "target": target}
        analytics_plan = "custom"
        analytics_credits = credits

    frontend_url = os.environ["FRONTEND_URL"]
    stripe = stripe_client_module.get_stripe()
    # Attach the charge to the user's existing Stripe Customer when they have
    # one. In mode="payment", Checkout defaults customer_creation=if_required —
    # with only customer_email the pack payment would NOT land on the Customer
    # carrying their subscription, and support handling a refund would never
    # find it next to their invoices. (The subscription endpoint can rely on
    # customer_email because subscription mode always creates+attaches one.)
    # Unchanged for org-pool purchases too: there's no org-level Stripe
    # Customer, so the buying admin's own Customer/email is used either way.
    sub_res = sb.table("subscriptions").select("stripe_customer_id").eq("user_id", user_id).execute()
    customer_id = sub_res.data[0].get("stripe_customer_id") if sub_res.data else None
    customer_kwargs = {"customer": customer_id} if customer_id else {"customer_email": email}
    return_base = "/teams" if target != "user" else "/profile"
    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[line_item],
        # metadata.target is 'user' (personal wallet) or an org id (Phase B).
        metadata=metadata,
        success_url=f"{frontend_url}{return_base}?topup=success",
        cancel_url=f"{frontend_url}{return_base}?topup=canceled",
        **customer_kwargs,
    )
    analytics_capture(
        user_id,
        "checkout_started",
        {"plan": analytics_plan, "kind": "topup", "credits": analytics_credits},
    )
    return {"url": session.url}


class OrgTopupCheckoutRequest(BaseModel):
    org_id: str
    key: str  # credit_packs.key — the pack must carry a recurring Stripe price


@router.post("/org-topup-checkout")
async def create_org_topup_checkout(
    body: OrgTopupCheckoutRequest,
    user_id: str = Depends(get_current_user_id),
    email: str = Depends(get_current_user_email),
):
    """Recurring org top-up (spec 2026-08-15 §4.3): the SAME credit pack, sold
    as a monthly Stripe SUBSCRIPTION that refills the org's pool every period.

    Billed to the CALLER's personal Stripe customer — there is no org-level
    customer — using the same customer-resolution block as the pack path, so
    the charge lands next to that admin's other invoices for refund lookups.
    One top-up per org (409 on a second): a second subscription would be a
    second card silently billing for the same team.

    METADATA CONTRACT (load-bearing): the triple {org_id, kind, purchased_by}
    goes on BOTH objects — `metadata` is what checkout.session.completed reads,
    and `subscription_data.metadata` is what Stripe copies onto the
    Subscription and onto every invoice's `subscription_details.metadata`
    (which is how invoice.paid resolves the org). NEITHER carries a user_id:
    that key is what routes an event into the PERSONAL subscription handlers,
    and a top-up must never be mistaken for the admin's own plan.
    """
    from subscriptions.service import credits_enabled, licensing_enabled

    if not credits_enabled():
        raise HTTPException(status_code=409, detail="Credit top-ups aren't available yet.")
    # 404 (not 409) with licensing off — same "don't reveal the feature" stance
    # as the /orgs router gate and the pack path's org branch.
    if not licensing_enabled():
        raise HTTPException(status_code=404, detail="Not found")

    from main import get_supabase_client
    from orgs.authz import require_admin

    sb = get_supabase_client()
    # Authz FIRST, before any pack or org detail is read back — same ordering
    # (and same reason) as the pack path's org branch.
    require_admin(sb, user_id, body.org_id)

    org_res = (
        sb.table("organizations")
        .select("kind, archived_at, dissolved_at, topup_stripe_subscription_id")
        .eq("id", body.org_id)
        .execute()
    )
    org = org_res.data[0] if org_res.data else None
    if not org or org.get("kind") != "self_serve":
        raise HTTPException(status_code=409, detail="This organization is managed by Msanii")
    if org.get("dissolved_at"):
        # dissolved_at only exists on self-serve orgs (the kind check above
        # already refused any other kind), so "team" — the self-serve UI
        # vocabulary — is correct here, unlike the generic "organization"
        # phrasing on the archived/managed-by-Msanii branches around it.
        raise HTTPException(status_code=409, detail="This team has been dissolved")
    if org.get("archived_at"):
        raise HTTPException(status_code=409, detail="This organization is archived.")
    if org.get("topup_stripe_subscription_id"):
        raise HTTPException(status_code=409, detail="This team already has a monthly credit top-up.")
    # KNOWN RACE (Task 17, documented not closed — the fix lives in
    # stripe_events.py, owned by another workstream): two admins can both read
    # topup_stripe_subscription_id as NULL here and each open a Stripe Checkout
    # session below, creating two live top-up subscriptions for this org. The
    # second checkout.session.completed webhook overwrites the column, so the
    # FIRST subscription keeps billing with nothing in the DB naming it.
    # Accepted remedy for now: the losing subscription still shows up in the
    # purchasing admin's own Stripe billing portal and can be canceled there.
    # Closing this properly means a re-check inside the webhook handler
    # (compound uniqueness / re-read-before-write on topup_stripe_subscription_id)
    # before it accepts a second org_topup subscription for the same org.

    # Same lookup shape as the pack path: fetch by key, then let the Python
    # guard below decide. A pack with no recurring price is simply not part of
    # this catalog, and reads identically to an unknown key (no existence oracle).
    pack_res = sb.table("credit_packs").select("key, recurring_stripe_price_id, active").eq("key", body.key).execute()
    pack = pack_res.data[0] if pack_res.data else None
    if not pack or not pack.get("active") or not pack.get("recurring_stripe_price_id"):
        raise HTTPException(status_code=404, detail="That monthly top-up isn't available.")

    sub_res = sb.table("subscriptions").select("stripe_customer_id").eq("user_id", user_id).execute()
    customer_id = sub_res.data[0].get("stripe_customer_id") if sub_res.data else None
    customer_kwargs = {"customer": customer_id} if customer_id else {"customer_email": email}

    frontend_url = os.environ["FRONTEND_URL"]
    metadata = {"org_id": body.org_id, "kind": "org_topup", "purchased_by": user_id}
    session = stripe_client_module.get_stripe().checkout.Session.create(
        mode="subscription",
        line_items=[{"price": pack["recurring_stripe_price_id"], "quantity": 1}],
        metadata=metadata,
        subscription_data={"metadata": dict(metadata)},
        success_url=f"{frontend_url}/teams?topup=success",
        cancel_url=f"{frontend_url}/teams?topup=canceled",
        **customer_kwargs,
    )
    analytics_capture(user_id, "checkout_started", {"plan": pack["key"], "kind": "org_topup", "org_id": body.org_id})
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
    except Exception as e:
        # ONLY a genuine unique-violation means "already processed". Anything
        # else (transient DB error) must 500 so Stripe RETRIES — acking here
        # would permanently drop a paid event the moment the DB hiccups.
        is_duplicate = getattr(e, "code", None) == "23505" or "23505" in str(e) or "duplicate key" in str(e).lower()
        if is_duplicate:
            return {"received": True, "duplicate": True}
        raise HTTPException(status_code=500, detail=f"Idempotency insert failed: {e}")

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
