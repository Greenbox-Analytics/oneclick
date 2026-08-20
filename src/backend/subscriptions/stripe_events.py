"""Stripe webhook event handlers.

Idempotency is handled at the router level (via stripe_events table). Each
handler is safe to invoke multiple times for the same event_id: we use upsert
or update operations that converge to the same final state.

Note on `invoice.payment_succeeded`: intentionally NOT handled — the parallel
customer.subscription.updated event carries the same period_end info, and
handling both would create redundant DB writes.
"""

import json
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


TIER_RANK = {"free": 0, "basic": 1, "pro": 2}


def _tier_for_price(price_id: str | None) -> str:
    """Map a Stripe price id to a tier. Unknown/legacy prices default to 'basic'.

    The env var names still say PRO_MAX: they point at the $50 plan's Stripe
    prices, which is the tier now KEYED 'pro'. Renaming them would mean rotating
    deploy secrets for a cosmetic win, so the mapping is documented instead.

    A price id that matches NEITHER tier's env vars still falls back to 'basic'
    (a paying customer must never be dropped to free by a config drift), but it
    is logged as an ERROR: a rotated/added Stripe price that nobody wired into
    STRIPE_PRICE_* would otherwise silently misassign tiers forever.
    """
    top_tier_prices = {
        os.getenv("STRIPE_PRICE_PRO_MAX_MONTHLY"),
        os.getenv("STRIPE_PRICE_PRO_MAX_ANNUAL"),
    }
    if price_id and price_id in top_tier_prices:
        return "pro"
    basic_prices = {os.getenv("STRIPE_PRICE_MONTHLY"), os.getenv("STRIPE_PRICE_ANNUAL")}
    if price_id not in basic_prices:
        logger.error(
            "unknown Stripe price id %r — not in any STRIPE_PRICE_* env var; defaulting tier to 'basic'",
            price_id,
        )
    return "basic"


def _capped_topup(supabase, wallet: dict, grant: int) -> int:
    """Anti-farming bundle top-up amount: cap by BOTH the unspent bundle AND what's
    already been granted this period (summed from the ledger since period_start) so
    a spend→downgrade→re-upgrade loop can't refill — never additive. (The spec
    assumes portal downgrades apply at period end; this guard holds even if that
    configuration drifts.)

    TOCTOU residual (accepted): concurrent subscription.updated deliveries can both
    read a stale bundle and over-grant; bounded by the period-sum cap on the next
    event and clamped at next rollover.
    """
    granted_rows = (
        supabase.table("credit_ledger")
        .select("delta, created_at")
        .eq("wallet_id", wallet["id"])
        .eq("kind", "monthly_grant")
        .gte("created_at", wallet.get("period_start") or "1970-01-01")
        .execute()
    )
    granted_this_period = sum(r["delta"] for r in (granted_rows.data or []))
    return min(
        max(grant - max(wallet.get("bundle_balance", 0), 0), 0),
        max(grant - granted_this_period, 0),
    )


def _topup_bundle(supabase, user_id: str, grant: int, metadata: dict, request_id: str) -> dict | None:
    """Top the user's bundle UP TO *grant* (capped by `_capped_topup`, never
    additive) as an idempotent `monthly_grant` keyed on *request_id*. Returns
    the wallet row (so a caller can re-anchor its period), or None when the
    user has no wallet. No try/except: failures must 500 so Stripe retries."""
    wallet_res = (
        supabase.table("credit_wallets")
        .select("id, bundle_balance, period_start")
        .eq("owner_type", "user")
        .eq("owner_id", user_id)
        .execute()
    )
    if not wallet_res.data:
        return None
    wallet = wallet_res.data[0]
    top_up = _capped_topup(supabase, wallet, grant)
    if top_up > 0:
        supabase.rpc(
            "grant_credits",
            {
                "p_wallet_id": wallet["id"],
                "p_amount": top_up,
                "p_kind": "monthly_grant",
                "p_bucket": "bundle",
                "p_metadata": metadata,
                "p_request_id": request_id,
            },
        ).execute()
    return wallet


def _align_wallet_to_checkout(supabase, user_id: str, tier: str, event_id: str, grant: int) -> None:
    """Checkout: top the bundle up to *grant* + re-anchor the period.

    MUST NOT use rollover_wallet here: its self-defense predicate
    (period_end <= now()) makes it silently return false for a wallet whose
    period_end is still in the future — exactly a mid-period upgrader's state.
    grant_credits does the idempotent top-up (keyed on the Stripe event id);
    the period is re-anchored with a direct service-role UPDATE (checkout is
    rare enough that the unguarded period write is acceptable).

    `grant` is resolved by the CALLER (Task 1 follow-up, spec review — this is
    a fifth grant-writing site): a same-tier re-checkout (past_due recovery,
    interval switch) must top up at an already-grandfathered grant, not the
    tier default, or the period reset below makes the wrong number stick for
    a month. This function no longer does its own tier_entitlements lookup.

    No try/except: failures must 500 so Stripe retries; the grant is
    request-id idempotent, so retries are safe.
    """
    wallet = _topup_bundle(supabase, user_id, grant, {"reason": "checkout", "tier": tier}, f"checkout:{event_id}")
    if wallet is None:
        return
    now = datetime.now(UTC)
    supabase.table("credit_wallets").update(
        {
            "period_start": now.isoformat(),
            "period_end": (now + relativedelta(months=1)).isoformat(),
            "overage_this_period": 0,
        }
    ).eq("id", wallet["id"]).execute()


def _plain(obj):
    """Stripe payload → plain dict, for the DICT-STYLE reads below.

    stripe-python >= 12 (pinned: 15.1.0) StripeObject is NO LONGER a dict
    subclass: `isinstance(o, dict)` is False and `o.get(...)` raises
    AttributeError. Every `x.metadata.get(...)` in this module therefore worked
    only on the dicts our tests feed and blew up (or silently read as empty) on
    live payloads. Attribute and SUBSCRIPT access still work on StripeObject,
    which is why the rest of the module (`sub.status`, `sub["items"]["data"]`)
    is unaffected — only the `.get()` sites needed this.

    `str(StripeObject)` is its JSON rendering, so one roundtrip yields nested
    plain dicts. Plain dicts and MagicMocks (the test shapes) pass through
    untouched. NOT `to_dict_recursive()` — that is private (`_to_dict_recursive`)
    in v15.
    """
    return json.loads(str(obj)) if isinstance(obj, stripe.StripeObject) else obj


def _subscription_metadata(invoice) -> dict:
    """The Subscription metadata Stripe copies onto EVERY invoice of that
    subscription (`invoice.subscription_details.metadata`).

    This is the only reliable way to tell an org top-up's invoices apart:
    webhooks are unordered, so the first `invoice.paid` can arrive before
    `checkout.session.completed` writes organizations.topup_stripe_subscription_id
    — matching on that column would silently drop the first month's credits.
    Returns {} for anything that isn't a plain mapping.
    """
    details = _plain(getattr(invoice, "subscription_details", None))
    meta = details.get("metadata") if isinstance(details, dict) else None
    return meta if isinstance(meta, dict) else {}


def handle_checkout_session_completed(event, supabase) -> None:
    """User finished Checkout → upsert subscriptions row with tier resolved from price."""
    session = event.data.object
    meta = _plain(session.metadata) or {}
    if meta.get("kind") == "org_topup":
        # Recurring ORG top-up (spec 2026-08-15 §4.3). MUST branch FIRST —
        # before the mode=="payment" pack branch and everything below it.
        # These sessions carry NO user_id, and the price is a pack's recurring
        # price that no STRIPE_PRICE_* env var knows: the personal path would
        # resolve it through _tier_for_price (unknown → 'basic') and upsert the
        # PURCHASING ADMIN's own subscriptions row, demoting a Pro admin to
        # Basic with the top-up subscription as their plan.
        org_id = meta.get("org_id")
        if not org_id:
            logger.error("org top-up checkout %s carries no org_id", getattr(session, "id", "?"))
            return
        admin_id = meta.get("purchased_by")
        supabase.table("organizations").update(
            {"topup_stripe_subscription_id": session.subscription, "topup_admin_id": admin_id}
        ).eq("id", org_id).execute()
        # `or org_id` so a hand-made session missing purchased_by can't 500 the
        # webhook on a None distinct id after the columns are already written.
        analytics_capture(admin_id or org_id, "org_topup_started", {"org_id": org_id})
        return
    if getattr(session, "mode", None) == "payment":
        # One-time credit pack — no subscription object exists on these
        # sessions, so this MUST branch before the Subscription.retrieve below.
        _handle_topup_completed(event, supabase)
        return
    user_id = meta.get("user_id")
    if not user_id:
        return  # Shouldn't happen; metadata is set at session creation

    subscription_id = session.subscription
    customer_id = session.customer

    # Fetch the Subscription to get price + period info
    sub = stripe.Subscription.retrieve(subscription_id)
    price_id = sub["items"]["data"][0]["price"]["id"]
    tier = _tier_for_price(price_id)

    # Read the stored tier BEFORE upserting (Task 1, spec §1, review r2): a
    # grandfathered Pro checking out Basic must not keep the old grant — this
    # is the third tier-mutating site, alongside handle_subscription_updated
    # and handle_subscription_deleted. Defaults to None (not "free", unlike
    # handle_subscription_updated below) because a missing row here means a
    # brand-new subscriber: nothing was ever grandfathered, so there is
    # nothing to compare against or clear.
    prev_res = (
        supabase.table("subscriptions")
        .select("tier, grandfathered_monthly_credits, grandfathered_until")
        .eq("user_id", user_id)
        .execute()
    )
    prev_tier = prev_res.data[0]["tier"] if prev_res.data else None
    prev_gf = prev_res.data[0].get("grandfathered_monthly_credits") if prev_res.data else None
    prev_gf_until = prev_res.data[0].get("grandfathered_until") if prev_res.data else None

    payload = {
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
    }
    if prev_tier is not None and prev_tier != tier:
        payload["grandfathered_monthly_credits"] = None
        payload["grandfathered_until"] = None  # hygiene: clear the expiry alongside the grant

    supabase.table("subscriptions").upsert(payload, on_conflict="user_id").execute()
    analytics_capture(
        user_id, "subscription_activated", {"stripe_price_id": price_id, "status": sub.status, "tier": tier}
    )
    try:
        analytics_identify(user_id, {"plan": tier})
    except Exception as e:
        logger.warning("analytics identify on subscription_activated failed: %s", e)

    from subscriptions.service import EntitlementsService, credits_enabled

    if credits_enabled():
        tier_res = supabase.table("tier_entitlements").select("monthly_credits").eq("tier", tier).execute()
        if not tier_res.data:
            logger.warning(
                "checkout wallet alignment: no tier_entitlements row for tier=%s — paying user %s gets no grant",
                tier,
                user_id,
            )
        tier_grant = tier_res.data[0]["monthly_credits"] if tier_res.data else 0
        # Same-tier re-checkout (past_due recovery, interval switch) must top
        # up at the EXISTING grandfathered grant, not the tier default — a
        # different tier already had its grandfathering nulled above, so the
        # tier default is correct there. Grandfathering expires with the
        # already-paid period (spec §1): a same-tier keep also requires
        # `grandfathered_until` to still be in the future — an expired (or
        # never-stamped) grant falls through to the tier default same as a
        # cleared one.
        # Reuse the single source of truth for grant precedence (see sweep.py's
        # rollover step, which calls this the same way) instead of
        # re-implementing the override>grandfathered>tier chain here. Old-tier
        # grandfathering only applies on a same-tier re-checkout — a tier
        # change already gets its grandfathering nulled above — so the
        # synthetic "sub" is empty (falls through to tier_grant) whenever
        # prev_tier != tier.
        sub_for_grant = (
            {"grandfathered_monthly_credits": prev_gf, "grandfathered_until": prev_gf_until}
            if prev_tier == tier
            else {}
        )
        grant = EntitlementsService._resolve_monthly_grant(sub_for_grant, None, tier_grant)
        _align_wallet_to_checkout(supabase, user_id, tier, event.id, grant)


def _handle_topup_completed(event, supabase) -> None:
    """One-time credit pack purchase (spec 2026-07-19 §3).

    Idempotent on topup:{session.id} — NOT the event id: delayed payment
    methods redeliver the same session as checkout.session.async_payment_succeeded
    under a DIFFERENT event id, which would double-grant on an event-keyed id.
    Failures raise so the webhook 500s and Stripe retries (grant is idempotent).

    Deliberately NOT gated on credits_enabled() (unlike the purchase endpoint):
    a session someone already PAID for must always grant — gating here would
    turn a flag flip into silent money-taken-no-credits.

    Licensing Phase B: metadata["target"] is either "user" (Phase A, default —
    legacy sessions with no "target" key at all fall through here too) or an
    org id. A non-"user" target hands off to `_handle_org_topup_grant`, which
    grants into the org's POOL wallet and re-checks cumulative activation
    (spec rule 3) instead of the personal-wallet path below.
    """
    session = event.data.object
    meta = _plain(session.metadata) or {}
    user_id = meta.get("user_id")
    pack_key = meta.get("pack_key")
    if not user_id or not pack_key:
        logger.error("topup: session %s missing metadata", getattr(session, "id", "?"))
        return
    # FAIL-CLOSED: grant only on exactly "paid". A MISSING field (malformed
    # object, future Stripe shape change) must not default to granting on a
    # money path; async methods deliver async_payment_succeeded later.
    if getattr(session, "payment_status", None) != "paid":
        return

    pack_res = supabase.table("credit_packs").select("credits, price_cents").eq("key", pack_key).execute()
    if not pack_res.data:
        logger.error("topup: unknown pack %r (session %s)", pack_key, session.id)
        return
    credits = pack_res.data[0]["credits"]
    price_cents = pack_res.data[0]["price_cents"]

    target = meta.get("target")
    if target and target != "user":
        _handle_org_topup_grant(
            supabase,
            user_id,
            target,
            pack_key,
            credits,
            price_cents,
            request_id=f"topup:{session.id}",
            event_name="topup_purchased",
        )
        return

    wallet_res = (
        supabase.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", user_id).execute()
    )
    if not wallet_res.data:
        # Nearly unreachable (signup trigger + migration backfill create user
        # wallets), but: INSERT-with-ignore, NOT upsert — an upsert's on-conflict
        # UPDATE would reset period_start/period_end if two deliveries raced,
        # and a fresh period_end=now() re-triggers a rollover grant. Standard
        # user-wallet seeding is fine here (user wallets roll over); Phase B
        # seat wallets must NOT use this path (spec §4).
        now = datetime.now(UTC)
        try:
            supabase.table("credit_wallets").insert(
                {
                    "owner_type": "user",
                    "owner_id": user_id,
                    "period_start": (now - relativedelta(months=1)).isoformat(),
                    "period_end": now.isoformat(),
                }
            ).execute()
        except Exception:
            pass  # duplicate insert lost a race — the re-read below wins either way
        wallet_res = (
            supabase.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", user_id).execute()
        )
        if not wallet_res.data:
            raise RuntimeError(f"topup: wallet create failed for user {user_id}")

    res = supabase.rpc(
        "grant_credits",
        {
            "p_wallet_id": wallet_res.data[0]["id"],
            "p_amount": credits,
            "p_kind": "purchase",
            "p_bucket": "reserve",
            "p_metadata": {"pack_key": pack_key, "price_cents": price_cents},
            "p_request_id": f"topup:{session.id}",
        },
    ).execute()
    # grant_credits reports {duplicate: bool}. Gate analytics on it: the
    # async_payment_succeeded redelivery replays this handler, and an
    # unconditional capture would double-count revenue in PostHog —
    # topup_purchased is the only pack-revenue signal.
    if not (isinstance(res.data, dict) and res.data.get("duplicate")):
        analytics_capture(
            user_id,
            "topup_purchased",
            {"pack": pack_key, "credits": credits, "usd": price_cents / 100, "target": "user"},
        )


def _handle_org_topup_grant(
    supabase,
    user_id: str,
    org_id: str,
    pack_key: str,
    credits: int,
    price_cents: int,
    *,
    request_id: str,
    event_name: str,
) -> None:
    """Grant one pack into an org's POOL wallet (Licensing Phase B, spec §4 +
    rule 3). Two callers: the pack-checkout branch of `_handle_topup_completed`
    (`topup:{session.id}`, `topup_purchased`) and the recurring top-up renewal
    in `handle_invoice_paid` (`orgtopup:{invoice.id}`, `org_topup_renewed`) —
    a renewal is ledger-indistinguishable from a pack purchase apart from its
    request id.

    Grants via `orgs.wallets.read_or_create_org_wallet`, NEVER the user-wallet
    seeding helper above, since pool wallets are NULL-period/reserve-only by
    construction (rule 1) — under the caller's idempotency key, so a
    redelivery converges identically.

    After the grant call (fresh OR duplicate — re-running this is harmless,
    the sum is unchanged either way), re-evaluates cumulative activation via
    the shared `orgs.wallets.maybe_activate_org` — the SAME implementation
    the dispersal sweep uses (subscriptions/sweep.py step 4), so the two
    activation paths can never drift. A 'pending' org whose lifetime paid-in
    SUM on this wallet reaches the effective minimum flips to 'active';
    already-active/suspended/archived orgs are never touched.

    No try/except anywhere: a failure must 500 the webhook so Stripe
    retries — every step here is idempotent (request-id'd grant, a re-run of
    the SUM, and a status flip that's a no-op once already 'active').
    """
    from orgs.wallets import maybe_activate_org, read_or_create_org_wallet

    wallet = read_or_create_org_wallet(supabase, org_id)
    wallet_id = wallet["id"]

    res = supabase.rpc(
        "grant_credits",
        {
            "p_wallet_id": wallet_id,
            "p_amount": credits,
            "p_kind": "purchase",
            "p_bucket": "reserve",
            "p_metadata": {"pack_key": pack_key, "price_cents": price_cents, "org_id": org_id},
            "p_request_id": request_id,
        },
    ).execute()
    # Gate analytics on the RPC's duplicate flag: a redelivery must not report
    # the revenue twice.
    if not (isinstance(res.data, dict) and res.data.get("duplicate")):
        analytics_capture(
            user_id,
            event_name,
            {"pack": pack_key, "credits": credits, "usd": price_cents / 100, "target": "org", "org_id": org_id},
        )

    maybe_activate_org(supabase, org_id, wallet_id)


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
    meta = _plain(sub.metadata) or {}
    if meta.get("kind") == "org_topup":
        # An org top-up subscription lives on the purchasing ADMIN's personal
        # Stripe customer but is NOT their plan. FIRST line on purpose: every
        # branch below (tier sync, grandfathering, upgrade top-up) would
        # rewrite that admin's personal subscription from the top-up's price.
        return
    user_id = meta.get("user_id")
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
    # flipping CREDITS_ENABLED off is a clean rollback to that behavior. The
    # grandfather-null below rides the SAME gate (unlike the checkout and
    # deletion sites, which null unconditionally): tier itself is only
    # written here under credits_enabled(), so nulling would be nulling a
    # column whose sibling "tier" write never happened.
    if credits_enabled():
        update["tier"] = new_tier
        if new_tier != prev_tier:
            # Tier movement ends grandfathering (Task 1, spec §1) — a stale
            # bundle from the old tier must not survive a plan switch.
            update["grandfathered_monthly_credits"] = None
            update["grandfathered_until"] = None  # hygiene: clear the expiry alongside the grant

    supabase.table("subscriptions").update(update).eq("user_id", user_id).execute()

    if credits_enabled() and TIER_RANK.get(new_tier, 0) > TIER_RANK.get(prev_tier, 0):
        # Upgrade: TOP UP the bundle to the new tier's grant — never additive.
        # Using the raw tier grant (not grandfather-aware) is safe ONLY because
        # this branch requires new_tier != prev_tier, which means the update
        # above already NULLed grandfathered_monthly_credits for this user —
        # there is no grandfathered value left to clobber. Do not "fix" this
        # to read _resolve_monthly_grant; a same-tier re-checkout (which DOES
        # need grandfather-awareness) is handled separately in
        # _align_wallet_to_checkout, not here.
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
        # Stripe redelivers events; a handler that failed AFTER granting would
        # re-grant on retry without the request-id key.
        _topup_bundle(
            supabase,
            user_id,
            new_grant,
            {"reason": "tier_upgrade_topup", "from": prev_tier, "to": new_tier},
            f"tier-upgrade:{event.id}",
        )


def handle_subscription_deleted(event, supabase) -> None:
    """Subscription truly ended (cancel-at-period-end fired, or hard delete).
    User loses Pro access; SP3 gates re-engage on next request."""
    sub = event.data.object
    meta = _plain(sub.metadata) or {}
    if meta.get("kind") == "org_topup":
        # Same isolation as handle_subscription_updated (FIRST line, above the
        # tier reset AND the final-overage billing), plus: release the org's
        # pointer so an admin can start a new top-up. Conditioned on the id
        # still matching, so a late delete of a REPLACED subscription can't
        # wipe the columns of the one that replaced it. Best-effort: the
        # personal subscriptions row must not be touched either way.
        org_id = meta.get("org_id")
        if org_id:
            try:
                supabase.table("organizations").update(
                    {"topup_stripe_subscription_id": None, "topup_admin_id": None}
                ).eq("id", org_id).eq("topup_stripe_subscription_id", sub.id).execute()
            except Exception:
                logger.exception("org top-up deleted: clearing columns failed org_id=%s", org_id)
        return
    user_id = meta.get("user_id")
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
            # Deletion always ends grandfathering (Task 1, spec §1) — free is
            # the only grant this user gets going forward.
            "grandfathered_monthly_credits": None,
            "grandfathered_until": None,  # hygiene: clear the expiry alongside the grant
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
    user_id = (_plain(sub.metadata) or {}).get("user_id")
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
    if _subscription_metadata(invoice).get("kind") == "org_topup":
        # This invoice belongs to an ORG top-up subscription, which lives on
        # the admin's PERSONAL customer — and the lookup below matches by
        # customer. Without this filter their personal credit-overage items
        # would attach to the org's top-up invoice (right card, wrong label).
        return
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


def handle_invoice_paid(event, supabase) -> None:
    """Recurring ORG top-up renewal (spec 2026-08-15 §4.3): every paid invoice
    of a top-up subscription grants that pack's credits into the org's POOL.

    Org-topup invoices are identified — and the org resolved — ONLY from
    `subscription_details.metadata`, never from
    organizations.topup_stripe_subscription_id (see `_subscription_metadata`).
    Any other invoice is somebody's personal subscription: no-op, the
    customer.subscription.* events already carry everything we need from it.

    The grant IS the pack fulfilment (`_handle_org_topup_grant`) under the
    request id `orgtopup:{invoice.id}`, which makes Stripe's redeliveries no-ops.

    No try/except: anything unresolvable on a kind-tagged invoice RAISES so the
    webhook 500s and Stripe retries. Money has already changed hands here — an
    ack we can't fulfil would silently swallow a paid month.
    """
    invoice = event.data.object
    meta = _subscription_metadata(invoice)
    if meta.get("kind") != "org_topup":
        return
    org_id = meta.get("org_id")
    if not org_id:
        raise RuntimeError(f"org top-up invoice {invoice.id}: no org_id in subscription metadata")

    # EVERY line's price, not line 0's: an invoice for this subscription can
    # also carry the admin's pending storage/credit-overage InvoiceItems, which
    # Stripe auto-collects onto the next invoice created for their customer
    # (handle_invoice_created's guard only stops the explicit straggler attach).
    # An ad-hoc overage line sorting first would miss the pack and 500 the
    # webhook forever, so the org's paid month would never grant.
    lines = _plain(getattr(invoice, "lines", None)) or {}
    price_ids = [pid for line in (lines.get("data") or []) if (pid := (line.get("price") or {}).get("id"))]
    # Deliberately NOT filtered on `active`: the customer has already been
    # charged, so a pack the operator has since retired must still fulfil the
    # renewals of subscriptions sold while it was live.
    pack_res = (
        supabase.table("credit_packs")
        .select("key, credits, price_cents")
        .in_("recurring_stripe_price_id", price_ids)
        .execute()
    )
    if not pack_res.data:
        raise RuntimeError(f"org top-up invoice {invoice.id}: no credit_packs row for recurring price {price_ids!r}")
    pack = pack_res.data[0]
    _handle_org_topup_grant(
        supabase,
        meta.get("purchased_by") or org_id,
        org_id,
        pack["key"],
        pack["credits"],
        pack["price_cents"],
        request_id=f"orgtopup:{invoice.id}",
        event_name="org_topup_renewed",
    )


# Dispatcher: maps Stripe event types to handler functions.
# Keys must match Stripe's exact event type strings.
HANDLERS = {
    "checkout.session.completed": handle_checkout_session_completed,
    "checkout.session.async_payment_succeeded": handle_checkout_session_completed,
    "customer.subscription.updated": handle_subscription_updated,
    "customer.subscription.deleted": handle_subscription_deleted,
    "invoice.payment_failed": handle_invoice_payment_failed,
    "invoice.created": handle_invoice_created,
    # Org top-up renewals only (every other invoice.paid is a no-op) — see the
    # module note on invoice.payment_succeeded for why personal renewals are
    # still handled through customer.subscription.updated instead.
    "invoice.paid": handle_invoice_paid,
}
