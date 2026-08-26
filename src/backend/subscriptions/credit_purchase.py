"""Pricing + Stripe line-item construction for one-time credit purchases.

Pure functions, no I/O — the money math lives here so it is testable without a
Stripe key or a database, and so exactly one implementation answers "what does
N credits cost?".

Two things are sold through `POST /billing/create-topup-session`:

  * a PACK — a preset bundle whose price is the `credit_packs.price_cents`
    column (volume-discounted: $0.020/cr at 500 down to $0.012/cr at 50,000);
  * a CUSTOM amount — any credit count in [MIN_CUSTOM_CREDITS,
    MAX_CUSTOM_CREDITS], priced at the flat list rate below.

Custom amounts are priced at `ai_pricing.overage_usd_per_credit()` — the SAME
dial that prices pay-per-use overage — deliberately, not coincidentally: buying
a credit up front and burning one on overage must never cost different amounts,
or the cheaper path becomes an arbitrage. It also keeps every volume discount
inside the packs, so a pack is always at least as good as the equivalent custom
amount and the preset stays the better deal.

The CLIENT NEVER SENDS A PRICE. It sends a credit COUNT and the server derives
the cents; `price_cents_for_credits` is the only place that conversion happens.
"""

import os

from subscriptions.ai_pricing import overage_usd_per_credit

# Floor is a real transaction minimum, not a Stripe one (Stripe's is $0.50):
# below ~$5 the card fee eats the margin. Ceiling is a fat-finger/chargeback
# guard — someone who genuinely wants more than $2,000 of credits at once is a
# conversation with sales, not a self-serve checkout.
MIN_CUSTOM_CREDITS = 250  # $5.00 at the $0.02 list rate
MAX_CUSTOM_CREDITS = 100_000  # $2,000.00 at the $0.02 list rate


def per_credit_cents() -> float:
    """List price of one credit, in cents.

    Fractional on purpose — CREDIT_OVERAGE_USD is a dial, and a rate like
    $0.025 is 2.5 cents. Rounding to whole cents HERE and multiplying would
    undercharge by half a cent on every credit; `price_cents_for_credits`
    rounds once, on the total, instead.

    Raises ValueError on a non-positive rate: a 0 would mint free credits
    through checkout, so this fails loudly rather than charging $0.
    """
    usd = overage_usd_per_credit()
    if usd <= 0:
        raise ValueError(f"CREDIT_OVERAGE_USD must be positive, got {usd!r}")
    return round(usd * 100, 6)


def price_cents_for_credits(credits: int) -> int:
    """USD cents charged for a CUSTOM purchase of `credits` credits.

    Rounds ONCE, on the total, to whole cents — Stripe's `unit_amount` is an
    integer and a float would be silently truncated, undercharging by up to a
    cent per purchase. The round(..., 6) before it kills IEEE754 dust the same
    way `ai_pricing.credits_for_cost` does.
    """
    if credits <= 0:
        raise ValueError(f"credits must be positive, got {credits!r}")
    cents = int(round(round(credits * per_credit_cents(), 6)))
    if cents <= 0:
        raise ValueError(f"computed a non-chargeable price for {credits} credits")
    return cents


def validate_custom_credits(credits: object) -> int:
    """Coerce + bound-check a client-supplied custom credit count.

    Raises ValueError with user-facing copy — the caller surfaces it as a 400.
    `bool` is rejected explicitly because it is an `int` subclass in Python and
    `True` would otherwise sail through as 1 credit.
    """
    if isinstance(credits, bool) or not isinstance(credits, int):
        raise ValueError("Enter a whole number of credits.")
    if credits < MIN_CUSTOM_CREDITS:
        raise ValueError(f"The smallest credit purchase is {MIN_CUSTOM_CREDITS:,} credits.")
    if credits > MAX_CUSTOM_CREDITS:
        raise ValueError(f"The largest credit purchase is {MAX_CUSTOM_CREDITS:,} credits. Contact support for more.")
    return credits


def credits_line_item(*, price_cents: int, name: str) -> dict:
    """An ad-hoc Stripe Checkout line item for `price_cents` of credits.

    Uses `price_data` rather than a pre-created Price so a pack is sellable the
    moment `credit_packs.active` flips — no Stripe dashboard step, and custom
    amounts (arbitrary by definition) have no Price to point at anyway.

    Set STRIPE_CREDITS_PRODUCT_ID to an existing Stripe Product id to keep the
    catalog tidy: without it Stripe creates one ad-hoc Product per checkout,
    which is harmless but noisy in the dashboard over time.
    """
    if price_cents <= 0:
        raise ValueError(f"price_cents must be positive, got {price_cents!r}")
    price_data: dict = {"currency": "usd", "unit_amount": price_cents}
    product_id = (os.getenv("STRIPE_CREDITS_PRODUCT_ID") or "").strip()
    if product_id:
        price_data["product"] = product_id
    else:
        price_data["product_data"] = {"name": name}
    return {"price_data": price_data, "quantity": 1}


def custom_config() -> dict:
    """Bounds + unit price for the custom-amount UI (camelCase, API-shaped).

    Served from GET /billing/credit-packs so the picker can quote a live total
    while typing; the server still recomputes authoritatively at checkout.
    """
    return {
        "minCredits": MIN_CUSTOM_CREDITS,
        "maxCredits": MAX_CUSTOM_CREDITS,
        "perCreditCents": per_credit_cents(),
    }
