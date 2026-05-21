"""Stripe SDK initialization and webhook signature verification.

Singleton pattern: `get_stripe()` lazy-initializes on first call (sets api_key
from env var) and caches the initialized state in a module-level flag. Tests
can reset `_initialized` to False to re-run init with a different env value.
"""

import os

import stripe

_initialized = False


def get_stripe():
    """Returns the configured stripe module. Lazy-initializes on first call.

    Raises KeyError on first call if STRIPE_SECRET_KEY is unset.
    """
    global _initialized
    if not _initialized:
        api_key = os.environ["STRIPE_SECRET_KEY"]  # raises KeyError if unset
        stripe.api_key = api_key
        stripe.api_version = "2024-06-20"  # pin to a known-good API version
        _initialized = True
    return stripe


def verify_webhook(payload: bytes, sig_header: str) -> stripe.Event:
    """Verify Stripe webhook signature. Returns parsed Event; raises on invalid.

    Raises:
        stripe.error.SignatureVerificationError: if sig_header is missing/invalid
        ValueError: if payload isn't valid JSON
        KeyError: if STRIPE_WEBHOOK_SECRET is unset
    """
    secret = os.environ["STRIPE_WEBHOOK_SECRET"]
    return get_stripe().Webhook.construct_event(payload, sig_header, secret)
