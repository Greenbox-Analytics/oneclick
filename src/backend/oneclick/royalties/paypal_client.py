"""PayPal Orders v2 client for royalty payout checkout (client-credentials flow).

The platform's PayPal REST app brokers a checkout order whose payee is the
collaborator's email — the paying user approves with their own PayPal account,
so money flows payer -> collaborator directly and never touches a platform
account.

Environment:
  PAYPAL_ENV            "sandbox" (default) | "live"
  PAYPAL_CLIENT_ID      REST app client id
  PAYPAL_CLIENT_SECRET  REST app secret (backend only)
"""

import base64
import logging
import os
import threading
import time
from decimal import ROUND_HALF_UP, Decimal

import httpx

logger = logging.getLogger(__name__)

# Currencies PayPal accepts for payments. BRL and MYR are deliberately
# excluded — PayPal restricts them to in-country accounts, which would fail
# opaquely for most of our users.
PAYPAL_SUPPORTED_CURRENCIES = frozenset(
    {
        "AUD",
        "CAD",
        "CHF",
        "CZK",
        "DKK",
        "EUR",
        "GBP",
        "HKD",
        "HUF",
        "ILS",
        "JPY",
        "MXN",
        "NOK",
        "NZD",
        "PHP",
        "PLN",
        "SEK",
        "SGD",
        "THB",
        "TWD",
        "USD",
    }
)

# PayPal rejects decimal amounts in these currencies.
ZERO_DECIMAL_CURRENCIES = frozenset({"HUF", "JPY", "TWD"})

# Cached access token shared across requests; keyed by env so a sandbox->live
# flip never reuses a stale token.
_token_cache: dict = {"access_token": None, "expires_at": 0.0, "env": None}
_token_lock = threading.Lock()


class PayPalError(Exception):
    """A PayPal API call failed. `issue` carries the first issue code from the
    error body (e.g. "ORDER_ALREADY_CAPTURED") when PayPal provides one."""

    def __init__(self, message: str, status_code: int | None = None, issue: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.issue = issue


def _env() -> str:
    return (os.getenv("PAYPAL_ENV") or "sandbox").strip().lower()


def _base_url() -> str:
    return "https://api-m.paypal.com" if _env() == "live" else "https://api-m.sandbox.paypal.com"


def _get_access_token() -> str:
    """Return a cached OAuth2 access token, refreshing if expired or unset."""
    client_id = os.getenv("PAYPAL_CLIENT_ID")
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET must be set")

    env = _env()
    now = time.time()
    if _token_cache["access_token"] and _token_cache["env"] == env and _token_cache["expires_at"] - 60 > now:
        return _token_cache["access_token"]

    with _token_lock:
        # Re-check after acquiring the lock — another thread may have refreshed.
        if (
            _token_cache["access_token"]
            and _token_cache["env"] == env
            and _token_cache["expires_at"] - 60 > time.time()
        ):
            return _token_cache["access_token"]

        auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                f"{_base_url()}/v1/oauth2/token",
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials"},
            )
        if resp.status_code >= 400:
            logger.error("PayPal token request failed (%s): %s", resp.status_code, resp.text)
            raise PayPalError("Could not authenticate with PayPal", status_code=resp.status_code)
        data = resp.json()

        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + int(data.get("expires_in", 3600))
        _token_cache["env"] = env
        return _token_cache["access_token"]


def format_amount(amount: float, currency: str) -> str:
    """Format a payout amount as the string PayPal expects.

    Two decimals with ROUND_HALF_UP for most currencies, whole units for
    zero-decimal currencies (JPY/HUF/TWD). Raises ValueError for amounts that
    round to zero or below — never create a zero-value order.
    """
    quantum = Decimal("1") if currency.upper() in ZERO_DECIMAL_CURRENCIES else Decimal("0.01")
    value = Decimal(str(amount)).quantize(quantum, rounding=ROUND_HALF_UP)
    if value <= 0:
        raise ValueError(f"Payout amount must be positive, got {amount}")
    return str(value)


def _request(method: str, path: str, json_body: dict | None = None) -> dict:
    token = _get_access_token()
    with httpx.Client(timeout=20.0) as client:
        resp = client.request(
            method,
            f"{_base_url()}{path}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=json_body,
        )
    if resp.status_code >= 400:
        issue = None
        message = f"PayPal API error ({resp.status_code})"
        try:
            body = resp.json()
            details = body.get("details") or []
            if details and isinstance(details[0], dict):
                issue = details[0].get("issue")
            message = body.get("message") or message
        except ValueError:
            pass
        logger.error("PayPal %s %s failed (%s): %s", method, path, resp.status_code, resp.text)
        raise PayPalError(message, status_code=resp.status_code, issue=issue)
    return resp.json()


def create_order(
    payee_email: str,
    amount_value: str,
    currency: str,
    reference_id: str,
    description: str | None = None,
) -> dict:
    """Create a CAPTURE-intent order paying `payee_email` directly."""
    purchase_unit: dict = {
        "reference_id": reference_id,
        "amount": {"currency_code": currency.upper(), "value": amount_value},
        "payee": {"email_address": payee_email},
    }
    if description:
        # PayPal caps purchase unit descriptions at 127 chars.
        purchase_unit["description"] = description[:127]
    return _request("POST", "/v2/checkout/orders", {"intent": "CAPTURE", "purchase_units": [purchase_unit]})


def capture_order(order_id: str) -> dict:
    """Capture an approved order. Raises PayPalError with issue
    "ORDER_ALREADY_CAPTURED" if it was captured previously."""
    return _request("POST", f"/v2/checkout/orders/{order_id}/capture")


def get_order(order_id: str) -> dict:
    """Fetch the current state of an order."""
    return _request("GET", f"/v2/checkout/orders/{order_id}")
