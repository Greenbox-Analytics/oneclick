"""Tests for oneclick/royalties/paypal_client.py — no network calls.

Covers:
  1. format_amount money correctness (ROUND_HALF_UP, zero-decimal currencies,
     non-positive rejection, float-repr trap)
  2. Access-token caching (reuse, expiry, env flip, missing creds)
  3. PayPal error-body parsing (issue code extraction)
"""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import paypal_client

# ---------------------------------------------------------------------------
# format_amount
# ---------------------------------------------------------------------------


class TestFormatAmount:
    @pytest.mark.parametrize(
        ("amount", "currency", "expected"),
        [
            (10, "USD", "10.00"),
            (10.005, "USD", "10.01"),  # ROUND_HALF_UP
            (10.004, "USD", "10.00"),
            (0.01, "USD", "0.01"),
            (1234.56, "EUR", "1234.56"),
            (1234.5, "JPY", "1235"),  # zero-decimal, rounded
            (1234.4, "JPY", "1234"),
            (500, "HUF", "500"),
            (99.999, "TWD", "100"),
            (10, "usd", "10.00"),  # lowercase currency
        ],
    )
    def test_formatting(self, amount, currency, expected):
        assert paypal_client.format_amount(amount, currency) == expected

    def test_float_repr_trap(self):
        # Decimal(str(x)) must be used — Decimal(0.1) would carry float noise.
        assert paypal_client.format_amount(0.1 + 0.2, "USD") == "0.30"

    @pytest.mark.parametrize("amount", [0, -5, 0.001])  # 0.001 rounds to 0.00
    def test_non_positive_rejected(self, amount):
        with pytest.raises(ValueError):
            paypal_client.format_amount(amount, "USD")


# ---------------------------------------------------------------------------
# Token caching
# ---------------------------------------------------------------------------


def _fresh_cache():
    paypal_client._token_cache.update({"access_token": None, "expires_at": 0.0, "env": None})


def _mock_httpx_client(response_json, status_code=200):
    """Return a MagicMock usable as `with httpx.Client(...) as client:`."""
    resp = MagicMock(status_code=status_code)
    resp.json.return_value = response_json
    resp.text = str(response_json)
    client = MagicMock()
    client.post.return_value = resp
    client.request.return_value = resp
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=client)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, client


class TestAccessToken:
    def setup_method(self):
        _fresh_cache()

    def teardown_method(self):
        _fresh_cache()

    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)
        monkeypatch.delenv("PAYPAL_CLIENT_SECRET", raising=False)
        with pytest.raises(RuntimeError):
            paypal_client._get_access_token()

    def test_token_cached_across_calls(self, monkeypatch):
        monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
        monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sec")
        monkeypatch.setenv("PAYPAL_ENV", "sandbox")
        ctx, client = _mock_httpx_client({"access_token": "tok-1", "expires_in": 3600})
        with patch.object(paypal_client.httpx, "Client", return_value=ctx):
            assert paypal_client._get_access_token() == "tok-1"
            assert paypal_client._get_access_token() == "tok-1"
        assert client.post.call_count == 1

    def test_expired_token_refetched(self, monkeypatch):
        monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
        monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sec")
        paypal_client._token_cache.update({"access_token": "old", "expires_at": 0.0, "env": "sandbox"})
        ctx, client = _mock_httpx_client({"access_token": "tok-2", "expires_in": 3600})
        with patch.object(paypal_client.httpx, "Client", return_value=ctx):
            assert paypal_client._get_access_token() == "tok-2"
        assert client.post.call_count == 1

    def test_env_flip_invalidates_cache(self, monkeypatch):
        monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
        monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sec")
        monkeypatch.setenv("PAYPAL_ENV", "live")
        # Cache holds a perfectly valid sandbox token…
        paypal_client._token_cache.update({"access_token": "sandbox-tok", "expires_at": 9999999999.0, "env": "sandbox"})
        ctx, client = _mock_httpx_client({"access_token": "live-tok", "expires_in": 3600})
        with patch.object(paypal_client.httpx, "Client", return_value=ctx):
            assert paypal_client._get_access_token() == "live-tok"
        # …and the request went to the live host.
        assert "api-m.paypal.com" in client.post.call_args[0][0]

    def test_token_endpoint_error_raises_paypal_error(self, monkeypatch):
        monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
        monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sec")
        ctx, _ = _mock_httpx_client({"error": "invalid_client"}, status_code=401)
        with patch.object(paypal_client.httpx, "Client", return_value=ctx), pytest.raises(paypal_client.PayPalError):
            paypal_client._get_access_token()


# ---------------------------------------------------------------------------
# Error-body parsing on API requests
# ---------------------------------------------------------------------------


class TestRequestErrors:
    def setup_method(self):
        _fresh_cache()

    def teardown_method(self):
        _fresh_cache()

    def test_issue_code_extracted_from_422(self, monkeypatch):
        monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
        monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sec")
        monkeypatch.setenv("PAYPAL_ENV", "sandbox")
        paypal_client._token_cache.update({"access_token": "tok", "expires_at": 9999999999.0, "env": "sandbox"})
        body = {
            "name": "UNPROCESSABLE_ENTITY",
            "message": "The requested action could not be performed.",
            "details": [{"issue": "ORDER_ALREADY_CAPTURED", "description": "Order already captured."}],
        }
        ctx, _ = _mock_httpx_client(body, status_code=422)
        with (
            patch.object(paypal_client.httpx, "Client", return_value=ctx),
            pytest.raises(paypal_client.PayPalError) as excinfo,
        ):
            paypal_client.capture_order("ORDER-1")
        assert excinfo.value.issue == "ORDER_ALREADY_CAPTURED"
        assert excinfo.value.status_code == 422

    def test_create_order_payload_shape(self, monkeypatch):
        monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
        monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "sec")
        monkeypatch.setenv("PAYPAL_ENV", "sandbox")
        paypal_client._token_cache.update({"access_token": "tok", "expires_at": 9999999999.0, "env": "sandbox"})
        ctx, client = _mock_httpx_client({"id": "ORDER-9", "status": "CREATED"})
        with patch.object(paypal_client.httpx, "Client", return_value=ctx):
            result = paypal_client.create_order(
                payee_email="payee@example.com",
                amount_value="150.00",
                currency="usd",
                reference_id="payout-1",
                description="x" * 200,
            )
        assert result["id"] == "ORDER-9"
        sent = client.request.call_args.kwargs["json"]
        assert sent["intent"] == "CAPTURE"
        unit = sent["purchase_units"][0]
        assert unit["payee"]["email_address"] == "payee@example.com"
        assert unit["amount"] == {"currency_code": "USD", "value": "150.00"}
        assert unit["reference_id"] == "payout-1"
        assert len(unit["description"]) == 127  # truncated to PayPal's cap
