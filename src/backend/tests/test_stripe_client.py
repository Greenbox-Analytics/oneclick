"""Unit tests for stripe_client module."""

from unittest.mock import MagicMock, patch

import pytest


class TestGetStripe:
    def test_returns_stripe_module(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        from subscriptions import stripe_client

        # Reset module state so the lazy init fires
        monkeypatch.setattr(stripe_client, "_initialized", False)

        result = stripe_client.get_stripe()
        assert hasattr(result, "Webhook")
        assert hasattr(result, "checkout")

    def test_lazy_init_only_runs_once(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_first")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        stripe_client.get_stripe()
        # Change env var; should NOT re-read because already initialized
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_second")
        stripe_client.get_stripe()

        # api_key reflects the FIRST set value
        import stripe

        assert stripe.api_key == "sk_test_first"


class TestVerifyWebhook:
    def test_valid_signature_returns_event(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        fake_event = MagicMock(id="evt_123", type="checkout.session.completed")
        with patch("stripe.Webhook.construct_event", return_value=fake_event) as m:
            result = stripe_client.verify_webhook(b'{"id":"evt_123"}', "t=1,v1=sig")
            assert result.id == "evt_123"
            m.assert_called_once_with(b'{"id":"evt_123"}', "t=1,v1=sig", "whsec_test_dummy")

    def test_invalid_signature_raises(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)
        import stripe

        with (
            patch(
                "stripe.Webhook.construct_event",
                side_effect=stripe.error.SignatureVerificationError("bad sig", "t=1,v1=bad"),
            ),
            pytest.raises(stripe.error.SignatureVerificationError),
        ):
            stripe_client.verify_webhook(b"{}", "t=1,v1=bad")
