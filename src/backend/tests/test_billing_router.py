"""Endpoint tests for billing_router."""

from unittest.mock import MagicMock, patch

import stripe

from tests.conftest import TEST_USER_ID, MockQueryBuilder


class TestCreateCheckoutSession:
    def test_monthly_returns_checkout_url(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")

        # Reset stripe client init so it picks up the env var
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_test_123")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "monthly"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["url"] == "https://checkout.stripe.com/c/pay/cs_test_123"
        call_kwargs = m.call_args.kwargs
        assert call_kwargs["line_items"][0]["price"] == "price_monthly_test"
        assert call_kwargs["mode"] == "subscription"
        assert call_kwargs["metadata"]["user_id"] == TEST_USER_ID
        # Ensure subscription_data.metadata.user_id is also set (so subscription.updated events have it)
        assert call_kwargs["subscription_data"]["metadata"]["user_id"] == TEST_USER_ID

    def test_annual_uses_annual_price(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_test_annual")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "annual"})

        assert resp.status_code == 200, resp.text
        assert m.call_args.kwargs["line_items"][0]["price"] == "price_annual_test"

    def test_invalid_plan_returns_400(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")

        resp = client.post("/billing/create-checkout-session", json={"plan": "weekly"})
        assert resp.status_code == 400

    def test_missing_plan_field_returns_422(self, client, mock_supabase):
        """Pydantic validation rejects missing required field."""
        resp = client.post("/billing/create-checkout-session", json={})
        assert resp.status_code == 422


class TestCreatePortalSession:
    def test_with_stripe_customer_returns_url(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        # Override the conftest default to return a stripe_customer_id
        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(
                    data=[{"stripe_customer_id": "cus_test_123"}],
                    count=1,
                )
            return b

        mock_supabase.table.side_effect = _table

        fake_portal = MagicMock(url="https://billing.stripe.com/p/session/test_xyz")
        with patch("stripe.billing_portal.Session.create", return_value=fake_portal) as m:
            resp = client.post("/billing/create-portal-session")

        assert resp.status_code == 200, resp.text
        assert resp.json()["url"] == "https://billing.stripe.com/p/session/test_xyz"
        assert m.call_args.kwargs["customer"] == "cus_test_123"

    def test_no_stripe_customer_returns_404(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                # No stripe_customer_id (e.g., user has only an override)
                b.execute.return_value = MagicMock(
                    data=[{"stripe_customer_id": None}],
                    count=1,
                )
            return b

        mock_supabase.table.side_effect = _table

        resp = client.post("/billing/create-portal-session")
        assert resp.status_code == 404
        detail = resp.json()["detail"].lower()
        assert "support" in detail or "subscription" in detail


class TestWebhook:
    def test_invalid_signature_returns_400(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")

        from subscriptions import stripe_client

        with patch.object(
            stripe_client,
            "verify_webhook",
            side_effect=stripe.error.SignatureVerificationError("bad sig", "t=1,v1=bad"),
        ):
            resp = client.post(
                "/billing/webhook",
                content=b'{"id":"evt_1"}',
                headers={"stripe-signature": "t=1,v1=bad"},
            )
        assert resp.status_code == 400

    def test_valid_event_dispatches_to_handler(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")

        from subscriptions import stripe_client, stripe_events

        fake_event = MagicMock(id="evt_test_1", type="checkout.session.completed")
        fake_event.to_dict.return_value = {"id": "evt_test_1", "type": "checkout.session.completed"}

        handler_mock = MagicMock()
        with (
            patch.object(stripe_client, "verify_webhook", return_value=fake_event),
            patch.dict(stripe_events.HANDLERS, {"checkout.session.completed": handler_mock}),
        ):
            resp = client.post(
                "/billing/webhook",
                content=b'{"id":"evt_test_1"}',
                headers={"stripe-signature": "t=1,v1=sig"},
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("handled") is True
        handler_mock.assert_called_once()
        # First arg is the event, second is the supabase client
        assert handler_mock.call_args[0][0] is fake_event

    def test_duplicate_event_returns_duplicate_true(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")

        from subscriptions import stripe_client

        fake_event = MagicMock(id="evt_dup_1", type="checkout.session.completed")
        fake_event.to_dict.return_value = {"id": "evt_dup_1"}

        # Make the stripe_events INSERT raise (simulate duplicate-key conflict)
        original_side_effect = mock_supabase.table.side_effect

        def _table(name):
            if name == "stripe_events":
                b = MockQueryBuilder()
                b.insert.return_value.execute.side_effect = Exception("duplicate key value violates unique constraint")
                return b
            return original_side_effect(name)

        mock_supabase.table.side_effect = _table

        with patch.object(stripe_client, "verify_webhook", return_value=fake_event):
            resp = client.post(
                "/billing/webhook",
                content=b'{"id":"evt_dup_1"}',
                headers={"stripe-signature": "t=1,v1=sig"},
            )

        assert resp.status_code == 200
        assert resp.json().get("duplicate") is True

    def test_handler_exception_deletes_idempotency_row_and_returns_500(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")

        from subscriptions import stripe_client, stripe_events

        fake_event = MagicMock(id="evt_fail_1", type="checkout.session.completed")
        fake_event.to_dict.return_value = {"id": "evt_fail_1"}

        delete_mock = MagicMock()
        original_side_effect = mock_supabase.table.side_effect

        def _table(name):
            if name == "stripe_events":
                b = MockQueryBuilder()
                # INSERT succeeds (this is a fresh event)
                b.insert.return_value.execute.return_value = MagicMock(data=[{"event_id": "evt_fail_1"}])
                # DELETE chain
                b.delete.return_value.eq.return_value.execute = delete_mock
                return b
            return original_side_effect(name)

        mock_supabase.table.side_effect = _table

        with (
            patch.object(stripe_client, "verify_webhook", return_value=fake_event),
            patch.dict(
                stripe_events.HANDLERS, {"checkout.session.completed": MagicMock(side_effect=RuntimeError("DB down"))}
            ),
        ):
            resp = client.post(
                "/billing/webhook",
                content=b'{"id":"evt_fail_1"}',
                headers={"stripe-signature": "t=1,v1=sig"},
            )

        assert resp.status_code == 500
        # Idempotency row was deleted so Stripe retries
        delete_mock.assert_called_once()

    def test_unknown_event_type_acked_but_not_handled(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")

        from subscriptions import stripe_client

        fake_event = MagicMock(id="evt_unknown_1", type="customer.subscription.trial_will_end")
        fake_event.to_dict.return_value = {"id": "evt_unknown_1"}

        with patch.object(stripe_client, "verify_webhook", return_value=fake_event):
            resp = client.post(
                "/billing/webhook",
                content=b'{"id":"evt_unknown_1"}',
                headers={"stripe-signature": "t=1,v1=sig"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("handled") is False
