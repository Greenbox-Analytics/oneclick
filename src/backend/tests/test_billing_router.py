"""Endpoint tests for billing_router."""

from unittest.mock import MagicMock, patch

import pytest
import stripe

from tests.conftest import TEST_USER_EMAIL, TEST_USER_ID, MockQueryBuilder


def _subscriptions_table(mock_supabase, rows):
    """Route the endpoint's `subscriptions` read to `rows`; every other table stays empty."""

    def _table(name):
        b = MockQueryBuilder()
        if name == "subscriptions":
            b.execute.return_value = MagicMock(data=rows, count=len(rows))
        return b

    mock_supabase.table.side_effect = _table


def _checkout_env(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
    monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
    monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pro_monthly_test")
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")
    from subscriptions import stripe_client

    monkeypatch.setattr(stripe_client, "_initialized", False)


def _stripe_subscription(**overrides):
    """A real StripeObject, as `stripe.Subscription.retrieve` returns one (v15:
    not a dict — attribute access only)."""
    data = {
        "id": "sub_live",
        "object": "subscription",
        "status": "active",
        "cancel_at_period_end": False,
        "cancel_at": None,
        "current_period_start": 1700000000,
        "current_period_end": 1702592000,
        "canceled_at": None,
    }
    data.update(overrides)
    return stripe.Subscription.construct_from(data, "sk_test_x")


# What `sync_period_from_stripe` writes for the fixture above (see `_ts`).
_MIRRORED_PERIOD = {
    "current_period_start": "2023-11-14T22:13:20+00:00",
    "current_period_end": "2023-12-14T22:13:20+00:00",
}


class TestCreateCheckoutSession:
    def test_basic_monthly_returns_checkout_url(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")

        # Reset stripe client init so it picks up the env var
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_test_123")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "basic_monthly"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["url"] == "https://checkout.stripe.com/c/pay/cs_test_123"
        call_kwargs = m.call_args.kwargs
        assert call_kwargs["line_items"][0]["price"] == "price_monthly_test"
        assert call_kwargs["mode"] == "subscription"
        assert call_kwargs["metadata"]["user_id"] == TEST_USER_ID
        # Ensure subscription_data.metadata.user_id is also set (so subscription.updated events have it)
        assert call_kwargs["subscription_data"]["metadata"]["user_id"] == TEST_USER_ID

    def test_basic_annual_uses_annual_price(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_test_annual")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "basic_annual"})

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

    # --- one live personal subscription per user (2026-09-10) ---

    @pytest.mark.parametrize("status", ["active", "trialing", "past_due"])
    def test_refuses_a_second_subscription_while_one_is_live(self, client, mock_supabase, monkeypatch, status):
        """A Basic user clicking "Upgrade to Pro" used to get a SECOND Stripe
        subscription, and the webhook overwrote the stored id — the old one
        kept billing with nothing naming it. Any not-canceled status is live:
        a past-due user fixes their card in the portal, they don't buy another
        plan. Plan changes go through the Customer Portal instead."""
        _checkout_env(monkeypatch)
        _subscriptions_table(
            mock_supabase,
            [{"stripe_customer_id": "cus_1", "stripe_subscription_id": "sub_live", "status": status, "tier": "basic"}],
        )

        with (
            patch("stripe.checkout.Session.create") as create,
            patch("subscriptions.billing_router.analytics_capture") as capture,
        ):
            resp = client.post("/billing/create-checkout-session", json={"plan": "pro_monthly"})

        assert resp.status_code == 409, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "subscription_exists"
        assert detail["tier"] == "basic"
        assert "subscription" in detail["reason"].lower()
        create.assert_not_called()
        capture.assert_not_called()  # no checkout_started for a checkout that never existed

    def test_resubscribe_after_cancellation_reuses_the_stripe_customer(self, client, mock_supabase, monkeypatch):
        """handle_subscription_deleted keeps stripe_customer_id "for re-subscribe
        convenience" — honour it: the new subscription lands on the Customer
        that already carries this user's invoices (and any proration credit),
        not on a fresh Customer per checkout. Same block as the top-up paths."""
        _checkout_env(monkeypatch)
        _subscriptions_table(
            mock_supabase,
            [{"stripe_customer_id": "cus_1", "stripe_subscription_id": None, "status": "canceled", "tier": "free"}],
        )

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_again")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "basic_monthly"})

        assert resp.status_code == 200, resp.text
        kwargs = m.call_args.kwargs
        assert kwargs["customer"] == "cus_1"
        assert "customer_email" not in kwargs

    def test_a_canceled_row_that_still_names_a_subscription_is_not_live(self, client, mock_supabase, monkeypatch):
        """Live = id set AND status != canceled. A stale id on a canceled row
        (a lost `deleted` webhook) must not lock the user out of buying again."""
        _checkout_env(monkeypatch)
        _subscriptions_table(
            mock_supabase,
            [
                {
                    "stripe_customer_id": "cus_1",
                    "stripe_subscription_id": "sub_dead",
                    "status": "canceled",
                    "tier": "free",
                }
            ],
        )

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_x")
        with patch("stripe.checkout.Session.create", return_value=fake_session):
            resp = client.post("/billing/create-checkout-session", json={"plan": "basic_monthly"})

        assert resp.status_code == 200, resp.text

    def test_admin_granted_paid_tier_may_still_buy(self, client, mock_supabase, monkeypatch):
        """The conftest default row is a Pro tier with NO Stripe ids (an admin
        grant): nothing to conflict with, and no Customer to reuse."""
        _checkout_env(monkeypatch)

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_x")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "pro_monthly"})

        assert resp.status_code == 200, resp.text
        kwargs = m.call_args.kwargs
        assert kwargs["customer_email"] == TEST_USER_EMAIL
        assert "customer" not in kwargs

    def test_no_subscriptions_row_at_all_may_buy(self, client, mock_supabase, monkeypatch):
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [])

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_x")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "basic_annual"})

        assert resp.status_code == 200, resp.text
        assert m.call_args.kwargs["customer_email"] == TEST_USER_EMAIL


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
        # The Portal's "Return to Msanii" link lands on /profile with a signal, so
        # the page can refetch: a cancel or "Renew plan" made on the Portal home
        # races its webhook the same way Checkout does. No flow on the home session.
        assert m.call_args.kwargs["return_url"] == "http://localhost:8080/profile?portal=return"
        assert "flow_data" not in m.call_args.kwargs

    _LIVE_ROW = {
        "stripe_customer_id": "cus_1",
        "stripe_subscription_id": "sub_live",
        "status": "active",
        "cancel_at_period_end": False,
    }

    def test_empty_body_is_the_home_flow(self, client, mock_supabase, monkeypatch):
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [self._LIVE_ROW])
        fake_portal = MagicMock(url="https://billing.stripe.com/p/session/home")
        with (
            patch("stripe.billing_portal.Session.create", return_value=fake_portal) as m,
            patch("subscriptions.billing_router.analytics_capture") as cap,
        ):
            resp = client.post("/billing/create-portal-session", json={})

        assert resp.status_code == 200, resp.text
        assert "flow_data" not in m.call_args.kwargs
        cap.assert_called_once_with(TEST_USER_ID, "billing_portal_opened", {"flow": "home"})

    def test_cancel_flow_deep_links_into_the_portal(self, client, mock_supabase, monkeypatch):
        """Only a Portal FLOW can redirect on its own after a cancel: Stripe sends
        the customer to after_completion.redirect.return_url the moment they
        confirm, and to the session's return_url if they back out instead."""
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [self._LIVE_ROW])
        fake_portal = MagicMock(url="https://billing.stripe.com/p/session/cancel")
        with (
            patch("stripe.Subscription.retrieve", return_value=_stripe_subscription()) as get,
            patch("stripe.billing_portal.Session.create", return_value=fake_portal) as m,
            patch("subscriptions.billing_router.analytics_capture") as cap,
        ):
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["url"] == "https://billing.stripe.com/p/session/cancel"
        # Stripe, not the row, decides whether there is still something to cancel.
        get.assert_called_once_with("sub_live")
        kwargs = m.call_args.kwargs
        assert kwargs["customer"] == "cus_1"
        assert kwargs["return_url"] == "http://localhost:8080/profile?portal=return"
        assert kwargs["flow_data"] == {
            "type": "subscription_cancel",
            "subscription_cancel": {"subscription": "sub_live"},
            "after_completion": {
                "type": "redirect",
                "redirect": {"return_url": "http://localhost:8080/profile?portal=canceled"},
            },
        }
        cap.assert_called_once_with(TEST_USER_ID, "billing_portal_opened", {"flow": "subscription_cancel"})

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param({**_LIVE_ROW, "stripe_subscription_id": None}, id="no-subscription-id"),
            pytest.param({**_LIVE_ROW, "stripe_subscription_id": "sub_stale", "status": "canceled"}, id="canceled-row"),
        ],
    )
    def test_cancel_flow_needs_a_live_subscription(self, client, mock_supabase, monkeypatch, row):
        """Same "live" rule as create-checkout-session: an id AND a not-canceled
        status. An admin-granted tier (no id) has nothing at Stripe to cancel."""
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [row])
        with (
            patch("stripe.Subscription.retrieve") as get,
            patch("stripe.billing_portal.Session.create") as m,
        ):
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["code"] == "nothing_to_cancel"
        assert "cancel" in resp.json()["detail"]["reason"].lower()
        get.assert_not_called()
        m.assert_not_called()

    def test_cancel_flow_refuses_when_already_canceling(self, client, mock_supabase, monkeypatch):
        """A row that already knows needs no Stripe read."""
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [{**self._LIVE_ROW, "cancel_at_period_end": True}])
        with (
            patch("stripe.Subscription.retrieve") as get,
            patch("stripe.billing_portal.Session.create") as m,
        ):
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["code"] == "already_canceling"
        assert "already" in resp.json()["detail"]["reason"].lower()
        get.assert_not_called()
        m.assert_not_called()

    def _subscriptions_table_recording_writes(self, mock_supabase, row):
        """Like `_subscriptions_table`, with `update`/`eq` recorded for assertions."""
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=[row], count=1)
        b.update = MagicMock(return_value=b)
        b.eq = MagicMock(return_value=b)
        mock_supabase.table.side_effect = lambda name: b if name == "subscriptions" else MockQueryBuilder()
        return b

    @pytest.mark.parametrize(
        "live",
        [
            pytest.param({"cancel_at_period_end": True, "cancel_at": 1702592000}, id="classic-flag"),
            pytest.param({"cancel_at_period_end": False, "cancel_at": 1702592000}, id="flexible-cancel-at"),
        ],
    )
    def test_cancel_flow_heals_a_row_that_missed_the_cancel_webhook(self, client, mock_supabase, monkeypatch, live):
        """The row says "renews" but Stripe already holds the cancel — its
        webhook is in flight, or never arrived (a cancel made on the Portal
        home, in the Dashboard, or through this flow with the listener down).
        Stripe refuses a second cancel flow for such a subscription, so ask it
        first: 409 like a row that knew, and write the cancel onto the row so
        the refetch the 409 triggers flips the card to "Ends"."""
        _checkout_env(monkeypatch)
        b = self._subscriptions_table_recording_writes(mock_supabase, self._LIVE_ROW)
        sub = _stripe_subscription(**live, canceled_at=1702585200)
        with (
            patch("stripe.Subscription.retrieve", return_value=sub),
            patch("stripe.billing_portal.Session.create") as m,
        ):
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["code"] == "already_canceling"
        m.assert_not_called()
        b.update.assert_called_once_with(
            {"cancel_at_period_end": True, **_MIRRORED_PERIOD, "canceled_at": "2023-12-14T20:20:00+00:00"}
        )
        # Conditioned on the row still naming this subscription: a checkout
        # that replaced it between the read and the write must not be touched.
        filters = [c.args for c in b.eq.call_args_list]
        assert ("user_id", TEST_USER_ID) in filters
        assert ("stripe_subscription_id", "sub_live") in filters

    def test_cancel_flow_when_stripe_reports_canceled_is_nothing_to_cancel(self, client, mock_supabase, monkeypatch):
        """A missed `deleted` webhook: the row still says active. Freeing the
        plan is that handler's job (tier, grandfathering, analytics), not this
        endpoint's — refuse, write nothing, and say so in the log."""
        _checkout_env(monkeypatch)
        b = self._subscriptions_table_recording_writes(mock_supabase, self._LIVE_ROW)
        with (
            patch("stripe.Subscription.retrieve", return_value=_stripe_subscription(status="canceled")),
            patch("stripe.billing_portal.Session.create") as m,
        ):
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["code"] == "nothing_to_cancel"
        m.assert_not_called()
        b.update.assert_not_called()

    def test_cancel_flow_surfaces_an_unreadable_subscription_as_502(self, client, mock_supabase, monkeypatch):
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [self._LIVE_ROW])
        with (
            patch("stripe.Subscription.retrieve", side_effect=stripe.InvalidRequestError("No such subscription", "id")),
            patch("stripe.billing_portal.Session.create") as m,
        ):
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 502, resp.text
        assert resp.json()["detail"]["code"] == "portal_flow_unavailable"
        m.assert_not_called()

    def test_unknown_flow_is_rejected(self, client, mock_supabase, monkeypatch):
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [self._LIVE_ROW])
        with patch("stripe.billing_portal.Session.create") as m:
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_update"})

        assert resp.status_code == 422, resp.text
        m.assert_not_called()

    def test_cancel_flow_surfaces_a_stripe_refusal_as_502(self, client, mock_supabase, monkeypatch):
        """Stripe doesn't document what a Portal configuration with cancellation
        switched off does to a subscription_cancel flow. Whatever it is, ops
        must see a named error, and the frontend falls back to the Portal home."""
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [self._LIVE_ROW])
        with (
            patch("stripe.Subscription.retrieve", return_value=_stripe_subscription()),
            patch(
                "stripe.billing_portal.Session.create",
                side_effect=stripe.InvalidRequestError("Cancellation is disabled", "flow_data"),
            ),
        ):
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 502, resp.text
        assert resp.json()["detail"]["code"] == "portal_flow_unavailable"

    def test_cancel_flow_without_a_customer_is_still_404(self, client, mock_supabase, monkeypatch):
        _checkout_env(monkeypatch)
        _subscriptions_table(mock_supabase, [{**self._LIVE_ROW, "stripe_customer_id": None}])
        with patch("stripe.billing_portal.Session.create") as m:
            resp = client.post("/billing/create-portal-session", json={"flow": "subscription_cancel"})

        assert resp.status_code == 404, resp.text
        m.assert_not_called()

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


class TestSyncSubscription:
    """Every Portal return calls this so the plan card flips within one round
    trip instead of waiting on the `.updated` webhook (which still lands and
    writes the same truth)."""

    _LIVE_ROW = {"stripe_subscription_id": "sub_live", "status": "active"}

    def _recording_table(self, mock_supabase, row):
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=[row] if row else [], count=1 if row else 0)
        b.update = MagicMock(return_value=b)
        b.eq = MagicMock(return_value=b)
        mock_supabase.table.side_effect = lambda name: b if name == "subscriptions" else MockQueryBuilder()
        return b

    @pytest.mark.parametrize(
        ("live", "expected_flag", "expected_canceled_at"),
        [
            pytest.param(
                {"cancel_at_period_end": True, "cancel_at": 1702592000, "canceled_at": 1702585200},
                True,
                "2023-12-14T20:20:00+00:00",
                id="cancel-scheduled-at-stripe",
            ),
            pytest.param({}, False, None, id="renewed-at-stripe"),
        ],
    )
    def test_mirrors_the_period_and_cancel_fields(
        self, client, mock_supabase, monkeypatch, live, expected_flag, expected_canceled_at
    ):
        _checkout_env(monkeypatch)
        b = self._recording_table(mock_supabase, self._LIVE_ROW)
        with patch("stripe.Subscription.retrieve", return_value=_stripe_subscription(**live)) as get:
            resp = client.post("/billing/sync-subscription")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"synced": True}
        get.assert_called_once_with("sub_live")
        b.update.assert_called_once_with(
            {"cancel_at_period_end": expected_flag, **_MIRRORED_PERIOD, "canceled_at": expected_canceled_at}
        )
        filters = [c.args for c in b.eq.call_args_list]
        assert ("user_id", TEST_USER_ID) in filters
        assert ("stripe_subscription_id", "sub_live") in filters

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param(None, id="no-row"),
            pytest.param({"stripe_subscription_id": None, "status": "active"}, id="admin-granted-no-id"),
            pytest.param({"stripe_subscription_id": "sub_old", "status": "canceled"}, id="canceled-row"),
        ],
    )
    def test_nothing_live_to_mirror(self, client, mock_supabase, monkeypatch, row):
        _checkout_env(monkeypatch)
        b = self._recording_table(mock_supabase, row)
        with patch("stripe.Subscription.retrieve") as get:
            resp = client.post("/billing/sync-subscription")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"synced": False}
        get.assert_not_called()
        b.update.assert_not_called()

    def test_a_subscription_stripe_reports_canceled_is_left_to_the_deleted_handler(
        self, client, mock_supabase, monkeypatch
    ):
        _checkout_env(monkeypatch)
        b = self._recording_table(mock_supabase, self._LIVE_ROW)
        with patch("stripe.Subscription.retrieve", return_value=_stripe_subscription(status="canceled")):
            resp = client.post("/billing/sync-subscription")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"synced": False}
        b.update.assert_not_called()

    def test_stripe_unreachable_is_a_502_so_the_caller_waits_on_the_webhook(self, client, mock_supabase, monkeypatch):
        _checkout_env(monkeypatch)
        b = self._recording_table(mock_supabase, self._LIVE_ROW)
        with patch("stripe.Subscription.retrieve", side_effect=stripe.APIConnectionError("down")):
            resp = client.post("/billing/sync-subscription")

        assert resp.status_code == 502, resp.text
        assert resp.json()["detail"]["code"] == "stripe_unavailable"
        b.update.assert_not_called()

    def test_unauthenticated_returns_401(self, mock_supabase):
        from fastapi.testclient import TestClient

        import main

        original = dict(main.app.dependency_overrides)
        main.app.dependency_overrides.clear()
        try:
            with TestClient(main.app) as tc:
                resp = tc.post("/billing/sync-subscription")
            assert resp.status_code == 401, resp.text
        finally:
            main.app.dependency_overrides.update(original)


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

    def test_transient_idempotency_insert_failure_returns_500_so_stripe_retries(
        self, client, mock_supabase, monkeypatch
    ):
        """FIX: a NON-duplicate insert failure (transient DB error) must NOT be
        acked as a duplicate — a 200 would make Stripe stop retrying and
        permanently drop the event (e.g. a paid checkout.session.completed)."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")

        from subscriptions import stripe_client

        fake_event = MagicMock(id="evt_transient_1", type="checkout.session.completed")
        fake_event.to_dict.return_value = {"id": "evt_transient_1"}

        original_side_effect = mock_supabase.table.side_effect

        def _table(name):
            if name == "stripe_events":
                b = MockQueryBuilder()
                b.insert.return_value.execute.side_effect = Exception("connection reset by peer")
                return b
            return original_side_effect(name)

        mock_supabase.table.side_effect = _table

        with patch.object(stripe_client, "verify_webhook", return_value=fake_event):
            resp = client.post(
                "/billing/webhook",
                content=b'{"id":"evt_transient_1"}',
                headers={"stripe-signature": "t=1,v1=sig"},
            )

        assert resp.status_code == 500
        assert "duplicate" not in resp.json().get("detail", "").lower()

    def test_postgrest_error_code_23505_still_acked_as_duplicate(self, client, mock_supabase, monkeypatch):
        """The Supabase client surfaces unique violations as APIError-like
        exceptions carrying code='23505' — that shape must still be acked."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test_dummy")

        from subscriptions import stripe_client

        fake_event = MagicMock(id="evt_dup_2", type="checkout.session.completed")
        fake_event.to_dict.return_value = {"id": "evt_dup_2"}

        class FakeApiError(Exception):
            code = "23505"

        original_side_effect = mock_supabase.table.side_effect

        def _table(name):
            if name == "stripe_events":
                b = MockQueryBuilder()
                b.insert.return_value.execute.side_effect = FakeApiError("conflict")
                return b
            return original_side_effect(name)

        mock_supabase.table.side_effect = _table

        with patch.object(stripe_client, "verify_webhook", return_value=fake_event):
            resp = client.post(
                "/billing/webhook",
                content=b'{"id":"evt_dup_2"}',
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
