"""Unit tests for stripe_events handlers."""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID


def _mock_supabase():
    """Return a mock supabase client whose .table().upsert/update/eq chain works."""
    sb = MagicMock()
    return sb


def _checkout_session_event(user_id=TEST_USER_ID, subscription_id="sub_123", customer_id="cus_123"):
    """Build a mock checkout.session.completed event."""
    e = MagicMock()
    e.id = "evt_checkout_1"
    e.type = "checkout.session.completed"
    e.data.object.metadata = {"user_id": user_id} if user_id else {}
    e.data.object.subscription = subscription_id
    e.data.object.customer = customer_id
    return e


def _subscription_event(
    event_type,
    user_id=TEST_USER_ID,
    status="active",
    cancel_at_period_end=False,
    price_id="price_monthly_123",
    current_period_start=1700000000,
    current_period_end=1702592000,
):
    """Build a mock customer.subscription.* event."""
    e = MagicMock()
    e.id = f"evt_{event_type.replace('.', '_')}_1"
    e.type = event_type
    obj = e.data.object
    obj.metadata = {"user_id": user_id} if user_id else {}
    obj.status = status
    obj.cancel_at_period_end = cancel_at_period_end
    obj.canceled_at = None if not cancel_at_period_end else 1700100000
    obj.current_period_start = current_period_start
    obj.current_period_end = current_period_end
    obj.__getitem__ = lambda self, k: {"items": {"data": [{"price": {"id": price_id}}]}}[k] if k == "items" else None
    return e


class TestHandleCheckoutSessionCompleted:
    def test_upserts_subscription_with_tier_pro(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _checkout_session_event()

        fake_sub = MagicMock(
            status="active",
            cancel_at_period_end=False,
            canceled_at=None,
            current_period_start=1700000000,
            current_period_end=1702592000,
        )
        fake_sub.__getitem__ = lambda self, k: (
            {"items": {"data": [{"price": {"id": "price_monthly_123"}}]}}[k] if k == "items" else None
        )
        with patch("stripe.Subscription.retrieve", return_value=fake_sub):
            stripe_events.handle_checkout_session_completed(event, sb)

        sb.table.assert_any_call("subscriptions")
        upsert_call = sb.table("subscriptions").upsert.call_args
        assert upsert_call is not None
        payload = upsert_call[0][0]
        assert payload["user_id"] == TEST_USER_ID
        assert payload["tier"] == "pro"
        assert payload["stripe_subscription_id"] == "sub_123"
        assert payload["stripe_customer_id"] == "cus_123"
        assert payload["stripe_price_id"] == "price_monthly_123"
        # on_conflict kwarg
        assert upsert_call.kwargs.get("on_conflict") == "user_id"

    def test_no_op_when_user_id_missing(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _checkout_session_event(user_id=None)

        stripe_events.handle_checkout_session_completed(event, sb)
        sb.table.assert_not_called()


class TestHandleSubscriptionUpdated:
    def test_syncs_status_and_period_without_tier(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.updated", status="active", cancel_at_period_end=True)

        stripe_events.handle_subscription_updated(event, sb)

        sb.table.assert_any_call("subscriptions")
        update_call = sb.table("subscriptions").update.call_args
        payload = update_call[0][0]
        assert payload["status"] == "active"
        assert payload["cancel_at_period_end"] is True
        # tier is NOT in the update payload — only subscription.deleted changes tier
        assert "tier" not in payload

    def test_no_op_when_user_id_missing(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.updated", user_id=None)

        stripe_events.handle_subscription_updated(event, sb)
        sb.table.assert_not_called()


class TestHandleSubscriptionDeleted:
    def test_sets_tier_free_and_clears_stripe_ids(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.deleted", status="canceled")

        stripe_events.handle_subscription_deleted(event, sb)

        update_call = sb.table("subscriptions").update.call_args
        payload = update_call[0][0]
        assert payload["tier"] == "free"
        assert payload["status"] == "canceled"
        assert payload["stripe_subscription_id"] is None
        assert payload["stripe_price_id"] is None
        assert payload["current_period_end"] is None
        assert payload["cancel_at_period_end"] is False

    def test_no_op_when_user_id_missing(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.deleted", user_id=None)

        stripe_events.handle_subscription_deleted(event, sb)
        sb.table.assert_not_called()


class TestHandleInvoicePaymentFailed:
    def test_sets_status_past_due(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()

        event = MagicMock()
        event.id = "evt_invoice_failed_1"
        event.type = "invoice.payment_failed"
        event.data.object.subscription = "sub_456"

        fake_sub = MagicMock()
        fake_sub.metadata = {"user_id": TEST_USER_ID}
        with patch("stripe.Subscription.retrieve", return_value=fake_sub):
            stripe_events.handle_invoice_payment_failed(event, sb)

        update_call = sb.table("subscriptions").update.call_args
        assert update_call[0][0]["status"] == "past_due"
        # tier stays "pro" — Stripe retries; we keep access during retry window
        assert "tier" not in update_call[0][0]

    def test_no_op_when_subscription_id_missing(self):
        """One-off invoice (not subscription-related) → no action."""
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = MagicMock()
        event.id = "evt_invoice_failed_oneoff"
        event.type = "invoice.payment_failed"
        event.data.object.subscription = None

        stripe_events.handle_invoice_payment_failed(event, sb)
        sb.table.assert_not_called()

    def test_no_op_when_user_id_missing_from_subscription(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = MagicMock()
        event.id = "evt_invoice_failed_2"
        event.type = "invoice.payment_failed"
        event.data.object.subscription = "sub_orphan"

        fake_sub = MagicMock()
        fake_sub.metadata = {}
        with patch("stripe.Subscription.retrieve", return_value=fake_sub):
            stripe_events.handle_invoice_payment_failed(event, sb)
        sb.table.assert_not_called()


class TestHandlersDispatcher:
    def test_handlers_dict_has_4_event_types(self):
        from subscriptions.stripe_events import HANDLERS

        assert "checkout.session.completed" in HANDLERS
        assert "customer.subscription.updated" in HANDLERS
        assert "customer.subscription.deleted" in HANDLERS
        assert "invoice.payment_failed" in HANDLERS

    def test_handlers_does_not_have_payment_succeeded(self):
        """invoice.payment_succeeded is intentionally NOT handled — redundant
        with customer.subscription.updated which arrives alongside it."""
        from subscriptions.stripe_events import HANDLERS

        assert "invoice.payment_succeeded" not in HANDLERS

    def test_unknown_event_type_returns_none(self):
        from subscriptions.stripe_events import HANDLERS

        assert HANDLERS.get("customer.subscription.trial_will_end") is None
