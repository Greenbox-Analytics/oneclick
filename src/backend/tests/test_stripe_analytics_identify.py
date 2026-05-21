"""Tests that Stripe webhook handlers fire analytics.identify for plan changes.

Real handler names + signatures (verified):
  handle_checkout_session_completed(event, supabase)
  handle_subscription_deleted(event, supabase)
  handle_invoice_payment_failed(event, supabase)
"""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID


def _setup(monkeypatch):
    """Set up mocks for analytics calls. Returns list to capture identify calls."""
    identify_calls = []
    monkeypatch.setattr(
        "subscriptions.stripe_events.analytics_identify",
        lambda uid, props: identify_calls.append((uid, dict(props))),
    )
    monkeypatch.setattr(
        "subscriptions.stripe_events.analytics_capture",
        lambda *a, **kw: None,
    )
    return identify_calls


def test_checkout_completed_identifies_pro(monkeypatch):
    """handle_checkout_session_completed identifies with plan='pro'."""
    from subscriptions import stripe_events

    identify_calls = _setup(monkeypatch)
    sb = MagicMock()

    # Build mock event matching what handler reads:
    # - session.metadata.get("user_id")
    # - session.subscription (id)
    # - session.customer (id)
    # - stripe.Subscription.retrieve(subscription_id) → object with items.data[0].price.id
    event = MagicMock()
    event.data.object.metadata = {"user_id": TEST_USER_ID}
    event.data.object.subscription = "sub_123"
    event.data.object.customer = "cus_123"

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

    # Verify identify was called once with correct plan
    assert len(identify_calls) == 1, f"Expected 1 identify call, got {len(identify_calls)}"
    user_id, props = identify_calls[0]
    assert user_id == TEST_USER_ID
    assert props["plan"] == "pro"


def test_subscription_deleted_identifies_free(monkeypatch):
    """handle_subscription_deleted identifies with plan='free'."""
    from subscriptions import stripe_events

    identify_calls = _setup(monkeypatch)
    sb = MagicMock()

    # Build mock event matching what handler reads:
    # - event.data.object (subscription) with metadata.get("user_id")
    event = MagicMock()
    event.data.object.metadata = {"user_id": TEST_USER_ID}
    event.data.object.status = "canceled"
    event.data.object.canceled_at = 1700100000
    event.data.object.current_period_start = 1700000000
    event.data.object.current_period_end = 1702592000
    event.data.object.cancel_at_period_end = False

    stripe_events.handle_subscription_deleted(event, sb)

    # Verify identify was called once with correct plan
    assert len(identify_calls) == 1, f"Expected 1 identify call, got {len(identify_calls)}"
    user_id, props = identify_calls[0]
    assert user_id == TEST_USER_ID
    assert props["plan"] == "free"


def test_payment_failed_identifies_free(monkeypatch):
    """handle_invoice_payment_failed identifies with plan='free'."""
    from subscriptions import stripe_events

    identify_calls = _setup(monkeypatch)
    sb = MagicMock()

    # Build mock event matching what handler reads:
    # - invoice.subscription (id)
    # - stripe.Subscription.retrieve(subscription_id) → object with metadata.get("user_id")
    event = MagicMock()
    event.data.object.subscription = "sub_456"

    fake_sub = MagicMock()
    fake_sub.metadata = {"user_id": TEST_USER_ID}

    with patch("stripe.Subscription.retrieve", return_value=fake_sub):
        stripe_events.handle_invoice_payment_failed(event, sb)

    # Verify identify was called once with correct plan
    assert len(identify_calls) == 1, f"Expected 1 identify call, got {len(identify_calls)}"
    user_id, props = identify_calls[0]
    assert user_id == TEST_USER_ID
    assert props["plan"] == "free"
