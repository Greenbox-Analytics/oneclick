"""Unit tests for stripe_events handlers."""

from unittest.mock import MagicMock, patch

import pytest
import stripe

from tests.conftest import TEST_USER_ID, MockQueryBuilder


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
    subscription_id="sub_123",
    cancel_at=None,
):
    """Build a mock customer.subscription.* event."""
    e = MagicMock()
    e.id = f"evt_{event_type.replace('.', '_')}_1"
    e.type = event_type
    obj = e.data.object
    obj.id = subscription_id
    obj.metadata = {"user_id": user_id} if user_id else {}
    obj.status = status
    obj.cancel_at_period_end = cancel_at_period_end
    obj.cancel_at = cancel_at  # explicit: a bare MagicMock attribute is truthy
    obj.canceled_at = None if not cancel_at_period_end else 1700100000
    obj.current_period_start = current_period_start
    obj.current_period_end = current_period_end
    obj.__getitem__ = lambda self, k: {"items": {"data": [{"price": {"id": price_id}}]}}[k] if k == "items" else None
    return e


def _fake_stripe_subscription(price_id="price_monthly_123", status="active"):
    """What stripe.Subscription.retrieve hands handle_checkout_session_completed."""
    fake_sub = MagicMock(
        status=status,
        cancel_at_period_end=False,
        cancel_at=None,
        canceled_at=None,
        current_period_start=1700000000,
        current_period_end=1702592000,
    )
    fake_sub.__getitem__ = lambda self, k: (
        {"items": {"data": [{"price": {"id": price_id}}]}}[k] if k == "items" else None
    )
    return fake_sub


def _row_read(sb, row):
    """Point the bare mock's single-.eq() read chain at `row` — the shape every
    personal handler uses to read the caller's own subscriptions row."""
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[row] if row is not None else []
    )


class TestTierForPrice:
    def test_unknown_price_id_logs_error_but_keeps_basic_fallback(self, monkeypatch, caplog):
        """FIX: a rotated/unwired Stripe price must not silently misassign
        tiers — the 'basic' fallback stays (never drop a paying customer to
        free), but an ERROR is logged naming the price id."""
        import logging

        from subscriptions.stripe_events import _tier_for_price

        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_basic_m")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_basic_a")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pro_m")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_pro_a")

        with caplog.at_level(logging.ERROR):
            assert _tier_for_price("price_1RotatedUnknown") == "basic"

        errors = [r for r in caplog.records if "price_1RotatedUnknown" in r.getMessage()]
        assert len(errors) == 1 and errors[0].levelno == logging.ERROR

    def test_known_prices_do_not_log(self, monkeypatch, caplog):
        import logging

        from subscriptions.stripe_events import _tier_for_price

        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_basic_m")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_basic_a")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pro_m")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_pro_a")

        with caplog.at_level(logging.ERROR):
            assert _tier_for_price("price_pro_a") == "pro"
            assert _tier_for_price("price_basic_m") == "basic"

        assert not [r for r in caplog.records if "Stripe price" in r.getMessage()]


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
        assert payload["tier"] == "basic"
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

    def test_checkout_into_different_tier_clears_grandfathering(self):
        """Task 1 (spec §1, review r2): the third tier-mutating site. A stored
        tier ("pro") that differs from the checked-out tier ("basic") must not
        let the old grandfathered grant (e.g. 8,000) survive the switch."""
        from subscriptions import stripe_events

        sb = _mock_supabase()
        sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"tier": "pro"}]
        )
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

        payload = sb.table("subscriptions").upsert.call_args[0][0]
        assert payload["tier"] == "basic"
        assert payload["grandfathered_monthly_credits"] is None
        assert payload["grandfathered_until"] is None  # hygiene: expiry cleared alongside the grant

    def test_checkout_into_same_tier_keeps_grandfathering(self):
        """Renewal/no-op checkout into the SAME stored tier must not touch
        grandfathering — the key stays absent from the upsert payload."""
        from subscriptions import stripe_events

        sb = _mock_supabase()
        sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"tier": "basic"}]
        )
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

        payload = sb.table("subscriptions").upsert.call_args[0][0]
        assert payload["tier"] == "basic"
        assert "grandfathered_monthly_credits" not in payload
        assert "grandfathered_until" not in payload

    # --- one live personal subscription per user (2026-09-10) ---

    def test_replacing_a_live_subscription_cancels_the_old_one_at_stripe(self, monkeypatch):
        """The row named sub_old (live) and a checkout for sub_new completed —
        a race past the endpoint's 409, or a session created before it
        shipped. The user paid for sub_new, so it becomes the plan; sub_old is
        canceled at Stripe right away (prorated) instead of billing on with
        nothing in the DB naming it."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(
            sb,
            {"tier": "basic", "stripe_subscription_id": "sub_old", "stripe_customer_id": "cus_old", "status": "active"},
        )
        event = _checkout_session_event(subscription_id="sub_new", customer_id="cus_new")

        def _cancel(sub_id, **params):
            # Ordering is load-bearing: the row must already name sub_new when
            # the `deleted` event Stripe fires for sub_old arrives, so that
            # event hits the id-guard instead of freeing the new plan.
            assert sb.table("subscriptions").upsert.called
            return MagicMock()

        with (
            patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()),
            patch("stripe.Subscription.cancel", side_effect=_cancel) as cancel,
        ):
            stripe_events.handle_checkout_session_completed(event, sb)

        assert sb.table("subscriptions").upsert.call_args[0][0]["stripe_subscription_id"] == "sub_new"
        cancel.assert_called_once_with("sub_old", prorate=True)

    def test_same_subscription_redelivered_cancels_nothing(self, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"tier": "basic", "stripe_subscription_id": "sub_123", "status": "active"})
        event = _checkout_session_event(subscription_id="sub_123")

        with (
            patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()),
            patch("stripe.Subscription.cancel") as cancel,
        ):
            stripe_events.handle_checkout_session_completed(event, sb)

        cancel.assert_not_called()
        assert sb.table("subscriptions").upsert.called

    def test_replacing_a_canceled_subscription_cancels_nothing(self, monkeypatch):
        """A canceled row is not live (a lost `deleted` webhook left the id behind)."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"tier": "free", "stripe_subscription_id": "sub_dead", "status": "canceled"})
        event = _checkout_session_event(subscription_id="sub_new")

        with (
            patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()),
            patch("stripe.Subscription.cancel") as cancel,
        ):
            stripe_events.handle_checkout_session_completed(event, sb)

        cancel.assert_not_called()
        assert sb.table("subscriptions").upsert.call_args[0][0]["stripe_subscription_id"] == "sub_new"

    def test_old_subscription_already_gone_is_not_an_error(self, monkeypatch):
        """Stripe refusing the cancel (already canceled / no such id) IS the goal
        state — log it, ack the event; a 500 here would make Stripe retry a
        handler whose real work (the upsert) is already done."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"tier": "basic", "stripe_subscription_id": "sub_old", "status": "active"})
        event = _checkout_session_event(subscription_id="sub_new")

        with (
            patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()),
            patch(
                "stripe.Subscription.cancel",
                side_effect=stripe.InvalidRequestError("No such subscription: 'sub_old'", "id"),
            ),
        ):
            stripe_events.handle_checkout_session_completed(event, sb)  # no raise

        assert sb.table("subscriptions").upsert.call_args[0][0]["stripe_subscription_id"] == "sub_new"

    def test_stripe_outage_during_the_cancel_propagates_so_stripe_retries(self, monkeypatch):
        """Anything but "already gone" must 500 the webhook: the upsert and the
        wallet alignment are idempotent, so a retry is safe, and a swallowed
        outage would leave sub_old billing with only a log line to show for it."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"tier": "basic", "stripe_subscription_id": "sub_old", "status": "active"})
        event = _checkout_session_event(subscription_id="sub_new")

        with (
            patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()),
            patch("stripe.Subscription.cancel", side_effect=stripe.APIConnectionError("down")),
            pytest.raises(stripe.APIConnectionError),
        ):
            stripe_events.handle_checkout_session_completed(event, sb)


class TestAlignWalletToCheckoutGrandfather:
    """Task 1 follow-up (spec review): _align_wallet_to_checkout is a FIFTH
    grant-writing site — a same-tier re-checkout (past_due recovery, interval
    switch) must top up at an existing grandfathered grant, not the tier
    default, since the period reset makes the wrong number stick for a month.

    Needs a table-name-aware mock (unlike this file's other bare _mock_supabase()
    tests) because the checkout path now reads BOTH `subscriptions` (prev tier +
    grandfather) and `tier_entitlements` (tier default) before topping up —
    two distinct rows a bare Mock can't distinguish by table name.
    """

    # Sentinel: "unset" means "far-future when gf is set, else irrelevant" —
    # existing call sites that only pass gf=... keep testing the unexpired
    # path without touching each one. Pass gf_until explicitly to test expiry.
    _UNSET = object()

    def _table_aware_supabase(self, *, prev_tier, gf, tier_grant, gf_until=_UNSET, bundle_balance=0):
        until = ("2099-01-01T00:00:00+00:00" if gf is not None else None) if gf_until is self._UNSET else gf_until

        def side_effect(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(
                    data=[{"tier": prev_tier, "grandfathered_monthly_credits": gf, "grandfathered_until": until}],
                    count=1,
                )
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[{"monthly_credits": tier_grant}], count=1)
            elif name == "credit_wallets":
                b.execute.return_value = MagicMock(
                    data=[{"id": "w1", "bundle_balance": bundle_balance, "period_start": None}], count=1
                )
            elif name == "credit_ledger":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb = MagicMock()
        sb.table.side_effect = side_effect
        return sb

    def _checkout(self, sb, price_id="price_monthly_123"):
        from subscriptions import stripe_events

        event = _checkout_session_event()
        fake_sub = MagicMock(
            status="active",
            cancel_at_period_end=False,
            canceled_at=None,
            current_period_start=1700000000,
            current_period_end=1702592000,
        )
        fake_sub.__getitem__ = lambda self, k: (
            {"items": {"data": [{"price": {"id": price_id}}]}}[k] if k == "items" else None
        )
        with patch("stripe.Subscription.retrieve", return_value=fake_sub):
            stripe_events.handle_checkout_session_completed(event, sb)

    def test_same_tier_recheckout_tops_up_at_grandfathered_grant(self, monkeypatch):
        """Stored tier == checked-out tier ("basic") and a grandfathered grant
        (8,000) exists — the top-up must use it, not the tier default (5,000)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = self._table_aware_supabase(prev_tier="basic", gf=8000, tier_grant=5000)

        self._checkout(sb)

        grant_call = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"][0]
        assert grant_call.args[1]["p_amount"] == 8000

    def test_different_tier_recheckout_uses_tier_default_not_stale_grandfather(self, monkeypatch):
        """Stored tier ("pro") differs from checked-out tier ("basic") — the
        top-up uses the tier default (5,000), never the old grandfathered grant
        (8,000), matching the grandfather-null this same handler already writes."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = self._table_aware_supabase(prev_tier="pro", gf=8000, tier_grant=5000)

        self._checkout(sb)

        grant_call = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"][0]
        assert grant_call.args[1]["p_amount"] == 5000

    def test_no_grandfather_uses_tier_default(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = self._table_aware_supabase(prev_tier="basic", gf=None, tier_grant=5000)

        self._checkout(sb)

        grant_call = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"][0]
        assert grant_call.args[1]["p_amount"] == 5000

    def test_expired_grandfather_uses_tier_default(self, monkeypatch):
        """Owner policy clarification (spec §1): same tier, a grandfathered
        grant exists, but its grandfathered_until has already passed — the
        top-up must use the tier default (5,000), not the stale grant (8,000)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = self._table_aware_supabase(
            prev_tier="basic", gf=8000, tier_grant=5000, gf_until="2020-01-01T00:00:00+00:00"
        )

        self._checkout(sb)

        grant_call = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"][0]
        assert grant_call.args[1]["p_amount"] == 5000


class TestHandleSubscriptionUpdated:
    def test_syncs_status_period_and_tier(self, monkeypatch):
        """Task 11: tier IS now synced here — portal-driven Pro<->Pro Max switches
        surface via this event. This deliberately supersedes the old "never touch
        tier" isolation (checkout.session.completed and subscription.deleted are no
        longer the only tier-mutating events). Gated behind CREDITS_ENABLED (see
        test_credits_off_does_not_sync_tier for the rollback path)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        # This bare mock's .table() ignores the table name, so every query with
        # the same .select()/.eq() shape shares one chain — the credit_wallets
        # read (two chained .eq() calls) is the only one with that exact shape,
        # so this targets it precisely. Empty data => the upgrade top-up block's
        # `if wallet_res.data:` short-circuits before doing any real dict
        # arithmetic on the (unconfigured, non-dict-shaped) mock wallet row.
        sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[]
        )
        event = _subscription_event("customer.subscription.updated", status="active", cancel_at_period_end=True)

        stripe_events.handle_subscription_updated(event, sb)

        sb.table.assert_any_call("subscriptions")
        update_call = sb.table("subscriptions").update.call_args
        payload = update_call[0][0]
        assert payload["status"] == "active"
        assert payload["cancel_at_period_end"] is True
        # price_id "price_monthly_123" is not a pro_max price → tier resolves to "basic"
        assert payload["tier"] == "basic"

    def test_credits_off_does_not_sync_tier(self, monkeypatch):
        """Pins the clean-rollback guarantee: with CREDITS_ENABLED off, tier is
        NOT included in the update payload (pre-credits behavior), while every
        other field is still synced exactly as today."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.updated", status="active", cancel_at_period_end=True)

        stripe_events.handle_subscription_updated(event, sb)

        update_call = sb.table("subscriptions").update.call_args
        payload = update_call[0][0]
        assert "tier" not in payload
        assert payload["status"] == "active"
        assert payload["cancel_at_period_end"] is True
        assert payload["current_period_start"] is not None
        assert payload["current_period_end"] is not None

    def test_no_op_when_user_id_missing(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.updated", user_id=None)

        stripe_events.handle_subscription_updated(event, sb)
        sb.table.assert_not_called()

    def test_tier_change_ends_grandfathering(self, monkeypatch):
        """Task 1 (spec §1): a tier CHANGE clears grandfathering — a stale
        bundle from the old tier must not survive a plan switch."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        # Stored tier ("pro") differs from the price-resolved tier ("basic",
        # since no STRIPE_PRICE_* env vars are set in this test module) — a
        # downgrade, which also keeps the upgrade top-up block (a second read
        # sharing this same single-.eq() mock node) from ever firing.
        sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"tier": "pro"}]
        )
        event = _subscription_event("customer.subscription.updated", status="active")

        stripe_events.handle_subscription_updated(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["tier"] == "basic"
        assert payload["grandfathered_monthly_credits"] is None
        assert payload["grandfathered_until"] is None  # hygiene: expiry cleared alongside the grant

    def test_same_tier_renewal_keeps_grandfathering(self, monkeypatch):
        """Task 1 (spec §1): a renewal that resolves to the SAME tier must not
        touch grandfathering — the key stays absent from the update payload."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"tier": "basic"}]
        )
        event = _subscription_event("customer.subscription.updated", status="active")

        stripe_events.handle_subscription_updated(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["tier"] == "basic"
        assert "grandfathered_monthly_credits" not in payload
        assert "grandfathered_until" not in payload

    def test_interval_switch_same_tier_keeps_grandfathering(self, monkeypatch):
        """Owner-ruled behavior (spec §1): an annual<->monthly interval switch
        maps to the SAME tier via _tier_for_price, so grandfathering SURVIVES —
        the spec ends grandfathering on a TIER change, and an interval switch
        isn't one, even though the Stripe price id itself changes. Survival is
        still BOUNDED, not indefinite: it reads live at grandfathered_until,
        which this handler doesn't touch on a same-tier move either way."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"tier": "basic"}]
        )
        # Different price_id (interval switch) — still resolves to "basic" (no
        # STRIPE_PRICE_* env vars set in this module), so tier does NOT move.
        event = _subscription_event("customer.subscription.updated", status="active", price_id="price_annual_456")

        stripe_events.handle_subscription_updated(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["tier"] == "basic"
        assert payload["stripe_price_id"] == "price_annual_456"
        assert "grandfathered_monthly_credits" not in payload
        assert "grandfathered_until" not in payload

    def test_flexible_billing_cancel_at_reads_as_scheduled_to_end(self, monkeypatch):
        """Stripe's flexible billing mode reports a Portal cancel as
        cancel_at_period_end=false + cancel_at=<period end> (classic mode, which
        the pinned API version still is, sets both). The profile's "Ends <date>"
        keys entirely on the stored flag, so either shape must set it."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event(
            "customer.subscription.updated", status="active", cancel_at_period_end=False, cancel_at=1702592000
        )

        stripe_events.handle_subscription_updated(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["cancel_at_period_end"] is True

    def test_renewed_plan_clears_the_flag(self, monkeypatch):
        """Portal "Renew plan": both cancel fields come back null/false."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.updated", status="active", cancel_at_period_end=False)

        stripe_events.handle_subscription_updated(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["cancel_at_period_end"] is False


class TestScheduledToEnd:
    @pytest.mark.parametrize(
        ("cancel_at_period_end", "cancel_at", "expected"),
        [
            pytest.param(True, None, True, id="classic-flag"),
            pytest.param(False, 1702592000, True, id="flexible-cancel-at"),
            pytest.param(True, 1702592000, True, id="both"),
            pytest.param(False, None, False, id="neither"),
            pytest.param(False, "1702592000", False, id="cancel-at-must-be-a-timestamp"),
        ],
    )
    def test_reads_either_stripe_shape(self, cancel_at_period_end, cancel_at, expected):
        from subscriptions.stripe_events import _scheduled_to_end

        sub = MagicMock(cancel_at_period_end=cancel_at_period_end, cancel_at=cancel_at)
        assert _scheduled_to_end(sub) is expected

    def test_unconfigured_mock_attribute_is_not_a_cancel(self):
        """The isinstance check is the live contract (Stripe sends an int or
        null) and keeps bare-MagicMock fixtures honest."""
        from subscriptions.stripe_events import _scheduled_to_end

        sub = MagicMock(cancel_at_period_end=False)  # .cancel_at is an auto MagicMock
        assert _scheduled_to_end(sub) is False

    def test_real_stripe_object_without_cancel_at(self):
        from subscriptions.stripe_events import _scheduled_to_end

        sub = stripe.Subscription.construct_from(
            {"id": "sub_1", "object": "subscription", "cancel_at_period_end": False}, "sk_test_x"
        )
        assert _scheduled_to_end(sub) is False
        sub = stripe.Subscription.construct_from(
            {"id": "sub_1", "object": "subscription", "cancel_at_period_end": False, "cancel_at": 1702592000},
            "sk_test_x",
        )
        assert _scheduled_to_end(sub) is True


class TestSyncPeriodFromStripe:
    """The row mirror behind the Portal cancel deep-link and
    `POST /billing/sync-subscription`: a live subscription's period and cancel
    fields, written from Stripe when the webhook is late or lost."""

    @staticmethod
    def _sub(**overrides):
        data = {
            "id": "sub_123",
            "object": "subscription",
            "status": "active",
            "cancel_at_period_end": True,
            "cancel_at": 1702592000,
            "current_period_start": 1700000000,
            "current_period_end": 1702592000,
            "canceled_at": 1702585200,
        }
        data.update(overrides)
        return stripe.Subscription.construct_from(data, "sk_test_x")

    @staticmethod
    def _recording(sb):
        b = MockQueryBuilder()
        b.update = MagicMock(return_value=b)
        b.eq = MagicMock(return_value=b)
        sb.table.return_value = b
        return b

    def test_writes_the_fields_the_updated_handler_would(self):
        from subscriptions.stripe_events import sync_period_from_stripe

        sb = _mock_supabase()
        b = self._recording(sb)

        assert sync_period_from_stripe(sb, TEST_USER_ID, self._sub()) is True

        sb.table.assert_called_once_with("subscriptions")
        b.update.assert_called_once_with(
            {
                "cancel_at_period_end": True,
                "current_period_start": "2023-11-14T22:13:20+00:00",
                "current_period_end": "2023-12-14T22:13:20+00:00",
                "canceled_at": "2023-12-14T20:20:00+00:00",
            }
        )
        assert [c.args for c in b.eq.call_args_list] == [
            ("user_id", TEST_USER_ID),
            ("stripe_subscription_id", "sub_123"),
        ]
        b.execute.assert_called_once()

    def test_flexible_billing_shape_counts_as_scheduled(self):
        from subscriptions.stripe_events import sync_period_from_stripe

        sb = _mock_supabase()
        b = self._recording(sb)
        assert sync_period_from_stripe(sb, TEST_USER_ID, self._sub(cancel_at_period_end=False)) is True
        assert b.update.call_args.args[0]["cancel_at_period_end"] is True

    def test_a_renewed_subscription_is_mirrored_too(self):
        """ "Renew plan" on the Portal home clears the cancel; the row must
        follow without waiting on the webhook."""
        from subscriptions.stripe_events import sync_period_from_stripe

        sb = _mock_supabase()
        b = self._recording(sb)
        sub = self._sub(cancel_at_period_end=False, cancel_at=None, canceled_at=None)
        assert sync_period_from_stripe(sb, TEST_USER_ID, sub) is False
        b.update.assert_called_once_with(
            {
                "cancel_at_period_end": False,
                "current_period_start": "2023-11-14T22:13:20+00:00",
                "current_period_end": "2023-12-14T22:13:20+00:00",
                "canceled_at": None,
            }
        )
        # Never status, tier or credits — those stay with the webhook handlers.
        assert "status" not in b.update.call_args.args[0]
        assert "tier" not in b.update.call_args.args[0]


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

    def test_deletion_ends_grandfathering(self):
        """Task 1 (spec §1): deletion unconditionally NULLs grandfathering —
        free is the only grant this user gets going forward."""
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.deleted", status="canceled")

        stripe_events.handle_subscription_deleted(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["grandfathered_monthly_credits"] is None
        assert payload["grandfathered_until"] is None  # hygiene: expiry cleared alongside the grant

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
        # tier stays "basic" — Stripe retries; we keep access during retry window
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


class TestNamesOtherLiveSubscription:
    """The one predicate behind every id-guard (2026-09-10)."""

    @pytest.mark.parametrize(
        "row, sub_id, expected",
        [
            ({"stripe_subscription_id": "sub_a", "status": "active"}, "sub_b", True),
            ({"stripe_subscription_id": "sub_a", "status": "past_due"}, "sub_b", True),
            ({"stripe_subscription_id": "sub_a", "status": "trialing"}, "sub_b", True),
            ({"stripe_subscription_id": "sub_a", "status": "active"}, "sub_a", False),  # the same one
            ({"stripe_subscription_id": "sub_a", "status": "canceled"}, "sub_b", False),  # stale id, not live
            ({"stripe_subscription_id": None, "status": "active"}, "sub_b", False),  # names none
            ({"stripe_subscription_id": "", "status": "active"}, "sub_b", False),
            ({"tier": "basic"}, "sub_b", False),  # column not selected
            (None, "sub_b", False),  # no row
            (MagicMock(), "sub_b", False),  # an unconfigured test double is never a live row
        ],
    )
    def test_predicate(self, row, sub_id, expected):
        from subscriptions.stripe_events import _names_other_live_subscription

        assert _names_other_live_subscription(row, sub_id) is expected


class TestForeignSubscriptionEventsAreIgnored:
    """One personal subscription per user (2026-09-10). Once the row names a
    subscription, events for any OTHER not-canceled one must not touch it.
    The money case: a duplicate the user (or ops) cancels in the portal —
    its `deleted` event used to free the plan they are actually paying for.
    A row naming nothing, or naming a canceled subscription, keeps today's
    permissive behaviour (dashboard-created subscriptions, out-of-order
    delivery before checkout.session.completed)."""

    def test_updated_for_a_subscription_the_row_does_not_name_is_ignored(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pro_monthly")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"tier": "basic", "stripe_subscription_id": "sub_current", "status": "active"})
        # A stale Pro event: without the guard this would ALSO fire the upgrade top-up.
        event = _subscription_event(
            "customer.subscription.updated", subscription_id="sub_stale", price_id="price_pro_monthly"
        )

        stripe_events.handle_subscription_updated(event, sb)

        sb.table("subscriptions").update.assert_not_called()
        sb.rpc.assert_not_called()

    def test_updated_for_the_named_subscription_still_syncs(self, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"tier": "basic", "stripe_subscription_id": "sub_123", "status": "active"})
        event = _subscription_event(
            "customer.subscription.updated", subscription_id="sub_123", cancel_at_period_end=True
        )

        stripe_events.handle_subscription_updated(event, sb)

        assert sb.table("subscriptions").update.call_args[0][0]["cancel_at_period_end"] is True

    def test_updated_when_the_named_subscription_is_canceled_syncs_the_new_one(self, monkeypatch):
        """A canceled row is not live: a lost `deleted` webhook must not lock the row forever."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"tier": "free", "stripe_subscription_id": "sub_dead", "status": "canceled"})
        event = _subscription_event("customer.subscription.updated", subscription_id="sub_new")

        stripe_events.handle_subscription_updated(event, sb)

        sb.table("subscriptions").update.assert_called_once()

    def test_deleted_for_a_replaced_subscription_leaves_the_row_alone(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"stripe_subscription_id": "sub_new", "status": "active", "stripe_customer_id": "cus_1"})
        event = _subscription_event("customer.subscription.deleted", subscription_id="sub_old", status="canceled")

        with (
            patch("subscriptions.stripe_events.analytics_capture") as capture,
            patch("subscriptions.overage_billing.bill_pending_overage") as bpo,
        ):
            stripe_events.handle_subscription_deleted(event, sb)

        sb.table("subscriptions").update.assert_not_called()
        capture.assert_not_called()  # no subscription_canceled — nothing was canceled for this user
        bpo.assert_not_called()  # they still have a live plan; the sweep still covers them

    def test_deleted_for_the_named_subscription_still_frees_the_row(self, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"stripe_subscription_id": "sub_123", "status": "active", "stripe_customer_id": "cus_1"})
        event = _subscription_event("customer.subscription.deleted", subscription_id="sub_123", status="canceled")

        stripe_events.handle_subscription_deleted(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["tier"] == "free"
        assert payload["stripe_subscription_id"] is None

    def test_payment_failed_on_a_replaced_subscription_does_not_mark_past_due(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"stripe_subscription_id": "sub_new", "status": "active"})
        event = MagicMock()
        event.id = "evt_invoice_failed_orphan"
        event.type = "invoice.payment_failed"
        event.data.object.subscription = "sub_old"
        fake_sub = MagicMock()
        fake_sub.metadata = {"user_id": TEST_USER_ID}

        with patch("stripe.Subscription.retrieve", return_value=fake_sub):
            stripe_events.handle_invoice_payment_failed(event, sb)

        sb.table("subscriptions").update.assert_not_called()

    def test_payment_failed_on_the_named_subscription_marks_past_due(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        _row_read(sb, {"stripe_subscription_id": "sub_456", "status": "active"})
        event = MagicMock()
        event.id = "evt_invoice_failed_named"
        event.type = "invoice.payment_failed"
        event.data.object.subscription = "sub_456"
        fake_sub = MagicMock()
        fake_sub.metadata = {"user_id": TEST_USER_ID}

        with patch("stripe.Subscription.retrieve", return_value=fake_sub):
            stripe_events.handle_invoice_payment_failed(event, sb)

        assert sb.table("subscriptions").update.call_args[0][0]["status"] == "past_due"


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


# ---------------------------------------------------------------------------
# Recurring ORG top-up isolation (spec 2026-08-15 §4.3, Task 11).
#
# A top-up is a Stripe SUBSCRIPTION living on the purchasing ADMIN's personal
# customer, so every personal-subscription handler sees its events. The whole
# point of these tests is that none of them acts on one.
# ---------------------------------------------------------------------------

ORG_ID = "10000000-0000-0000-0000-0000000000aa"
RECURRING_PRICE = "price_rec_500"
PACK_ROW = {"key": "pack_500", "credits": 500, "price_cents": 1000}


def _tables_touched(sb):
    return [c.args[0] for c in sb.table.call_args_list if c.args]


def _org_topup_checkout_event(org_id=ORG_ID, purchased_by=TEST_USER_ID, subscription_id="sub_topup_1"):
    """checkout.session.completed for an org top-up: mode='subscription',
    metadata carries {org_id, kind, purchased_by} and NO user_id."""
    e = MagicMock()
    e.id = "evt_topup_checkout_1"
    e.type = "checkout.session.completed"
    obj = e.data.object
    obj.id = "cs_topup_1"
    obj.mode = "subscription"
    meta = {"kind": "org_topup", "purchased_by": purchased_by}
    if org_id:
        meta["org_id"] = org_id
    obj.metadata = meta
    obj.subscription = subscription_id
    obj.customer = "cus_admin"
    return e


def _org_topup_invoice_event(
    event_type="invoice.paid",
    org_id=ORG_ID,
    price_id=RECURRING_PRICE,
    invoice_id="in_topup_1",
    kind="org_topup",
    extra_lines=(),
):
    """An invoice of the top-up subscription. Stripe copies the Subscription's
    metadata onto `subscription_details.metadata` of every invoice — the ONLY
    thing that identifies these (the org row's topup column may not be written
    yet when the first invoice lands)."""
    e = MagicMock()
    e.id = f"evt_{event_type.replace('.', '_')}_1"
    e.type = event_type
    obj = e.data.object
    obj.id = invoice_id
    meta = {"purchased_by": TEST_USER_ID}
    if kind:
        meta["kind"] = kind
    if org_id:
        meta["org_id"] = org_id
    obj.subscription_details = {"metadata": meta}
    obj.lines = {"data": [*extra_lines, {"price": {"id": price_id}}]}
    obj.customer = "cus_admin"
    obj.billing_reason = "subscription_cycle"
    return e


class TestOrgTopupCheckoutCompleted:
    def test_writes_org_columns_and_never_touches_subscriptions(self):
        """THE DEMOTION REGRESSION TEST (review r2): the branch must run BEFORE
        the personal path, which would resolve the top-up's price through
        _tier_for_price (unknown -> 'basic') and upsert the PURCHASING ADMIN's
        subscriptions row — silently demoting a Pro admin who bought credits
        for their team."""
        from subscriptions import stripe_events

        sb = _mock_supabase()
        with patch("stripe.Subscription.retrieve") as retrieve:
            stripe_events.handle_checkout_session_completed(_org_topup_checkout_event(), sb)

        assert "subscriptions" not in _tables_touched(sb)
        retrieve.assert_not_called()  # no Subscription fetch either
        payload = sb.table("organizations").update.call_args[0][0]
        assert payload == {"topup_stripe_subscription_id": "sub_topup_1", "topup_admin_id": TEST_USER_ID}

    def test_missing_org_id_writes_nothing(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        stripe_events.handle_checkout_session_completed(_org_topup_checkout_event(org_id=None), sb)
        sb.table.assert_not_called()


class TestOrgTopupSubscriptionEventsIgnored:
    def test_updated_is_a_no_op(self, monkeypatch):
        """Guard is the FIRST line — above the tier sync AND the grandfathering
        reads that were added later."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.updated")
        event.data.object.metadata = {"kind": "org_topup", "org_id": ORG_ID, "purchased_by": TEST_USER_ID}

        stripe_events.handle_subscription_updated(event, sb)
        sb.table.assert_not_called()
        sb.rpc.assert_not_called()

    def test_deleted_clears_org_columns_and_never_touches_subscriptions(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        event = _subscription_event("customer.subscription.deleted", status="canceled")
        event.data.object.id = "sub_topup_1"
        event.data.object.metadata = {"kind": "org_topup", "org_id": ORG_ID, "purchased_by": TEST_USER_ID}

        stripe_events.handle_subscription_deleted(event, sb)

        assert "subscriptions" not in _tables_touched(sb)
        payload = sb.table("organizations").update.call_args[0][0]
        assert payload == {"topup_stripe_subscription_id": None, "topup_admin_id": None}


class TestHandleInvoicePaid:
    """Renewals. Mirrors the pack fulfilment (_handle_topup_completed ->
    _handle_org_topup_grant): kind='purchase', bucket='reserve', granted into
    the org POOL wallet, so a renewal is ledger-indistinguishable from a pack
    purchase apart from its request id."""

    def _sb_with_pack(self, duplicate=False):
        sb = _mock_supabase()
        # credit_packs lookup over EVERY line's price: select().in_().execute()
        sb.table.return_value.select.return_value.in_.return_value.execute.return_value = MagicMock(data=[PACK_ROW])
        sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": duplicate})
        return sb

    def test_grants_into_pool_resolving_org_from_invoice_metadata(self):
        """THE ORDERING TEST: the org row's topup_stripe_subscription_id is
        still NULL (checkout.session.completed hasn't landed — webhooks are
        unordered), so the org MUST come from subscription_details.metadata.
        `organizations` is never even read here."""
        from subscriptions import stripe_events

        sb = self._sb_with_pack()
        with (
            patch("orgs.wallets.read_or_create_org_wallet", return_value={"id": "pool-1"}) as wallet,
            patch("orgs.wallets.maybe_activate_org") as activate,
        ):
            stripe_events.handle_invoice_paid(_org_topup_invoice_event(), sb)

        assert "organizations" not in _tables_touched(sb)
        wallet.assert_called_once_with(sb, ORG_ID)
        params = sb.rpc.call_args[0][1]
        assert sb.rpc.call_args[0][0] == "grant_credits"
        assert params["p_wallet_id"] == "pool-1"
        assert params["p_amount"] == 500
        assert params["p_kind"] == "purchase"  # same kind/bucket as a pack
        assert params["p_bucket"] == "reserve"
        assert params["p_request_id"] == "orgtopup:in_topup_1"
        assert params["p_metadata"]["org_id"] == ORG_ID
        activate.assert_called_once_with(sb, ORG_ID, "pool-1")

    def test_grants_when_an_overage_line_shares_the_invoice(self):
        """The top-up invoice is NOT single-line: pending overage InvoiceItems
        on the admin's own customer (sweep._bill_one_owner) are auto-collected
        onto the next invoice created for them, which can be this one — the
        invoice.created guard only stops the explicit straggler attach. Reading
        line 0 would miss the pack, raise, and 500 the webhook forever, so the
        org's paid month would never grant. Every line's price is searched."""
        from subscriptions import stripe_events

        sb = self._sb_with_pack()
        overage_lines = [{"price": None}, {"price": {"id": "price_adhoc_overage"}}]
        with (
            patch("orgs.wallets.read_or_create_org_wallet", return_value={"id": "pool-1"}),
            patch("orgs.wallets.maybe_activate_org"),
        ):
            stripe_events.handle_invoice_paid(_org_topup_invoice_event(extra_lines=overage_lines), sb)

        # The ad-hoc line has no pack; the recurring price is still found.
        searched = sb.table.return_value.select.return_value.in_.call_args[0][1]
        assert searched == ["price_adhoc_overage", RECURRING_PRICE]
        params = sb.rpc.call_args[0][1]
        assert params["p_amount"] == 500
        assert params["p_request_id"] == "orgtopup:in_topup_1"
        assert params["p_metadata"]["pack_key"] == "pack_500"

    def test_duplicate_grant_does_not_re_report_revenue(self):
        """Stripe redelivery: the RPC reports duplicate:true and the analytics
        event must not fire twice."""
        from subscriptions import stripe_events

        sb = self._sb_with_pack(duplicate=True)
        with (
            patch("orgs.wallets.read_or_create_org_wallet", return_value={"id": "pool-1"}),
            patch("orgs.wallets.maybe_activate_org"),
            patch("subscriptions.stripe_events.analytics_capture") as capture,
        ):
            stripe_events.handle_invoice_paid(_org_topup_invoice_event(), sb)
        capture.assert_not_called()

    def test_fresh_grant_reports_renewal(self):
        from subscriptions import stripe_events

        sb = self._sb_with_pack()
        with (
            patch("orgs.wallets.read_or_create_org_wallet", return_value={"id": "pool-1"}),
            patch("orgs.wallets.maybe_activate_org"),
            patch("subscriptions.stripe_events.analytics_capture") as capture,
        ):
            stripe_events.handle_invoice_paid(_org_topup_invoice_event(), sb)
        assert capture.call_args[0][1] == "org_topup_renewed"

    def test_raises_when_kind_tagged_invoice_has_no_org_id(self):
        """Unresolvable money must 500 so Stripe retries — never a silent ack."""
        from subscriptions import stripe_events

        sb = _mock_supabase()
        with pytest.raises(RuntimeError):
            stripe_events.handle_invoice_paid(_org_topup_invoice_event(org_id=None), sb)

    def test_raises_when_no_pack_matches_the_recurring_price(self):
        """A kind-tagged invoice with no recognizable pack line at all is a
        real fault: raise so Stripe retries rather than acking a paid month."""
        from subscriptions import stripe_events

        sb = _mock_supabase()
        sb.table.return_value.select.return_value.in_.return_value.execute.return_value = MagicMock(data=[])
        with pytest.raises(RuntimeError):
            stripe_events.handle_invoice_paid(_org_topup_invoice_event(), sb)

    def test_personal_invoice_is_a_no_op(self):
        from subscriptions import stripe_events

        sb = _mock_supabase()
        stripe_events.handle_invoice_paid(_org_topup_invoice_event(kind=None), sb)
        sb.table.assert_not_called()
        sb.rpc.assert_not_called()

    def test_dispatcher_routes_invoice_paid(self):
        from subscriptions.stripe_events import HANDLERS

        assert HANDLERS["invoice.paid"] is not None


class TestInvoiceCreatedSkipsOrgTopup:
    def test_org_topup_invoice_gets_no_personal_overage_items(self, monkeypatch):
        """handle_invoice_created matches by CUSTOMER, and the top-up sits on
        the admin's personal customer — without the filter their personal
        credit-overage items would attach to the org's top-up invoice."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb = _mock_supabase()
        stripe_events.handle_invoice_created(_org_topup_invoice_event(event_type="invoice.created"), sb)
        sb.table.assert_not_called()


class TestRealStripeObjects:
    """REGRESSION GUARD for the shape no mock reproduces (re-review r3).

    Every test above feeds MagicMocks or plain dicts. On the pinned
    stripe==15.1.0, `StripeObject` is NOT a dict subclass — `isinstance(o, dict)`
    is False and `o.get(...)` raises AttributeError — so a handler that reads
    metadata dict-style passes the mocked tests and, on live traffic, either
    500s forever (personal path) or silently returns 200 without granting
    (org top-up: `_subscription_metadata` fell through to {} and the
    kind check exited early — a paid month lost with no retry).

    `stripe.Event.construct_from` is the only way to build the genuine nested
    StripeObjects Stripe's SDK hands the webhook. These two tests fail loudly
    if `_plain` is removed.
    """

    def _event(self, payload):
        import stripe as stripe_sdk

        return stripe_sdk.Event.construct_from(payload, "sk_test")

    def test_stripe_object_is_not_a_dict(self):
        """Pins the premise — if a future SDK bump makes StripeObject dict-like
        again, this test tells you `_plain` is now dead weight."""
        import stripe as stripe_sdk

        ev = self._event({"id": "evt_x", "type": "invoice.paid", "data": {"object": {"id": "in_x", "metadata": {}}}})
        obj = ev.data.object
        assert isinstance(obj, stripe_sdk.StripeObject)
        assert not isinstance(obj, dict)
        assert not hasattr(obj, "get")

    def test_org_topup_invoice_paid_grants_on_a_real_payload(self):
        """The money path, end to end, on genuine StripeObjects: metadata read
        off subscription_details, org resolved from it, multi-line invoice
        (ad-hoc overage line first), grant RPC fired with the pack's credits."""
        from subscriptions import stripe_events

        event = self._event(
            {
                "id": "evt_real_invoice_paid",
                "type": "invoice.paid",
                "data": {
                    "object": {
                        "id": "in_real_1",
                        "object": "invoice",
                        "customer": "cus_admin",
                        "billing_reason": "subscription_cycle",
                        "subscription_details": {
                            "metadata": {"kind": "org_topup", "org_id": ORG_ID, "purchased_by": TEST_USER_ID}
                        },
                        "lines": {
                            "object": "list",
                            "data": [
                                {"id": "il_overage", "price": {"id": "price_adhoc_overage"}},
                                {"id": "il_pack", "price": {"id": RECURRING_PRICE}},
                            ],
                        },
                    }
                },
            }
        )
        sb = _mock_supabase()
        sb.table.return_value.select.return_value.in_.return_value.execute.return_value = MagicMock(data=[PACK_ROW])
        sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})

        with (
            patch("orgs.wallets.read_or_create_org_wallet", return_value={"id": "pool-1"}),
            patch("orgs.wallets.maybe_activate_org") as activate,
        ):
            stripe_events.handle_invoice_paid(event, sb)

        assert sb.table.return_value.select.return_value.in_.call_args[0][1] == [
            "price_adhoc_overage",
            RECURRING_PRICE,
        ]
        params = sb.rpc.call_args[0][1]
        assert (params["p_amount"], params["p_kind"], params["p_bucket"]) == (500, "purchase", "reserve")
        assert params["p_request_id"] == "orgtopup:in_real_1"
        assert params["p_metadata"]["org_id"] == ORG_ID
        activate.assert_called_once_with(sb, ORG_ID, "pool-1")

    def test_personal_subscription_updated_reaches_its_write_on_a_real_payload(self, monkeypatch):
        """The pre-existing personal path: `sub.metadata.get("user_id")` raised
        AttributeError on every live delivery before `_plain` (500 → Stripe
        retry loop). It must reach the subscriptions UPDATE."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_basic_m")
        from subscriptions import stripe_events

        event = self._event(
            {
                "id": "evt_real_sub_updated",
                "type": "customer.subscription.updated",
                "data": {
                    "object": {
                        "id": "sub_real_1",
                        "object": "subscription",
                        "status": "active",
                        "customer": "cus_1",
                        "metadata": {"user_id": TEST_USER_ID},
                        "items": {"object": "list", "data": [{"id": "si_1", "price": {"id": "price_basic_m"}}]},
                        "current_period_start": 1700000000,
                        "current_period_end": 1702592000,
                        "cancel_at_period_end": False,
                        "canceled_at": None,
                    }
                },
            }
        )
        sb = _mock_supabase()
        sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[]
        )

        stripe_events.handle_subscription_updated(event, sb)

        payload = sb.table("subscriptions").update.call_args[0][0]
        assert payload["status"] == "active"
        assert payload["tier"] == "basic"
        assert payload["stripe_price_id"] == "price_basic_m"

    def test_org_topup_checkout_completed_on_a_real_payload(self):
        """Session metadata read dict-style too — a real session must still hit
        the org branch (and not the personal path that would demote the admin)."""
        from subscriptions import stripe_events

        event = self._event(
            {
                "id": "evt_real_checkout",
                "type": "checkout.session.completed",
                "data": {
                    "object": {
                        "id": "cs_real_1",
                        "object": "checkout.session",
                        "mode": "subscription",
                        "subscription": "sub_topup_real",
                        "customer": "cus_admin",
                        "metadata": {"kind": "org_topup", "org_id": ORG_ID, "purchased_by": TEST_USER_ID},
                    }
                },
            }
        )
        sb = _mock_supabase()
        with patch("stripe.Subscription.retrieve") as retrieve:
            stripe_events.handle_checkout_session_completed(event, sb)

        retrieve.assert_not_called()
        assert "subscriptions" not in _tables_touched(sb)
        assert sb.table("organizations").update.call_args[0][0] == {
            "topup_stripe_subscription_id": "sub_topup_real",
            "topup_admin_id": TEST_USER_ID,
        }

    def test_deleted_id_guard_reads_a_real_subscription_object(self, monkeypatch):
        """The id-guard (2026-09-10) reads `sub.id` and `sub.metadata` off the
        event object. Attribute access works on StripeObject, but pin it on the
        genuine shape next to the `.get()` regression this class exists for:
        a foreign `deleted` leaves the row alone, the named one still frees it."""
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        from subscriptions import stripe_events

        def _deleted(sub_id):
            return self._event(
                {
                    "id": f"evt_real_deleted_{sub_id}",
                    "type": "customer.subscription.deleted",
                    "data": {
                        "object": {
                            "id": sub_id,
                            "object": "subscription",
                            "status": "canceled",
                            "customer": "cus_real",
                            "canceled_at": 1700000000,
                            "cancel_at_period_end": False,
                            "metadata": {"user_id": TEST_USER_ID},
                        }
                    },
                }
            )

        sb = _mock_supabase()
        _row_read(sb, {"stripe_subscription_id": "sub_keep", "status": "active", "stripe_customer_id": "cus_real"})
        stripe_events.handle_subscription_deleted(_deleted("sub_gone"), sb)
        sb.table("subscriptions").update.assert_not_called()

        sb = _mock_supabase()
        _row_read(sb, {"stripe_subscription_id": "sub_gone", "status": "active", "stripe_customer_id": "cus_real"})
        stripe_events.handle_subscription_deleted(_deleted("sub_gone"), sb)
        assert sb.table("subscriptions").update.call_args[0][0]["tier"] == "free"
