"""Task 11: Stripe Pro Max checkout, tier switches, grants, overage InvoiceItems.

Covers:
  - _tier_for_price price->tier mapping
  - handle_checkout_session_completed: tier resolution + wallet top-up/re-anchor
  - handle_subscription_updated: tier sync + upgrade top-up (with farming guard)
  - overage_billing.bill_overage_row / bill_pending_overage
  - handle_invoice_created safety-net handler
  - billing_router plan->env price mapping (pro_max_monthly / pro_max_annual)
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import TEST_USER_ID

# ---------------------------------------------------------------------------
# Shared helpers (mirrors the MagicMock-event patterns in test_stripe_events.py)
# ---------------------------------------------------------------------------


def _mock_supabase(table_data: dict):
    """Return (sb, builders): a MagicMock supabase client whose .table(name) returns a
    CACHED, chainable builder per table name — repeated calls to the same table name
    return the SAME builder, so `.update.call_args` / `.upsert.call_args` etc. can be
    inspected after the code under test runs.

    table_data maps table name -> list of rows for that table's default `.execute().data`.
    Tables not present default to an empty list.
    """
    builders: dict = {}

    def get_builder(name):
        if name not in builders:
            b = MagicMock()
            for chain_method in ("select", "eq", "update", "upsert", "insert", "in_", "order", "limit", "is_", "gte"):
                getattr(b, chain_method).return_value = b
            b.execute.return_value = MagicMock(data=list(table_data.get(name, [])))
            builders[name] = b
        return builders[name]

    sb = MagicMock()
    sb.table.side_effect = get_builder
    sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False, "balance_after": 0})
    return sb, builders


def _checkout_session_event(user_id=TEST_USER_ID, subscription_id="sub_123", customer_id="cus_123"):
    e = MagicMock()
    e.id = "evt_checkout_1"
    e.type = "checkout.session.completed"
    e.data.object.metadata = {"user_id": user_id} if user_id else {}
    e.data.object.subscription = subscription_id
    e.data.object.customer = customer_id
    return e


def _fake_stripe_subscription(price_id="price_monthly_123"):
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
    return fake_sub


def _subscription_event(
    event_type,
    user_id=TEST_USER_ID,
    status="active",
    cancel_at_period_end=False,
    price_id="price_monthly_123",
    current_period_start=1700000000,
    current_period_end=1702592000,
):
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


# ---------------------------------------------------------------------------
# _tier_for_price
# ---------------------------------------------------------------------------


class TestTierForPrice:
    def test_maps_pro_max_monthly_price(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_pm_annual")
        from subscriptions.stripe_events import _tier_for_price

        assert _tier_for_price("price_pm_monthly") == "pro_max"

    def test_maps_pro_max_annual_price(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_pm_annual")
        from subscriptions.stripe_events import _tier_for_price

        assert _tier_for_price("price_pm_annual") == "pro_max"

    def test_unknown_price_defaults_to_pro(self, monkeypatch):
        monkeypatch.delenv("STRIPE_PRICE_PRO_MAX_MONTHLY", raising=False)
        monkeypatch.delenv("STRIPE_PRICE_PRO_MAX_ANNUAL", raising=False)
        from subscriptions.stripe_events import _tier_for_price

        assert _tier_for_price("price_legacy_monthly_123") == "pro"

    def test_none_price_defaults_to_pro(self):
        from subscriptions.stripe_events import _tier_for_price

        assert _tier_for_price(None) == "pro"


# ---------------------------------------------------------------------------
# handle_checkout_session_completed — tier resolution + wallet alignment
# ---------------------------------------------------------------------------


class TestHandleCheckoutSessionCompletedCredits:
    def test_upserts_tier_pro_max_from_price(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        sb, builders = _mock_supabase({})
        event = _checkout_session_event()

        with patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription("price_pm_monthly")):
            stripe_events.handle_checkout_session_completed(event, sb)

        payload = builders["subscriptions"].upsert.call_args[0][0]
        assert payload["tier"] == "pro_max"

    def test_credits_on_grants_topup_and_reanchors_wallet_period(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb, builders = _mock_supabase(
            {
                "tier_entitlements": [{"monthly_credits": 3000}],
                "credit_wallets": [
                    {"id": "wallet-1", "bundle_balance": 500, "period_start": "2026-07-01T00:00:00+00:00"}
                ],
            }
        )
        event = _checkout_session_event()

        with patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()):
            stripe_events.handle_checkout_session_completed(event, sb)

        grant_calls = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"]
        assert len(grant_calls) == 1
        payload = grant_calls[0].args[1]
        assert payload["p_wallet_id"] == "wallet-1"
        assert payload["p_amount"] == 2500  # 3000 (grant) - 500 (existing bundle)
        assert payload["p_kind"] == "monthly_grant"
        assert payload["p_bucket"] == "bundle"
        assert payload["p_request_id"] == f"checkout:{event.id}"

        update_payload = builders["credit_wallets"].update.call_args[0][0]
        assert "period_start" in update_payload
        assert "period_end" in update_payload
        assert update_payload["overage_this_period"] == 0

    def test_checkout_farming_guard_caps_by_period_grant_sum(self, monkeypatch):
        """Spend→cancel→re-checkout in one period: bundle 0 (all spent) but ledger
        shows the full grant already issued this period → NO new grant."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb, builders = _mock_supabase(
            {
                "tier_entitlements": [{"monthly_credits": 3000}],
                "credit_wallets": [
                    {"id": "wallet-1", "bundle_balance": 0, "period_start": "2026-07-01T00:00:00+00:00"}
                ],
                "credit_ledger": [{"delta": 3000, "created_at": "2026-07-02T00:00:00+00:00"}],
            }
        )
        event = _checkout_session_event()

        with patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()):
            stripe_events.handle_checkout_session_completed(event, sb)

        assert not any(c.args[0] == "grant_credits" for c in sb.rpc.call_args_list)

    def test_credits_off_makes_no_rpc_calls(self):
        """Wallet + tier rows are present so the alignment WOULD grant if the
        credits_enabled() gate were removed — pins that the gate is load-bearing."""
        from subscriptions import stripe_events

        sb, builders = _mock_supabase(
            {
                "tier_entitlements": [{"monthly_credits": 3000}],
                "credit_wallets": [
                    {"id": "wallet-1", "bundle_balance": 0, "period_start": "2026-07-01T00:00:00+00:00"}
                ],
            }
        )
        event = _checkout_session_event()

        with patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()):
            stripe_events.handle_checkout_session_completed(event, sb)

        sb.rpc.assert_not_called()
        assert "credit_wallets" not in builders  # wallet never even read

    def test_grant_failure_propagates_so_stripe_retries(self, monkeypatch):
        """No swallow: a failed grant must raise so the webhook router 500s and
        Stripe redelivers (the grant is request-id idempotent, so retry is safe)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb, builders = _mock_supabase(
            {
                "tier_entitlements": [{"monthly_credits": 3000}],
                "credit_wallets": [
                    {"id": "wallet-1", "bundle_balance": 0, "period_start": "2026-07-01T00:00:00+00:00"}
                ],
            }
        )
        sb.rpc.return_value.execute.side_effect = RuntimeError("grant_credits down")
        event = _checkout_session_event()

        with (
            patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription()),
            pytest.raises(RuntimeError, match="grant_credits down"),
        ):
            stripe_events.handle_checkout_session_completed(event, sb)

    def test_pro_max_checkout_identifies_plan_pro_max(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        identify_calls = []
        capture_calls = []
        monkeypatch.setattr(
            "subscriptions.stripe_events.analytics_identify",
            lambda uid, props: identify_calls.append((uid, dict(props))),
        )
        monkeypatch.setattr(
            "subscriptions.stripe_events.analytics_capture",
            lambda uid, name, props: capture_calls.append((uid, name, dict(props))),
        )

        sb, builders = _mock_supabase({})
        event = _checkout_session_event()

        with patch("stripe.Subscription.retrieve", return_value=_fake_stripe_subscription("price_pm_monthly")):
            stripe_events.handle_checkout_session_completed(event, sb)

        assert identify_calls == [(TEST_USER_ID, {"plan": "pro_max"})]
        assert capture_calls[0][1] == "subscription_activated"
        assert capture_calls[0][2]["tier"] == "pro_max"


# ---------------------------------------------------------------------------
# handle_checkout_session_completed (mode="payment") — one-time credit packs
# ---------------------------------------------------------------------------


def _topup_event(
    session_id="cs_top_1", user_id=TEST_USER_ID, pack_key="pack_500", payment_status="paid", event_id="evt_top_1"
):
    session = MagicMock()
    session.mode = "payment"
    session.id = session_id
    session.metadata = {"user_id": user_id, "pack_key": pack_key, "target": "user"}
    session.payment_status = payment_status
    event = MagicMock()
    event.id = event_id
    event.data.object = session
    return event


class TestTopupCompleted:
    def _sb(self):
        return _mock_supabase(
            {
                "credit_packs": [{"key": "pack_500", "credits": 500, "price_cents": 1000}],
                "credit_wallets": [{"id": "w-top", "owner_type": "user", "owner_id": TEST_USER_ID}],
            }
        )

    def test_grants_purchase_on_paid_session(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        sb, _ = self._sb()
        stripe_events.handle_checkout_session_completed(_topup_event(), sb)
        name, params = sb.rpc.call_args[0]
        assert name == "grant_credits"
        assert params["p_amount"] == 500
        assert params["p_kind"] == "purchase"
        assert params["p_bucket"] == "reserve"
        assert params["p_request_id"] == "topup:cs_top_1"

    def test_async_redelivery_same_session_same_key(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        sb, _ = self._sb()
        stripe_events.handle_checkout_session_completed(_topup_event(event_id="evt_DIFFERENT"), sb)
        assert sb.rpc.call_args[0][1]["p_request_id"] == "topup:cs_top_1"

    def test_unpaid_session_skips_grant(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        sb, _ = self._sb()
        stripe_events.handle_checkout_session_completed(_topup_event(payment_status="unpaid"), sb)
        sb.rpc.assert_not_called()

    def test_unknown_pack_logs_and_returns(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        sb, _ = _mock_supabase({"credit_packs": []})
        stripe_events.handle_checkout_session_completed(_topup_event(pack_key="nope"), sb)
        sb.rpc.assert_not_called()

    def test_duplicate_grant_skips_analytics(self, monkeypatch):
        """Redelivery under a different event id: grant_credits reports
        duplicate=True — topup_purchased must NOT fire again (would
        double-count pack revenue in PostHog)."""
        import subscriptions.stripe_events as stripe_events

        sb, _ = self._sb()
        sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": True, "balance_after": 500})
        with patch("subscriptions.stripe_events.analytics_capture") as mock_capture:
            stripe_events.handle_checkout_session_completed(_topup_event(event_id="evt_DIFFERENT"), sb)

        assert not any(c.args[1] == "topup_purchased" for c in mock_capture.call_args_list)

    # No test_subscription_mode_untouched stub here: the existing
    # TestHandleCheckoutSessionCompletedCredits suite above IS the regression
    # guard for the mode dispatch — those events use plain MagicMock sessions
    # where `.mode` is an auto-attribute (not the string "payment"), so
    # `getattr(session, "mode", None) == "payment"` is False and they never
    # enter the topup branch. Covered by running that suite (see report).


# ---------------------------------------------------------------------------
# _handle_topup_completed — org-pool branch (Licensing Phase B, Task 8)
# ---------------------------------------------------------------------------


def _org_topup_event(
    session_id="cs_org_top_1",
    user_id=TEST_USER_ID,
    org_id="org-1",
    pack_key="pack_500",
    payment_status="paid",
    event_id="evt_org_top_1",
):
    """Same shape as _topup_event, but metadata['target'] is an org id
    (billing_router.create_topup_session sets this after admin-gating)."""
    session = MagicMock()
    session.mode = "payment"
    session.id = session_id
    session.metadata = {"user_id": user_id, "pack_key": pack_key, "target": org_id}
    session.payment_status = payment_status
    event = MagicMock()
    event.id = event_id
    event.data.object = session
    return event


class TestOrgTopupCompleted:
    """metadata['target'] is an org id -> credits land in the org's POOL
    wallet (never the personal wallet), and crossing the cumulative
    activation floor flips a 'pending' org to 'active' (spec rule 3)."""

    ORG_ID = "org-1"
    WALLET_ID = "w-org-pool"

    def _sb(self, org_status="pending", min_initial=None, ledger_rows=None):
        return _mock_supabase(
            {
                "credit_packs": [{"key": "pack_500", "credits": 500, "price_cents": 1000}],
                "credit_wallets": [{"id": self.WALLET_ID, "owner_type": "org", "owner_id": self.ORG_ID}],
                "organizations": [
                    {"id": self.ORG_ID, "status": org_status, "min_initial_purchase_credits": min_initial}
                ],
                "credit_ledger": ledger_rows or [],
            }
        )

    def test_grant_call_shape_targets_org_pool_wallet(self, monkeypatch):
        """Wallet id in the grant call is the ORG pool wallet (Task 4's
        read_or_create_org_wallet), never a personal wallet — and the
        idempotency key/kind/bucket match the personal path exactly."""
        import subscriptions.stripe_events as stripe_events

        sb, builders = self._sb()
        stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        grant_calls = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"]
        assert len(grant_calls) == 1
        params = grant_calls[0].args[1]
        assert params["p_wallet_id"] == self.WALLET_ID
        assert params["p_amount"] == 500
        assert params["p_kind"] == "purchase"
        assert params["p_bucket"] == "reserve"
        assert params["p_request_id"] == "topup:cs_org_top_1"
        assert params["p_metadata"]["org_id"] == self.ORG_ID
        assert params["p_metadata"]["pack_key"] == "pack_500"
        # Never touches the user-wallet seeding path (no create-on-miss
        # insert -- the pool wallet already exists in this fixture).
        builders["credit_wallets"].insert.assert_not_called()

    def test_analytics_fires_with_org_target_and_org_id(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        sb, _ = self._sb()
        with patch("subscriptions.stripe_events.analytics_capture") as mock_capture:
            stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        capture_calls = [c for c in mock_capture.call_args_list if c.args[1] == "topup_purchased"]
        assert len(capture_calls) == 1
        actor, name, props = capture_calls[0].args
        assert actor == TEST_USER_ID  # the buying admin, not the org
        assert props["target"] == "org"
        assert props["org_id"] == self.ORG_ID
        assert props["credits"] == 500

    def test_below_floor_purchase_grants_but_stays_pending(self, monkeypatch):
        """Cumulative purchases (3000) are under the default 10000 floor:
        the grant lands, but the org's status is never written."""
        import subscriptions.stripe_events as stripe_events

        sb, builders = self._sb(org_status="pending", ledger_rows=[{"delta": 3000}])
        stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        assert any(c.args[0] == "grant_credits" for c in sb.rpc.call_args_list)
        builders["organizations"].update.assert_not_called()

    def test_crossing_floor_flips_status_to_active(self, monkeypatch):
        """Cumulative purchases (12000) cross the default 10000 floor ->
        UPDATE status='active' (activation is CUMULATIVE, not last-purchase)."""
        import subscriptions.stripe_events as stripe_events

        sb, builders = self._sb(org_status="pending", ledger_rows=[{"delta": 12000}])
        stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        builders["organizations"].update.assert_called_once_with({"status": "active"})

    def test_already_active_org_never_updated(self, monkeypatch):
        """Already-active org: no status write, even when cumulative
        purchases are far past the floor (activation only moves
        pending -> active, never re-asserted)."""
        import subscriptions.stripe_events as stripe_events

        sb, builders = self._sb(org_status="active", ledger_rows=[{"delta": 999_999}])
        stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        builders["organizations"].update.assert_not_called()

    def test_suspended_org_never_updated(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        sb, builders = self._sb(org_status="suspended", ledger_rows=[{"delta": 999_999}])
        stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        builders["organizations"].update.assert_not_called()

    def test_duplicate_replay_skips_analytics_but_activation_still_runs(self, monkeypatch):
        """Duplicate grant (async redelivery): topup_purchased must NOT fire
        again, but the activation re-check still runs (harmless — the sum is
        unchanged either way) and still flips a crossing org to active."""
        import subscriptions.stripe_events as stripe_events

        sb, builders = self._sb(org_status="pending", ledger_rows=[{"delta": 12000}])
        sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": True, "balance_after": 500})

        with patch("subscriptions.stripe_events.analytics_capture") as mock_capture:
            stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        assert not any(c.args[1] == "topup_purchased" for c in mock_capture.call_args_list)
        builders["organizations"].update.assert_called_once_with({"status": "active"})

    def test_org_specific_min_initial_purchase_credits_overrides_env_default(self, monkeypatch):
        """The org's own min_initial_purchase_credits (500) wins over the
        ENTERPRISE_MIN_INITIAL_CREDITS env default (10000)."""
        import subscriptions.stripe_events as stripe_events

        monkeypatch.setenv("ENTERPRISE_MIN_INITIAL_CREDITS", "10000")
        sb, builders = self._sb(org_status="pending", min_initial=500, ledger_rows=[{"delta": 600}])
        stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        builders["organizations"].update.assert_called_once_with({"status": "active"})

    def test_falls_back_to_env_default_when_org_min_is_null(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        monkeypatch.setenv("ENTERPRISE_MIN_INITIAL_CREDITS", "200")
        sb, builders = self._sb(org_status="pending", min_initial=None, ledger_rows=[{"delta": 250}])
        stripe_events.handle_checkout_session_completed(_org_topup_event(org_id=self.ORG_ID), sb)

        builders["organizations"].update.assert_called_once_with({"status": "active"})

    def test_unpaid_session_skips_grant(self, monkeypatch):
        import subscriptions.stripe_events as stripe_events

        sb, _ = self._sb()
        stripe_events.handle_checkout_session_completed(
            _org_topup_event(org_id=self.ORG_ID, payment_status="unpaid"), sb
        )
        sb.rpc.assert_not_called()

    def test_user_target_tests_unaffected(self, monkeypatch):
        """Sanity check that org branch dispatch is keyed strictly off
        target != 'user' — a personal-target session run through the same
        handler still hits the pre-Phase-B personal-wallet path."""
        import subscriptions.stripe_events as stripe_events

        sb, builders = _mock_supabase(
            {
                "credit_packs": [{"key": "pack_500", "credits": 500, "price_cents": 1000}],
                "credit_wallets": [{"id": "w-personal", "owner_type": "user", "owner_id": TEST_USER_ID}],
            }
        )
        stripe_events.handle_checkout_session_completed(_topup_event(), sb)

        grant_calls = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"]
        assert len(grant_calls) == 1
        assert grant_calls[0].args[1]["p_wallet_id"] == "w-personal"
        assert "organizations" not in builders  # activation check never runs for the personal path


# ---------------------------------------------------------------------------
# handle_subscription_updated — tier sync + upgrade top-up
# ---------------------------------------------------------------------------


def _build_tier_update_sb(
    prev_tier: str,
    new_tier_grant: int,
    wallet_bundle: int,
    wallet_id: str = "wallet-1",
    granted_this_period: int = 0,
):
    """granted_this_period seeds a monthly_grant ledger row inside the wallet period
    so tests can exercise the period-sum anti-farming cap."""
    ledger_rows = (
        [{"delta": granted_this_period, "kind": "monthly_grant", "created_at": "2026-07-02T00:00:00+00:00"}]
        if granted_this_period
        else []
    )
    return _mock_supabase(
        {
            "subscriptions": [{"tier": prev_tier}],
            "tier_entitlements": [{"monthly_credits": new_tier_grant}],
            "credit_wallets": [
                {"id": wallet_id, "bundle_balance": wallet_bundle, "period_start": "2026-07-01T00:00:00+00:00"}
            ],
            "credit_ledger": ledger_rows,
        }
    )


class TestSubscriptionUpdatedTierSync:
    def test_upgrade_pro_to_pro_max_tops_up_bundle(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(prev_tier="pro", new_tier_grant=8000, wallet_bundle=2500)
        event = _subscription_event("customer.subscription.updated", price_id="price_pm_monthly")

        stripe_events.handle_subscription_updated(event, sb)

        update_payload = builders["subscriptions"].update.call_args[0][0]
        assert update_payload["tier"] == "pro_max"

        grant_calls = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"]
        assert len(grant_calls) == 1
        payload = grant_calls[0].args[1]
        assert payload["p_wallet_id"] == "wallet-1"
        assert payload["p_amount"] == 5500  # 8000 (new grant) - 2500 (existing bundle)
        assert payload["p_kind"] == "monthly_grant"
        assert payload["p_bucket"] == "bundle"
        assert payload["p_request_id"] == f"tier-upgrade:{event.id}"

    def test_repeat_upgrade_within_period_grants_nothing(self, monkeypatch):
        """Bundle already at the new tier's grant (farming guard): top_up clamps to 0."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(prev_tier="pro", new_tier_grant=8000, wallet_bundle=8000)
        event = _subscription_event("customer.subscription.updated", price_id="price_pm_monthly")

        stripe_events.handle_subscription_updated(event, sb)

        assert not any(c.args[0] == "grant_credits" for c in sb.rpc.call_args_list)

    def test_spent_bundle_after_full_grant_grants_nothing(self, monkeypatch):
        """Spend→downgrade→re-upgrade loop: bundle is 0 (all spent) but the ledger
        shows the full 8000 already granted this period → NO new grant. This is
        the config-independent guard — it holds even if the Stripe portal ever
        applies downgrades immediately instead of at period end."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(
            prev_tier="pro", new_tier_grant=8000, wallet_bundle=0, granted_this_period=8000
        )
        event = _subscription_event("customer.subscription.updated", price_id="price_pm_monthly")

        stripe_events.handle_subscription_updated(event, sb)

        assert not any(c.args[0] == "grant_credits" for c in sb.rpc.call_args_list)

    def test_legitimate_upgrade_after_partial_spend_caps_by_period_sum(self, monkeypatch):
        """Pro user granted 3000 this period, spent down to 500, upgrades to pro_max:
        top_up = min(8000 - 500, 8000 - 3000) = 5000 — spent credits are not refunded,
        but the user still reaches the pro_max grant total for the period."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(
            prev_tier="pro", new_tier_grant=8000, wallet_bundle=500, granted_this_period=3000
        )
        event = _subscription_event("customer.subscription.updated", price_id="price_pm_monthly")

        stripe_events.handle_subscription_updated(event, sb)

        grant_calls = [c for c in sb.rpc.call_args_list if c.args[0] == "grant_credits"]
        assert len(grant_calls) == 1
        assert grant_calls[0].args[1]["p_amount"] == 5000

    def test_upgrade_grant_failure_propagates_so_stripe_retries(self, monkeypatch):
        """No swallow: a failed top-up must raise so the webhook router 500s and
        Stripe redelivers (the grant is request-id idempotent, so retry is safe)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(prev_tier="pro", new_tier_grant=8000, wallet_bundle=2500)
        sb.rpc.return_value.execute.side_effect = RuntimeError("grant_credits down")
        event = _subscription_event("customer.subscription.updated", price_id="price_pm_monthly")

        with pytest.raises(RuntimeError, match="grant_credits down"):
            stripe_events.handle_subscription_updated(event, sb)

    def test_downgrade_pro_max_to_pro_syncs_tier_no_grant(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(prev_tier="pro_max", new_tier_grant=3000, wallet_bundle=100)
        event = _subscription_event("customer.subscription.updated", price_id="price_monthly_123")

        stripe_events.handle_subscription_updated(event, sb)

        update_payload = builders["subscriptions"].update.call_args[0][0]
        assert update_payload["tier"] == "pro"
        sb.rpc.assert_not_called()

    def test_unchanged_price_no_grant(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(prev_tier="pro", new_tier_grant=3000, wallet_bundle=100)
        event = _subscription_event("customer.subscription.updated", price_id="price_monthly_123")

        stripe_events.handle_subscription_updated(event, sb)

        update_payload = builders["subscriptions"].update.call_args[0][0]
        assert update_payload["tier"] == "pro"
        sb.rpc.assert_not_called()

    def test_credits_disabled_does_not_sync_tier_or_call_rpc(self, monkeypatch):
        """CREDITS_ENABLED unset (default) — tier is NOT synced (pre-credits
        behavior, clean rollback) and no grant logic runs.

        The pro_max env var IS set here, so the upgrade branch (pro→pro_max,
        low bundle) WOULD sync tier and grant if the credits_enabled() gate
        were removed — pins that the gate is load-bearing on both."""
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pm_monthly")
        from subscriptions import stripe_events

        sb, builders = _build_tier_update_sb(prev_tier="pro", new_tier_grant=8000, wallet_bundle=100)
        event = _subscription_event("customer.subscription.updated", price_id="price_pm_monthly")

        stripe_events.handle_subscription_updated(event, sb)

        update_payload = builders["subscriptions"].update.call_args[0][0]
        assert "tier" not in update_payload  # tier write is gated behind credits_enabled()
        assert update_payload["status"] == "active"  # other fields still sync
        sb.rpc.assert_not_called()
        assert "credit_wallets" not in builders  # wallet never even read


# ---------------------------------------------------------------------------
# overage_billing.bill_overage_row / bill_pending_overage
# ---------------------------------------------------------------------------


def _customer_sb(customer="cus_123"):
    return _mock_supabase({"subscriptions": [{"stripe_customer_id": customer}]})


class TestBillOverageRow:
    def test_already_billed_row_skips_stripe_call(self):
        from subscriptions import overage_billing

        sb, builders = _customer_sb()
        row = {"id": "ledger-1", "metadata": {"invoice_item_id": "ii_existing"}, "delta": 0}

        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe") as mock_get_stripe:
            result = overage_billing.bill_overage_row(sb, TEST_USER_ID, row)

        assert result == "ii_existing"
        mock_get_stripe.assert_not_called()

    def test_unbilled_row_with_credits_billed_creates_invoice_item(self, monkeypatch):
        monkeypatch.setenv("CREDIT_OVERAGE_USD", "0.02")
        from subscriptions import overage_billing

        sb, builders = _customer_sb()
        row = {"id": "ledger-2", "metadata": {"credits_billed": 21}, "delta": 0, "action": "oneclick_run"}

        fake_item = MagicMock(id="ii_new_1")
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = fake_item
        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe):
            result = overage_billing.bill_overage_row(sb, TEST_USER_ID, row)

        assert result == "ii_new_1"
        create_kwargs = fake_stripe.InvoiceItem.create.call_args.kwargs
        assert create_kwargs["amount"] == 42  # 21 credits * $0.02 * 100 cents
        assert create_kwargs["customer"] == "cus_123"
        assert create_kwargs["currency"] == "usd"
        # Stripe-side idempotency: sweep + invoice.created racing on the same
        # row cannot double-charge — Stripe rejects the reused key.
        assert create_kwargs["idempotency_key"] == "overage:ledger-2"

        update_payload = builders["credit_ledger"].update.call_args[0][0]
        assert update_payload["metadata"]["invoice_item_id"] == "ii_new_1"

    def test_legacy_row_without_credits_billed_falls_back_to_negative_delta(self, monkeypatch):
        monkeypatch.setenv("CREDIT_OVERAGE_USD", "0.02")
        from subscriptions import overage_billing

        sb, builders = _customer_sb()
        row = {"id": "ledger-3", "metadata": {}, "delta": -21, "action": "oneclick_run"}

        fake_item = MagicMock(id="ii_legacy_1")
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = fake_item
        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe):
            result = overage_billing.bill_overage_row(sb, TEST_USER_ID, row)

        assert result == "ii_legacy_1"
        assert fake_stripe.InvoiceItem.create.call_args.kwargs["amount"] == 42

    def test_zero_delta_no_credits_billed_skips_and_returns_none(self):
        from subscriptions import overage_billing

        sb, builders = _customer_sb()
        row = {"id": "ledger-4", "metadata": {}, "delta": 0, "action": "oneclick_run"}

        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe") as mock_get_stripe:
            result = overage_billing.bill_overage_row(sb, TEST_USER_ID, row)

        assert result is None
        mock_get_stripe.assert_not_called()

    def test_invoice_id_is_passed_through_to_stripe(self):
        from subscriptions import overage_billing

        sb, builders = _customer_sb()
        row = {"id": "ledger-5", "metadata": {"credits_billed": 10}, "delta": 0, "action": "zoe_message"}

        fake_item = MagicMock(id="ii_5")
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = fake_item
        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe):
            overage_billing.bill_overage_row(sb, TEST_USER_ID, row, invoice_id="in_draft_1")

        assert fake_stripe.InvoiceItem.create.call_args.kwargs["invoice"] == "in_draft_1"

    def test_no_stripe_customer_skips_and_returns_none(self):
        from subscriptions import overage_billing

        sb, builders = _mock_supabase({"subscriptions": []})
        row = {"id": "ledger-6", "metadata": {"credits_billed": 10}, "delta": 0}

        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe") as mock_get_stripe:
            result = overage_billing.bill_overage_row(sb, TEST_USER_ID, row)

        assert result is None
        mock_get_stripe.assert_not_called()

    def test_invoice_attached_row_is_stamped_swept(self):
        from subscriptions.overage_billing import bill_overage_row

        sb, builders = _customer_sb()
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = MagicMock(id="ii_new")
        row = {"id": "lr1", "delta": 0, "action": "zoe_message", "metadata": {"credits_billed": 3}}
        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe):
            bill_overage_row(sb, TEST_USER_ID, row, invoice_id="in_123")
        updated_meta = builders["credit_ledger"].update.call_args[0][0]["metadata"]
        assert updated_meta["invoice_item_id"] == "ii_new"
        # Attached directly to a real invoice → already consumed; without this
        # stamp the annual sweep retries an empty standalone invoice forever.
        assert updated_meta["swept"] is True

    def test_floating_row_is_not_stamped_swept(self):
        from subscriptions.overage_billing import bill_overage_row

        sb, builders = _customer_sb()
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = MagicMock(id="ii_new")
        row = {"id": "lr1", "delta": 0, "action": "zoe_message", "metadata": {"credits_billed": 3}}
        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe):
            bill_overage_row(sb, TEST_USER_ID, row)
        updated_meta = builders["credit_ledger"].update.call_args[0][0]["metadata"]
        assert "swept" not in updated_meta  # floating item — step 4 sweeps it later


class TestBillPendingOverage:
    def test_bills_only_unbilled_rows(self):
        from subscriptions import overage_billing

        sb, builders = _mock_supabase(
            {
                "credit_wallets": [{"id": "wallet-1"}],
                "credit_ledger": [
                    {"id": "l1", "metadata": {"credits_billed": 5}, "delta": 0, "action": "zoe_message"},
                    {"id": "l2", "metadata": {"invoice_item_id": "ii_already"}, "delta": 0},
                ],
                "subscriptions": [{"stripe_customer_id": "cus_123"}],
            }
        )
        fake_item = MagicMock(id="ii_new")
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = fake_item
        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe):
            billed = overage_billing.bill_pending_overage(sb, TEST_USER_ID)

        assert billed == 1
        fake_stripe.InvoiceItem.create.assert_called_once()
        # Query is bounded to unbilled rows via the JSON-path filter (the
        # Python-side re-check above is what this mock actually exercised).
        builders["credit_ledger"].is_.assert_called_once_with("metadata->invoice_item_id", "null")

    def test_no_wallet_returns_zero(self):
        from subscriptions import overage_billing

        sb, builders = _mock_supabase({"credit_wallets": []})
        billed = overage_billing.bill_pending_overage(sb, TEST_USER_ID)
        assert billed == 0

    def test_passes_invoice_id_through_to_each_row(self):
        from subscriptions import overage_billing

        sb, builders = _mock_supabase(
            {
                "credit_wallets": [{"id": "wallet-1"}],
                "credit_ledger": [
                    {"id": "l1", "metadata": {"credits_billed": 5}, "delta": 0, "action": "zoe_message"},
                ],
                "subscriptions": [{"stripe_customer_id": "cus_123"}],
            }
        )
        fake_item = MagicMock(id="ii_new")
        fake_stripe = MagicMock()
        fake_stripe.InvoiceItem.create.return_value = fake_item
        with patch("subscriptions.overage_billing.stripe_client_module.get_stripe", return_value=fake_stripe):
            overage_billing.bill_pending_overage(sb, TEST_USER_ID, invoice_id="in_draft_7")

        assert fake_stripe.InvoiceItem.create.call_args.kwargs["invoice"] == "in_draft_7"


# ---------------------------------------------------------------------------
# handle_invoice_created — safety net for stragglers
# ---------------------------------------------------------------------------


class TestHandleInvoiceCreated:
    def test_non_subscription_cycle_is_noop(self):
        from subscriptions import stripe_events

        sb = MagicMock()
        event = MagicMock()
        event.data.object.billing_reason = "manual"
        event.data.object.customer = "cus_1"

        stripe_events.handle_invoice_created(event, sb)
        sb.table.assert_not_called()

    def test_missing_customer_is_noop(self):
        from subscriptions import stripe_events

        sb = MagicMock()
        event = MagicMock()
        event.data.object.billing_reason = "subscription_cycle"
        event.data.object.customer = None

        stripe_events.handle_invoice_created(event, sb)
        sb.table.assert_not_called()

    def test_unknown_customer_is_noop(self, monkeypatch):
        # Credits ON so the unknown-customer branch (not the flag gate) is what
        # short-circuits — the credits_enabled() check now runs first.
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb, builders = _mock_supabase({"subscriptions": []})
        event = MagicMock()
        event.data.object.billing_reason = "subscription_cycle"
        event.data.object.customer = "cus_unknown"
        event.data.object.id = "in_1"

        with patch("subscriptions.overage_billing.bill_pending_overage") as mock_bill:
            stripe_events.handle_invoice_created(event, sb)

        mock_bill.assert_not_called()

    def test_credits_disabled_is_noop(self):
        from subscriptions import stripe_events

        sb, builders = _mock_supabase({"subscriptions": [{"user_id": TEST_USER_ID}]})
        event = MagicMock()
        event.data.object.billing_reason = "subscription_cycle"
        event.data.object.customer = "cus_1"
        event.data.object.id = "in_1"

        with patch("subscriptions.overage_billing.bill_pending_overage") as mock_bill:
            stripe_events.handle_invoice_created(event, sb)

        mock_bill.assert_not_called()
        # Gate is hoisted above the DB read — no subscriptions query either.
        assert "subscriptions" not in builders

    def test_happy_path_calls_bill_pending_overage_with_invoice_id(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        from subscriptions import stripe_events

        sb, builders = _mock_supabase({"subscriptions": [{"user_id": TEST_USER_ID}]})
        event = MagicMock()
        event.data.object.billing_reason = "subscription_cycle"
        event.data.object.customer = "cus_1"
        event.data.object.id = "in_draft_99"

        with patch("subscriptions.overage_billing.bill_pending_overage") as mock_bill:
            stripe_events.handle_invoice_created(event, sb)

        mock_bill.assert_called_once_with(sb, TEST_USER_ID, invoice_id="in_draft_99")

    def test_registered_in_handlers_dict(self):
        from subscriptions.stripe_events import HANDLERS, handle_invoice_created

        assert HANDLERS.get("invoice.created") is handle_invoice_created


# ---------------------------------------------------------------------------
# handle_subscription_deleted — final billing on cancellation (review finding #1)
# ---------------------------------------------------------------------------


def _deleted_event(user_id=TEST_USER_ID, customer="cus_del_1", event_id="evt_del_1"):
    sub = MagicMock()
    sub.metadata = {"user_id": user_id}
    sub.customer = customer
    sub.canceled_at = 1752000000
    event = MagicMock()
    event.id = event_id
    event.data.object = sub
    return event


class TestSubscriptionDeletedFinalBilling:
    def test_final_billing_runs_when_credits_on(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        import subscriptions.stripe_events as stripe_events

        sb, _ = _mock_supabase(
            {
                "subscriptions": [{"user_id": TEST_USER_ID, "stripe_customer_id": "cus_del_1"}],
                "credit_wallets": [{"id": "w-del", "owner_type": "user", "owner_id": TEST_USER_ID}],
            }
        )
        with (
            patch("subscriptions.overage_billing.bill_pending_overage", return_value=0) as bpo,
            patch(
                "subscriptions.overage_billing.invoice_unswept_items",
                return_value={"invoiced": True, "stamped": 2},
            ) as inv,
        ):
            stripe_events.handle_subscription_deleted(_deleted_event(), sb)
        bpo.assert_called_once_with(sb, TEST_USER_ID)
        inv.assert_called_once_with(sb, "w-del", "cus_del_1", idempotency_key="final:evt_del_1")

    def test_credits_off_no_billing_calls(self, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        import subscriptions.stripe_events as stripe_events

        sb, _ = _mock_supabase({"subscriptions": [{"user_id": TEST_USER_ID}]})
        with (
            patch("subscriptions.overage_billing.bill_pending_overage") as bpo,
            patch("subscriptions.overage_billing.invoice_unswept_items") as inv,
        ):
            stripe_events.handle_subscription_deleted(_deleted_event(), sb)
        bpo.assert_not_called()
        inv.assert_not_called()

    def test_no_wallet_skips_final_invoice(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        import subscriptions.stripe_events as stripe_events

        sb, _ = _mock_supabase(
            {"subscriptions": [{"user_id": TEST_USER_ID, "stripe_customer_id": "cus_del_1"}], "credit_wallets": []}
        )
        with (
            patch("subscriptions.overage_billing.bill_pending_overage", return_value=0),
            patch("subscriptions.overage_billing.invoice_unswept_items") as inv,
        ):
            stripe_events.handle_subscription_deleted(_deleted_event(), sb)
        inv.assert_not_called()

    def test_no_customer_id_anywhere_skips_both_but_tier_still_updates(self, monkeypatch):
        """No customer id on the event AND none in the subscriptions row → both
        billing calls are skipped, but the tier/status write (unconditional,
        earlier in the function) still happens."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        import subscriptions.stripe_events as stripe_events

        sb, builders = _mock_supabase(
            {
                "subscriptions": [{"user_id": TEST_USER_ID}],  # no stripe_customer_id key
                "credit_wallets": [{"id": "w-del"}],
            }
        )
        event = _deleted_event(customer=None)
        with (
            patch("subscriptions.overage_billing.bill_pending_overage") as bpo,
            patch("subscriptions.overage_billing.invoice_unswept_items") as inv,
        ):
            stripe_events.handle_subscription_deleted(event, sb)
        bpo.assert_not_called()
        inv.assert_not_called()
        update_payload = builders["subscriptions"].update.call_args[0][0]
        assert update_payload["tier"] == "free"
        assert update_payload["status"] == "canceled"

    def test_billing_failure_propagates_so_stripe_retries(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        import subscriptions.stripe_events as stripe_events

        sb, _ = _mock_supabase(
            {
                "subscriptions": [{"user_id": TEST_USER_ID, "stripe_customer_id": "cus_del_1"}],
                "credit_wallets": [{"id": "w-del"}],
            }
        )
        with (
            patch("subscriptions.overage_billing.bill_pending_overage", return_value=0),
            patch(
                "subscriptions.overage_billing.invoice_unswept_items",
                side_effect=RuntimeError("stripe down"),
            ),
            pytest.raises(RuntimeError),
        ):
            stripe_events.handle_subscription_deleted(_deleted_event(), sb)


# ---------------------------------------------------------------------------
# billing_router — plan -> env price mapping (pro_max_monthly / pro_max_annual)
# ---------------------------------------------------------------------------


class TestCreateCheckoutSessionProMaxPlans:
    def _set_env(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_MONTHLY", "price_pro_max_monthly_test")
        monkeypatch.setenv("STRIPE_PRICE_PRO_MAX_ANNUAL", "price_pro_max_annual_test")
        monkeypatch.setenv("FRONTEND_URL", "http://localhost:8080")
        from subscriptions import stripe_client

        monkeypatch.setattr(stripe_client, "_initialized", False)

    def test_pro_max_monthly_resolves_pro_max_monthly_price(self, client, mock_supabase, monkeypatch):
        self._set_env(monkeypatch)

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_test_pm_monthly")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "pro_max_monthly"})

        assert resp.status_code == 200, resp.text
        assert m.call_args.kwargs["line_items"][0]["price"] == "price_pro_max_monthly_test"

    def test_pro_max_annual_resolves_pro_max_annual_price(self, client, mock_supabase, monkeypatch):
        self._set_env(monkeypatch)

        fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/cs_test_pm_annual")
        with patch("stripe.checkout.Session.create", return_value=fake_session) as m:
            resp = client.post("/billing/create-checkout-session", json={"plan": "pro_max_annual"})

        assert resp.status_code == 200, resp.text
        assert m.call_args.kwargs["line_items"][0]["price"] == "price_pro_max_annual_test"

    def test_still_rejects_invalid_plan(self, client, mock_supabase, monkeypatch):
        self._set_env(monkeypatch)

        resp = client.post("/billing/create-checkout-session", json={"plan": "weekly"})
        assert resp.status_code == 400
