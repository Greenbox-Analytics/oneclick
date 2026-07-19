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
