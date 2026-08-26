"""Top-up packs: listing + one-time checkout session (spec 2026-07-19 §3)."""

from unittest.mock import MagicMock, patch

import orgs.authz as orgs_authz
from tests.conftest import TEST_USER_ID, MockQueryBuilder, _default_table_side_effect

ORG_ID = "10000000-0000-0000-0000-0000000000aa"

PACK_ROW = {
    "key": "pack_500",
    "credits": 500,
    "price_cents": 1000,
    "sort_order": 1,
    "active": True,
    "stripe_price_id": "price_topup_500",
}


class TestListCreditPacks:
    def test_returns_active_configured_packs(self, client, mock_supabase):
        # credit_packs is NOT in conftest's `_SUBSCRIPTION_TABLES`, so it needs
        # explicit wiring per test; everything else falls back to the conftest
        # default (unused here since this GET is unauthenticated).
        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(
                    data=[{k: PACK_ROW[k] for k in ("key", "credits", "price_cents", "sort_order")}]
                )
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        resp = client.get("/billing/credit-packs")
        assert resp.status_code == 200
        assert resp.json()["packs"][0]["key"] == "pack_500"

    def test_lists_a_pack_with_no_stripe_price(self, client, mock_supabase):
        """Listed on `active` ALONE. The old stripe_price_id filter left the
        whole ladder unsellable until an operator hand-created Prices in
        Stripe; the line item is now built ad-hoc from price_cents."""

        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(
                    data=[
                        {
                            **{k: PACK_ROW[k] for k in ("key", "credits", "price_cents", "sort_order")},
                            "label": "Starter",
                        }
                    ]
                )
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        body = client.get("/billing/credit-packs").json()
        assert body["packs"][0]["label"] == "Starter"

    def test_ships_custom_bounds_and_tool_prices(self, client, mock_supabase, monkeypatch):
        """The picker quotes "what this typically buys" off `prices` and sizes
        its custom-amount input off `custom` — one fetch, no hardcoded numbers."""
        monkeypatch.delenv("CREDIT_OVERAGE_USD", raising=False)

        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[])
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        body = client.get("/billing/credit-packs").json()
        assert body["custom"] == {"minCredits": 250, "maxCredits": 100000, "perCreditCents": 2}
        # Keys must mirror Entitlements.to_dict()'s `prices` block exactly, so
        # one frontend type serves both payloads.
        assert set(body["prices"]) == {"zoeMessage", "oneclickRun", "registryParse", "splitSheet"}

    def test_omits_prices_when_the_table_is_empty(self, client, mock_supabase):
        """Degrade to no subtitle rather than quoting "0 OneClick runs"."""

        def _side_effect(name):
            b = MockQueryBuilder()
            if name in ("credit_packs", "credit_prices"):
                b.execute.return_value = MagicMock(data=[])
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        assert "prices" not in client.get("/billing/credit-packs").json()


class TestCreateTopupSession:
    def test_409_when_credits_disabled(self, client, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500"})
        assert resp.status_code == 409

    def test_400_unknown_pack(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")

        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[])
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        resp = client.post("/billing/create-topup-session", json={"pack_key": "nope"})
        assert resp.status_code == 400

    def test_creates_payment_mode_session(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("FRONTEND_URL", "https://app.test")

        # conftest's default `subscriptions` row (_PRO_SUB_ROW) carries no
        # stripe_customer_id, so wire one explicitly here — this pins the
        # customer-attach branch (the charge must land on the user's existing
        # Customer, e.g. for refund lookups) rather than falling back to
        # customer_email.
        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[PACK_ROW])
                return b
            if name == "subscriptions":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(
                    data=[{"user_id": TEST_USER_ID, "stripe_customer_id": "cus_existing"}]
                )
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/xyz")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500"})
        assert resp.status_code == 200
        assert resp.json()["url"] == "https://checkout.stripe/xyz"
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs["mode"] == "payment"
        assert kwargs["line_items"] == [{"price": "price_topup_500", "quantity": 1}]
        assert kwargs["metadata"]["pack_key"] == "pack_500"
        assert kwargs["metadata"]["target"] == "user"
        assert "/profile?topup=success" in kwargs["success_url"]
        # wired a stripe_customer_id above →
        # the charge must attach to the existing Customer (refund lookups).
        assert kwargs.get("customer")
        assert "customer_email" not in kwargs

    def test_falls_back_to_customer_email_without_stripe_customer(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("FRONTEND_URL", "https://app.test")

        # Wire BOTH tables this endpoint reads: pack lookup + a subscriptions
        # row with no stripe_customer_id (e.g. free user buying their first pack).
        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[PACK_ROW])
                return b
            if name == "subscriptions":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[{"user_id": TEST_USER_ID, "stripe_customer_id": None}])
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/xyz")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500"})
        assert resp.status_code == 200
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs.get("customer_email")
        assert "customer" not in kwargs


class TestCreateTopupSessionCustomAmount:
    """Custom credit amounts: the client sends a COUNT, the server prices it."""

    def _side_effect(self, name):
        if name == "credit_packs":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[PACK_ROW])
            return b
        if name == "subscriptions":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"user_id": TEST_USER_ID, "stripe_customer_id": "cus_existing"}])
            return b
        return _default_table_side_effect(name)

    def _checkout(self, client, monkeypatch, body):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("FRONTEND_URL", "https://app.test")
        monkeypatch.delenv("CREDIT_OVERAGE_USD", raising=False)
        monkeypatch.delenv("STRIPE_CREDITS_PRODUCT_ID", raising=False)
        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/custom")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = client.post("/billing/create-topup-session", json=body)
        return resp, fake_stripe

    def test_prices_the_amount_server_side(self, client, mock_supabase, monkeypatch):
        mock_supabase.table.side_effect = self._side_effect
        resp, fake_stripe = self._checkout(client, monkeypatch, {"credits": 1300})

        assert resp.status_code == 200
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs["mode"] == "payment"
        # 1,300 credits at the $0.02 list rate = $26.00.
        assert kwargs["line_items"] == [
            {
                "price_data": {
                    "currency": "usd",
                    "unit_amount": 2600,
                    "product_data": {"name": "1,300 Msanii credits"},
                },
                "quantity": 1,
            }
        ]
        # No pack_key (there is no catalog row) and no `kind` — `kind` is the
        # org_topup discriminator the subscription handlers branch on first.
        assert kwargs["metadata"] == {"user_id": TEST_USER_ID, "credits": "1300", "target": "user"}
        assert "kind" not in kwargs["metadata"]
        assert kwargs["success_url"] == "https://app.test/profile?topup=success"

    def test_rejects_an_amount_below_the_minimum(self, client, mock_supabase, monkeypatch):
        mock_supabase.table.side_effect = self._side_effect
        resp, _ = self._checkout(client, monkeypatch, {"credits": 10})
        assert resp.status_code == 400
        assert "250" in resp.json()["detail"]

    def test_rejects_an_amount_above_the_ceiling(self, client, mock_supabase, monkeypatch):
        mock_supabase.table.side_effect = self._side_effect
        resp, _ = self._checkout(client, monkeypatch, {"credits": 100_001})
        assert resp.status_code == 400

    def test_rejects_both_pack_and_amount(self, client, mock_supabase, monkeypatch):
        mock_supabase.table.side_effect = self._side_effect
        resp, _ = self._checkout(client, monkeypatch, {"credits": 500, "pack_key": PACK_ROW["key"]})
        assert resp.status_code == 422

    def test_rejects_neither(self, client, mock_supabase, monkeypatch):
        mock_supabase.table.side_effect = self._side_effect
        resp, _ = self._checkout(client, monkeypatch, {})
        assert resp.status_code == 422

    def _org_side_effect(self, name):
        if name == "organizations":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"archived_at": None}])
            return b
        return self._side_effect(name)

    def test_custom_into_an_org_pool_targets_the_pool(self, client, mock_supabase, monkeypatch):
        """An org admin can buy a custom amount into the pool, same as a
        bundle — metadata.target carries the org id so fulfilment grants the
        POOL wallet, and the return path is the /teams console."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._org_side_effect

        resp, fake_stripe = self._checkout(client, monkeypatch, {"credits": 500, "org_id": ORG_ID})
        assert resp.status_code == 200
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs["metadata"] == {"user_id": TEST_USER_ID, "credits": "500", "target": ORG_ID}
        assert kwargs["line_items"][0]["price_data"]["unit_amount"] == 1000  # $10.00, server-priced
        assert kwargs["success_url"] == "https://app.test/teams?topup=success"

    def test_custom_org_purchase_still_requires_admin(self, client, mock_supabase, monkeypatch):
        """The org gates run before the bundle/custom fork, so the custom
        product sits behind the same require_admin wall as bundles."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: False)
        mock_supabase.table.side_effect = self._org_side_effect

        resp, _ = self._checkout(client, monkeypatch, {"credits": 500, "org_id": ORG_ID})
        assert resp.status_code == 403


class TestCreateTopupSessionAdHocPack:
    """A bundle with no operator-configured Stripe Price is still sellable —
    the line item is built from `price_cents` instead."""

    def test_builds_price_data_from_the_catalog_row(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("FRONTEND_URL", "https://app.test")
        monkeypatch.delenv("STRIPE_CREDITS_PRODUCT_ID", raising=False)

        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[{**PACK_ROW, "stripe_price_id": None}])
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/adhoc")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = client.post("/billing/create-topup-session", json={"pack_key": PACK_ROW["key"]})

        assert resp.status_code == 200
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs["line_items"] == [
            {
                "price_data": {
                    "currency": "usd",
                    "unit_amount": PACK_ROW["price_cents"],
                    "product_data": {"name": f"{PACK_ROW['credits']:,} Msanii credits"},
                },
                "quantity": 1,
            }
        ]
        # Metadata is unchanged from the configured-Price path, so fulfilment
        # (which reads pack_key and re-reads credits from the catalog) is too.
        assert kwargs["metadata"] == {"user_id": TEST_USER_ID, "pack_key": PACK_ROW["key"], "target": "user"}

    def test_inactive_pack_is_still_unsellable(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")

        def _side_effect(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[{**PACK_ROW, "active": False}])
                return b
            return _default_table_side_effect(name)

        mock_supabase.table.side_effect = _side_effect

        resp = client.post("/billing/create-topup-session", json={"pack_key": PACK_ROW["key"]})
        assert resp.status_code == 400


class TestCreateTopupSessionOrgTarget:
    """Licensing Phase B: `org_id` in the body routes the SAME pack purchase
    into that org's pool instead of the caller's personal wallet.
    Admin-gated, flag-gated, and blocked on an archived org — everything
    else (pack lookup, customer resolution) is shared with the user path."""

    def _side_effect(self, org_row=None):
        def _fn(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[PACK_ROW])
                return b
            if name == "subscriptions":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(
                    data=[{"user_id": TEST_USER_ID, "stripe_customer_id": "cus_existing"}]
                )
                return b
            if name == "organizations":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[org_row] if org_row is not None else [])
                return b
            return _default_table_side_effect(name)

        return _fn

    def test_403_when_caller_is_not_org_admin(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: False)
        mock_supabase.table.side_effect = self._side_effect()

        resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500", "org_id": ORG_ID})
        assert resp.status_code == 403

    def test_404_when_licensing_flag_off(self, client, mock_supabase, monkeypatch):
        """404, not 409/403 — same "don't reveal the feature" stance as the
        /orgs/* router-level gate. Caller isn't even checked for admin-ness:
        the flag gate is hoisted above the authz check."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        mock_supabase.table.side_effect = self._side_effect()

        resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500", "org_id": ORG_ID})
        assert resp.status_code == 404

    def test_409_when_org_is_archived(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(org_row={"archived_at": "2026-07-01T00:00:00+00:00"})

        resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500", "org_id": ORG_ID})
        assert resp.status_code == 409
        assert resp.json()["detail"] == "This organization is archived."

    def test_admin_happy_path_targets_org_pool(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("FRONTEND_URL", "https://app.test")
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(org_row={"archived_at": None})

        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/org")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500", "org_id": ORG_ID})

        assert resp.status_code == 200
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs["metadata"]["target"] == ORG_ID
        assert kwargs["metadata"]["user_id"] == TEST_USER_ID
        assert kwargs["metadata"]["pack_key"] == "pack_500"
        assert "/teams?topup=success" in kwargs["success_url"]
        assert "/teams?topup=canceled" in kwargs["cancel_url"]

    def test_user_flow_identical_kwargs_when_org_id_omitted(self, client, mock_supabase, monkeypatch):
        """Regression pin: with no org_id, kwargs must match the pre-Phase-B
        assertions exactly (mirrors
        TestCreateTopupSession.test_creates_payment_mode_session)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("FRONTEND_URL", "https://app.test")
        mock_supabase.table.side_effect = self._side_effect()

        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/xyz")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = client.post("/billing/create-topup-session", json={"pack_key": "pack_500"})

        assert resp.status_code == 200
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs["mode"] == "payment"
        assert kwargs["line_items"] == [{"price": "price_topup_500", "quantity": 1}]
        assert kwargs["metadata"] == {"user_id": TEST_USER_ID, "pack_key": "pack_500", "target": "user"}
        assert kwargs["success_url"] == "https://app.test/profile?topup=success"
        assert kwargs["cancel_url"] == "https://app.test/profile?topup=canceled"
        assert kwargs.get("customer") == "cus_existing"
        assert "customer_email" not in kwargs


# ---------------------------------------------------------------------------
# Recurring org top-up (spec 2026-08-15 §4.3, Task 11)
# ---------------------------------------------------------------------------

RECURRING_PACK_ROW = {**PACK_ROW, "recurring_stripe_price_id": "price_rec_500"}
LIVE_ORG = {
    "kind": "self_serve",
    "archived_at": None,
    "dissolved_at": None,
    "topup_stripe_subscription_id": None,
}


class TestCreditPacksCatalogRecurring:
    """The pack listing doubles as the recurring-top-up catalog: a pack is
    monthly-buyable exactly when the operator set recurring_stripe_price_id
    on it (Migration 20260816000002). No second catalog endpoint."""

    def _packs(self, rows):
        def _fn(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=rows)
                return b
            return _default_table_side_effect(name)

        return _fn

    def test_exposes_recurring_price_id(self, client, mock_supabase):
        mock_supabase.table.side_effect = self._packs([RECURRING_PACK_ROW])
        resp = client.get("/billing/credit-packs")
        assert resp.status_code == 200
        assert resp.json()["packs"][0]["recurringPriceId"] == "price_rec_500"

    def test_null_when_operator_has_not_configured_one(self, client, mock_supabase):
        mock_supabase.table.side_effect = self._packs([PACK_ROW])
        resp = client.get("/billing/credit-packs")
        assert resp.status_code == 200
        pack = resp.json()["packs"][0]
        assert pack["key"] == "pack_500"
        assert pack["recurringPriceId"] is None


class TestOrgTopupCheckout:
    """POST /billing/org-topup-checkout — the SAME pack bought as a monthly
    Stripe SUBSCRIPTION on the purchasing admin's personal customer, refilling
    the ORG pool each period.

    The metadata contract is load-bearing (review r2): BOTH the session's
    metadata (read by handle_checkout_session_completed) and
    subscription_data.metadata (copied by Stripe onto the Subscription and
    every invoice's subscription_details.metadata) carry
    {org_id, kind: 'org_topup', purchased_by} — and NEITHER carries user_id,
    which is what keeps the personal-subscription handlers off these events.
    """

    def _side_effect(self, org_row=None, pack=RECURRING_PACK_ROW, customer="cus_existing"):
        def _fn(name):
            if name == "credit_packs":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[pack] if pack else [])
                return b
            if name == "subscriptions":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[{"user_id": TEST_USER_ID, "stripe_customer_id": customer}])
                return b
            if name == "organizations":
                b = MockQueryBuilder()
                b.execute.return_value = MagicMock(data=[org_row] if org_row is not None else [])
                return b
            return _default_table_side_effect(name)

        return _fn

    def _post(self, client, key="pack_500"):
        return client.post("/billing/org-topup-checkout", json={"org_id": ORG_ID, "key": key})

    def _flags_on(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("FRONTEND_URL", "https://app.test")

    def test_404_when_licensing_flag_off(self, client, mock_supabase, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        mock_supabase.table.side_effect = self._side_effect(org_row=LIVE_ORG)
        assert self._post(client).status_code == 404

    def test_409_when_credits_flag_off(self, client, mock_supabase, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        mock_supabase.table.side_effect = self._side_effect(org_row=LIVE_ORG)
        assert self._post(client).status_code == 409

    def test_403_when_caller_is_not_an_active_admin(self, client, mock_supabase, monkeypatch):
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: False)
        mock_supabase.table.side_effect = self._side_effect(org_row=LIVE_ORG)
        assert self._post(client).status_code == 403

    def test_409_on_enterprise_org(self, client, mock_supabase, monkeypatch):
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(org_row={**LIVE_ORG, "kind": "enterprise"})
        assert self._post(client).status_code == 409

    def test_409_on_archived_org(self, client, mock_supabase, monkeypatch):
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(
            org_row={**LIVE_ORG, "archived_at": "2026-08-01T00:00:00+00:00"}
        )
        assert self._post(client).status_code == 409

    def test_409_when_org_already_has_a_topup(self, client, mock_supabase, monkeypatch):
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(
            org_row={**LIVE_ORG, "topup_stripe_subscription_id": "sub_existing"}
        )
        assert self._post(client).status_code == 409

    def test_404_when_pack_has_no_recurring_price(self, client, mock_supabase, monkeypatch):
        """A pack that exists but was never given a recurring Stripe price is
        not part of this catalog — 404, reading identically to an unknown key
        (PACK_ROW is the one-time-only pack)."""
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(org_row=LIVE_ORG, pack=PACK_ROW)
        assert self._post(client).status_code == 404

    def test_404_when_key_is_unknown(self, client, mock_supabase, monkeypatch):
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(org_row=LIVE_ORG, pack=None)
        assert self._post(client, key="nope").status_code == 404

    def test_creates_subscription_mode_session_with_both_metadata_objects(self, client, mock_supabase, monkeypatch):
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(org_row=LIVE_ORG)

        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/topup")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = self._post(client)

        assert resp.status_code == 200
        assert resp.json()["url"] == "https://checkout.stripe/topup"
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs["mode"] == "subscription"
        assert kwargs["line_items"] == [{"price": "price_rec_500", "quantity": 1}]
        expected = {"org_id": ORG_ID, "kind": "org_topup", "purchased_by": TEST_USER_ID}
        assert kwargs["metadata"] == expected
        assert kwargs["subscription_data"]["metadata"] == expected
        # NO user_id in either object — that key is what routes an event into
        # the personal-subscription handlers.
        assert "user_id" not in kwargs["metadata"]
        assert "user_id" not in kwargs["subscription_data"]["metadata"]
        # Same customer-resolution block as the pack path.
        assert kwargs.get("customer") == "cus_existing"
        assert "customer_email" not in kwargs
        assert "/teams?topup=success" in kwargs["success_url"]

    def test_falls_back_to_customer_email(self, client, mock_supabase, monkeypatch):
        self._flags_on(monkeypatch)
        monkeypatch.setattr(orgs_authz, "is_org_admin", lambda *a: True)
        mock_supabase.table.side_effect = self._side_effect(org_row=LIVE_ORG, customer=None)

        fake_stripe = MagicMock()
        fake_stripe.checkout.Session.create.return_value = MagicMock(url="https://checkout.stripe/topup")
        with patch("subscriptions.billing_router.stripe_client_module.get_stripe", return_value=fake_stripe):
            resp = self._post(client)

        assert resp.status_code == 200
        kwargs = fake_stripe.checkout.Session.create.call_args.kwargs
        assert kwargs.get("customer_email")
        assert "customer" not in kwargs
