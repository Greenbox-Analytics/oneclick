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
        assert "/organization?topup=success" in kwargs["success_url"]
        assert "/organization?topup=canceled" in kwargs["cancel_url"]

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
