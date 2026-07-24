"""Wallet plumbing + code-level flag retirement in EntitlementsService."""

from unittest.mock import MagicMock

from subscriptions.models import CreditGrant
from subscriptions.service import EntitlementsService
from tests.conftest import (
    _DEFAULT_CREDIT_PRICES,
    _DEFAULT_WALLET_ROW,
    _PRO_TIER_ROW,
    TEST_USER_ID,
    _default_table_side_effect,
)

# Licensing Phase B (Task 6) reuses test_billing_context's filter-aware org mock.
from tests.test_billing_context import (
    FAR_FUTURE,
    ORG,
    PRICES,
    PRO_TIER_ROW,
    _ctx_store,
    _ctx_supabase,
    _member,
    _org,
    _profile,
    _seat_wallet,
    _user_wallet,
)
from tests.test_billing_context import _sub_row as _ctx_sub_row  # avoid shadowing this file's _sub_row(user_id)
from tests.test_billing_sweep import _FilterBuilder

FREE_TIER_ROW = {
    "tier": "free",
    "max_artists": 3,
    "max_projects": 3,
    "max_boards": -1,
    "max_tasks": 50,
    "max_storage_bytes": 1073741824,
    "max_split_sheets_per_month": 5,
    "max_oneclick_runs_per_month": 1,
    "zoe_enabled": False,
    "oneclick_enabled": True,
    "registry_enabled": False,
    "integrations_allowed": ["google_drive"],
    "monthly_credits": 50,
    "max_works": 10,
    "included_storage_bytes": 1073741824,
}

FREE_SUB_ROW = {
    "user_id": TEST_USER_ID,
    "tier": "free",
    "status": "active",
    "overage_enabled": False,
    "overage_cap_credits": None,
    "storage_overage_enabled": False,
}


def _free_supabase():
    sb = MagicMock()

    def side_effect(name):
        b = _default_table_side_effect(name)
        if name == "tier_entitlements":
            b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
        elif name == "subscriptions":
            b.execute.return_value = MagicMock(data=[dict(FREE_SUB_ROW)], count=1)
        return b

    sb.table.side_effect = side_effect
    sb.rpc.return_value.execute.return_value = MagicMock(data=True)
    return sb


class TestFlagRetirement:
    def test_flags_forced_true_when_credits_enabled(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(_free_supabase()).get_for_user(TEST_USER_ID)
        assert ent.features.zoe_enabled is True
        assert ent.features.registry_enabled is True

    def test_flags_untouched_when_credits_disabled(self):
        ent = EntitlementsService(_free_supabase()).get_for_user(TEST_USER_ID)
        assert ent.features.zoe_enabled is False
        assert ent.features.registry_enabled is False


class TestWalletInEntitlements:
    def test_credits_info_populated(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(_free_supabase()).get_for_user(TEST_USER_ID)
        assert ent.credits is not None
        assert ent.credits.balance == _DEFAULT_WALLET_ROW["bundle_balance"]
        assert ent.credits.monthly_grant == 50  # free tier grant
        assert ent.credits.prices["zoe_message"] == 3

    def test_credits_none_when_disabled(self):
        ent = EntitlementsService(_free_supabase()).get_for_user(TEST_USER_ID)
        assert ent.credits is None


class TestWalletRollover:
    def test_expired_period_triggers_rollover_rpc(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = MagicMock()
        expired = dict(_DEFAULT_WALLET_ROW, period_end="2020-01-01T00:00:00+00:00")

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "credit_wallets":
                b.execute.return_value = MagicMock(data=[expired], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[dict(FREE_SUB_ROW)], count=1)
            return b

        sb.table.side_effect = side_effect
        sb.rpc.return_value.execute.return_value = MagicMock(data=True)
        EntitlementsService(sb).get_for_user(TEST_USER_ID)
        rpc_names = [c.args[0] for c in sb.rpc.call_args_list]
        assert "rollover_wallet" in rpc_names
        args = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"][0]
        assert args["p_monthly_grant"] == 50

    def test_future_period_no_rollover(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _free_supabase()
        EntitlementsService(sb).get_for_user(TEST_USER_ID)
        assert not any(c.args[0] == "rollover_wallet" for c in sb.rpc.call_args_list)


class TestAdminWalletGrant:
    def test_admin_expired_wallet_rolls_over_at_tier_grant_not_zero(self, monkeypatch):
        """The admin caps patch must never leak into the wallet grant (0-grant bug)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = MagicMock()
        expired = dict(_DEFAULT_WALLET_ROW, period_end="2020-01-01T00:00:00+00:00")

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "credit_wallets":
                b.execute.return_value = MagicMock(data=[expired], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[dict(FREE_SUB_ROW)], count=1)
            elif name == "profiles":
                b.execute.return_value = MagicMock(data=[{"is_admin": True}], count=1)
            return b

        sb.table.side_effect = side_effect
        sb.rpc.return_value.execute.return_value = MagicMock(data=True)
        ent = EntitlementsService(sb).get_for_user(TEST_USER_ID)
        args = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"][0]
        assert args["p_monthly_grant"] == 50  # the tier's grant, NOT the admin patch's sentinel
        assert ent.caps.monthly_credits == -1  # display keeps the unlimited sentinel


class TestDisabledPathIsolation:
    def test_disabled_path_touches_zero_credit_tables(self):
        """The rollback guarantee: flag off => no wallet/price/ledger reads, no RPCs."""
        sb = _free_supabase()
        EntitlementsService(sb).get_for_user(TEST_USER_ID)
        tables_touched = {c.args[0] for c in sb.table.call_args_list}
        assert tables_touched.isdisjoint({"credit_wallets", "credit_prices", "credit_ledger"})
        assert sb.rpc.call_args_list == []


# ---------------------------------------------------------------------------
# check_credits / debit_for_action / can() credits policies
# ---------------------------------------------------------------------------


def _paid_supabase(
    bundle=0,
    reserve=0,
    overage_enabled=False,
    cap=None,
    tier="pro",
    overage_used=0,
    storage_overage_enabled=False,
    status="active",
):
    sb = MagicMock()
    wallet = dict(
        _DEFAULT_WALLET_ROW,
        bundle_balance=bundle,
        reserve_balance=reserve,
        overage_this_period=overage_used,
    )

    def side_effect(name):
        b = _default_table_side_effect(name)
        if name == "credit_wallets":
            b.execute.return_value = MagicMock(data=[wallet], count=1)
        elif name == "subscriptions":
            b.execute.return_value = MagicMock(
                data=[
                    {
                        "user_id": TEST_USER_ID,
                        "tier": tier,
                        "status": status,
                        "overage_enabled": overage_enabled,
                        "overage_cap_credits": cap,
                        "storage_overage_enabled": storage_overage_enabled,
                    }
                ],
                count=1,
            )
        return b

    sb.table.side_effect = side_effect
    sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False, "balance_after": 0})
    return sb


class TestCheckCredits:
    def test_disabled_allows_free_price_zero(self):
        r = EntitlementsService(_paid_supabase()).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.price == 0

    def test_sufficient_balance_allows(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        r = EntitlementsService(_paid_supabase(bundle=100)).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and not r.use_overage and r.price == 3

    def test_reserve_counts_toward_balance(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        r = EntitlementsService(_paid_supabase(bundle=0, reserve=25)).check_credits(TEST_USER_ID, "oneclick_run")
        assert r.allowed and not r.use_overage  # 25 >= 21

    def test_insufficient_paid_overage_enabled_allows_via_overage(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        r = EntitlementsService(_paid_supabase(bundle=0, overage_enabled=True)).check_credits(
            TEST_USER_ID, "oneclick_run"
        )
        assert r.allowed and r.use_overage

    def test_overage_cap_blocks(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0, overage_enabled=True, cap=10)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "oneclick_run")
        assert not r.allowed and r.reason  # 0 + 21 > 10

    def test_overage_cap_counts_prior_usage(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0, overage_enabled=True, cap=30, overage_used=15)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "oneclick_run")
        assert not r.allowed  # 15 + 21 > 30

    def test_overage_under_cap_allows(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        # cap 100, used 0, price 21 → 21 <= 100 → allowed via overage
        sb = _paid_supabase(bundle=0, overage_enabled=True, cap=100)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "oneclick_run")
        assert r.allowed and r.use_overage

    def test_overage_exactly_at_cap_allows(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        # used 79, price 21 → 79+21 == 100 == cap → boundary must ALLOW (guards a > vs >= flip)
        sb = _paid_supabase(bundle=0, overage_enabled=True, cap=100, overage_used=79)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "oneclick_run")
        assert r.allowed and r.use_overage

    def test_overage_one_over_cap_denies(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        # used 80, price 21 → 101 > 100 → denied
        sb = _paid_supabase(bundle=0, overage_enabled=True, cap=100, overage_used=80)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "oneclick_run")
        assert not r.allowed

    def test_insufficient_paid_not_enabled_offers_overage(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        r = EntitlementsService(_paid_supabase(bundle=0)).check_credits(TEST_USER_ID, "zoe_message")
        assert not r.allowed and r.overage_available and not r.upgrade_required
        assert r.reset_date is not None

    def test_insufficient_free_requires_upgrade(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = MagicMock()

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "credit_wallets":
                b.execute.return_value = MagicMock(data=[dict(_DEFAULT_WALLET_ROW, bundle_balance=0)], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[dict(FREE_SUB_ROW)], count=1)
            return b

        sb.table.side_effect = side_effect
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert not r.allowed and r.upgrade_required and not r.overage_available

    def test_admin_short_circuits_before_wallet(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message", is_admin=True)
        assert r.allowed and r.price == 0
        tables_touched = {c.args[0] for c in sb.table.call_args_list}
        assert "credit_wallets" not in tables_touched

    def test_degraded_paid_fails_open(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        orig = sb.table.side_effect

        def flaky(name):
            if name == "credit_wallets":
                raise RuntimeError("db down")
            return orig(name)

        sb.table.side_effect = flaky
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.degraded and r.price == 0

    def test_degraded_free_fails_closed(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = MagicMock()

        def flaky(name):
            if name == "credit_wallets":
                raise RuntimeError("db down")
            b = _default_table_side_effect(name)
            if name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[dict(FREE_SUB_ROW)], count=1)
            return b

        sb.table.side_effect = flaky
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert not r.allowed and r.degraded

    def test_free_with_hostile_overage_enabled_row_never_gets_overage(self, monkeypatch):
        """overage_enabled on a free-tier sub row must never unlock overage."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0, overage_enabled=True, tier="free")
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            return b

        sb.table.side_effect = side_effect
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert not r.allowed and r.upgrade_required
        assert not r.use_overage and not r.overage_available

    def test_missing_tier_row_fails_degraded_without_rollover(self, monkeypatch):
        """Missing tier row must fail loud (degraded), never a destructive 0-grant rollover."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0)
        orig = sb.table.side_effect
        expired = dict(_DEFAULT_WALLET_ROW, bundle_balance=0, period_end="2020-01-01T00:00:00+00:00")

        def side_effect(name):
            b = orig(name)
            if name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "credit_wallets":
                b.execute.return_value = MagicMock(data=[expired], count=1)
            return b

        sb.table.side_effect = side_effect
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.degraded  # paid fails open
        assert not any(c.args[0] == "rollover_wallet" for c in sb.rpc.call_args_list)

    def test_missing_price_denies_all_tiers_not_degraded(self, monkeypatch):
        """Unseeded action is a config error: explicit deny, NOT a degraded outage."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "credit_prices":
                b.execute.return_value = MagicMock(data=[{"action": "oneclick_run", "credits": 21}], count=1)
            return b

        sb.table.side_effect = side_effect
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert not r.allowed and not r.degraded
        assert "support" in r.reason.lower()

    def test_subscription_read_failure_fails_closed(self, monkeypatch):
        """Tier is unknowable without the sub row → conservative deny, even for paid users."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        orig = sb.table.side_effect

        def flaky(name):
            if name == "subscriptions":
                raise RuntimeError("db down")
            return orig(name)

        sb.table.side_effect = flaky
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert not r.allowed and r.degraded

    def test_zero_price_action_always_allowed(self, monkeypatch):
        """A retuned-to-0 price (even with a negative balance) must never wall the action."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=-5)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "credit_prices":
                b.execute.return_value = MagicMock(data=[{"action": "zoe_message", "credits": 0}], count=1)
            return b

        sb.table.side_effect = side_effect
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.price == 0

    def test_price_read_failure_routes_to_degraded_not_config_deny(self, monkeypatch):
        """A price-table READ outage is degraded (paid open); a missing KEY is a
        config deny — this pins the split so a refactor can't collapse it."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        orig = sb.table.side_effect

        def flaky(name):
            if name == "credit_prices":
                raise RuntimeError("db down")
            return orig(name)

        sb.table.side_effect = flaky
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.degraded and r.price == 0

    def test_db_admin_leg_short_circuits(self, monkeypatch):
        """profiles.is_admin=True (DB leg, not the is_admin param) skips the wallet entirely."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "profiles":
                b.execute.return_value = MagicMock(data=[{"is_admin": True}], count=1)
            return b

        sb.table.side_effect = side_effect
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.price == 0
        tables_touched = {c.args[0] for c in sb.table.call_args_list}
        assert "credit_wallets" not in tables_touched


class TestPastDueOveragePause:
    """past_due must pause pay-per-use (a failing card must not accrue more
    debt) while spending an existing balance stays allowed."""

    def test_past_due_blocks_overage_even_when_opted_in(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0, overage_enabled=True, status="past_due")
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed is False
        assert r.use_overage is False
        assert r.overage_available is False
        assert "payment" in (r.reason or "").lower()

    def test_past_due_still_spends_existing_balance(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100, status="past_due")
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed is True
        assert r.use_overage is False

    def test_active_overage_path_unchanged(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0, overage_enabled=True, status="active")
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed is True
        assert r.use_overage is True


class TestDebitForAction:
    def test_debits_via_rpc(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        grant = CreditGrant(request_id="req-1", action="zoe_message", price=3, kind="debit", enabled=True)
        EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        args = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "debit_credits"][0]
        assert args["p_amount"] == 3 and args["p_request_id"] == "req-1" and args["p_kind"] == "debit"

    def test_overage_kind_passed_through(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        grant = CreditGrant(request_id="req-ov", action="oneclick_run", price=21, kind="overage_debit", enabled=True)
        EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        args = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "debit_credits"][0]
        assert args["p_kind"] == "overage_debit"

    def test_disabled_grant_is_noop(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        grant = CreditGrant(request_id="req-2", action="zoe_message", price=3, kind="debit", enabled=False)
        EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        assert not any(c.args[0] == "debit_credits" for c in sb.rpc.call_args_list)

    def test_rpc_failure_never_raises(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        sb.rpc.side_effect = RuntimeError("db down")
        grant = CreditGrant(request_id="req-3", action="zoe_message", price=3, kind="debit", enabled=True)
        EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)  # no raise


class TestStorageIncludedCheck:
    def test_paid_over_included_without_optin_denied(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "usage_counters":
                b.execute.return_value = MagicMock(
                    data=[
                        {
                            "user_id": TEST_USER_ID,
                            "total_storage_bytes": 107374182400,
                            "split_sheets_this_period": 0,
                            "zoe_queries_this_period": 0,
                            "oneclick_runs_this_period": 0,
                            "period_start": "2026-05-09T00:00:00+00:00",
                            "period_end": "2099-05-09T00:00:00+00:00",
                        }
                    ],
                    count=1,
                )
            return b

        sb.table.side_effect = side_effect
        from subscriptions.models import Action

        r = EntitlementsService(sb).can(TEST_USER_ID, Action.UPLOAD_BYTES, size=1)
        assert not r.allowed

    def test_paid_over_included_with_optin_allowed(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        # paid, storage_overage_enabled=True, usage at included, +1 byte → ALLOWED (opt-in)
        sb = _paid_supabase(bundle=100, storage_overage_enabled=True)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "usage_counters":
                b.execute.return_value = MagicMock(
                    data=[
                        {
                            "user_id": TEST_USER_ID,
                            "total_storage_bytes": 107374182400,
                            "split_sheets_this_period": 0,
                            "zoe_queries_this_period": 0,
                            "oneclick_runs_this_period": 0,
                            "period_start": "2026-05-09T00:00:00+00:00",
                            "period_end": "2099-05-09T00:00:00+00:00",
                        }
                    ],
                    count=1,
                )
            return b

        sb.table.side_effect = side_effect
        from subscriptions.models import Action

        r = EntitlementsService(sb).can(TEST_USER_ID, Action.UPLOAD_BYTES, size=1)
        assert r.allowed

    def test_disabled_flag_keeps_legacy_unlimited(self):
        sb = _paid_supabase(bundle=100)
        from subscriptions.models import Action

        r = EntitlementsService(sb).can(TEST_USER_ID, Action.UPLOAD_BYTES, size=10**12)
        assert r.allowed  # legacy: pro max_storage_bytes == -1 → unlimited


class TestMaxWorksCap:
    def test_at_cap_denied(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _free_supabase()
        from subscriptions.models import Action

        r = EntitlementsService(sb).can(TEST_USER_ID, Action.CREATE_WORK, current_count=10)
        assert not r.allowed  # free max_works = 10

    def test_under_cap_allowed(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _free_supabase()
        from subscriptions.models import Action

        r = EntitlementsService(sb).can(TEST_USER_ID, Action.CREATE_WORK, current_count=9)
        assert r.allowed


# ---------------------------------------------------------------------------
# Owner-scoping regression guard — a dropped .eq("owner_id", user_id) on the
# wallet read is a cross-tenant leak. The no-op mock used everywhere else in
# this file can't catch that (it ignores filters entirely and just returns
# whatever row was configured), so this uses the ACTUALLY-filtering builder
# from tests/test_billing_sweep.py instead.
# ---------------------------------------------------------------------------


def _filter_aware_supabase(table_data: dict):
    sb = MagicMock()
    sb.table.side_effect = lambda name: _FilterBuilder(list(table_data.get(name, [])))
    sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False, "balance_after": 0})
    return sb


USER_A = TEST_USER_ID
USER_B = "00000000-0000-0000-0000-000000000099"


def _sub_row(user_id):
    return {
        "user_id": user_id,
        "tier": "pro",
        "status": "active",
        "overage_enabled": False,
        "overage_cap_credits": None,
        "storage_overage_enabled": False,
    }


class TestWalletOwnerScoping:
    def test_check_credits_scopes_wallet_to_caller_not_other_owner(self, monkeypatch):
        """Two wallets, two owners: A has 100 credits, B has 0. If the
        .eq("owner_id", user_id) filter on the wallet read were ever dropped,
        both reads would collapse onto whichever wallet row sorts first,
        letting B piggyback on A's balance (or vice versa). Assert BOTH
        directions so the bug can't hide behind row ordering."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _filter_aware_supabase(
            {
                "profiles": [],
                "subscriptions": [_sub_row(USER_A), _sub_row(USER_B)],
                "tier_entitlements": [_PRO_TIER_ROW],
                "tier_overrides": [],
                "credit_prices": list(_DEFAULT_CREDIT_PRICES),
                "credit_wallets": [
                    dict(_DEFAULT_WALLET_ROW, id="wallet-a", owner_id=USER_A, bundle_balance=100, reserve_balance=0),
                    dict(_DEFAULT_WALLET_ROW, id="wallet-b", owner_id=USER_B, bundle_balance=0, reserve_balance=0),
                ],
            }
        )
        r_a = EntitlementsService(sb).check_credits(USER_A, "zoe_message")
        r_b = EntitlementsService(sb).check_credits(USER_B, "zoe_message")
        assert r_a.allowed  # sees A's own bundle (100 >= price 3)
        assert not r_b.allowed  # sees B's own bundle (0 < price 3), NOT A's 100

    def test_read_or_create_wallet_scopes_to_caller(self, monkeypatch):
        """Lower-level pin on _read_or_create_wallet directly (not just via
        check_credits): the same filter-aware mock, same two-wallet setup."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _filter_aware_supabase(
            {
                "credit_wallets": [
                    dict(_DEFAULT_WALLET_ROW, id="wallet-a", owner_id=USER_A, bundle_balance=100, reserve_balance=0),
                    dict(_DEFAULT_WALLET_ROW, id="wallet-b", owner_id=USER_B, bundle_balance=0, reserve_balance=0),
                ],
            }
        )
        wallet_a = EntitlementsService(sb)._read_or_create_wallet(USER_A)
        wallet_b = EntitlementsService(sb)._read_or_create_wallet(USER_B)
        assert wallet_a["id"] == "wallet-a" and wallet_a["bundle_balance"] == 100
        assert wallet_b["id"] == "wallet-b" and wallet_b["bundle_balance"] == 0


# ---------------------------------------------------------------------------
# Licensing Phase B — Task 6: check_credits seat path + wallet-targeted debits.
#
# Reuses test_billing_context's FILTER-AWARE mock so org-context reads (profiles
# preference, org_members seat, organizations status, seat wallet) resolve
# correctly and a personal-wallet select can be PROVEN absent. The shared no-op
# MockQueryBuilder used elsewhere in this file ignores .eq(), so it could never
# distinguish a seat wallet from a user wallet.
# ---------------------------------------------------------------------------

# test_billing_context.USER == TEST_USER_ID (the mock scopes everything to it).
CTX_USER = TEST_USER_ID


def _org_check_data(seat_wallets, *, org_status="active", member_status="active"):
    """Data for an org-context check_credits: profiles preference + active seat +
    org + prices + the given seat wallet row(s). Deliberately OMITS the personal
    wallet / subscription / tier tables — the org seat path must never read them."""
    return {
        "profiles": [_profile(context_org=ORG)],
        "org_members": [_member(status=member_status)],
        "organizations": [_org(status=org_status)],
        "credit_prices": list(PRICES),
        "credit_wallets": list(seat_wallets),
    }


def _personal_via_dead_org_data(*, org_status="suspended", member_status="active"):
    """An org preference that resolves DEAD (or pending) → check_credits runs the
    PERSONAL path. Provides the personal subscription/tier/wallet the fallback needs."""
    return {
        "profiles": [_profile(context_org=ORG)],
        "org_members": [_member(status=member_status)],
        "organizations": [_org(status=org_status)],
        "subscriptions": [_ctx_sub_row(tier="pro")],
        "tier_entitlements": [PRO_TIER_ROW],
        "tier_overrides": [],
        "credit_prices": list(PRICES),
        "credit_wallets": [_user_wallet(bundle=100, period_end=FAR_FUTURE)],
    }


class TestCheckCreditsOrgContext:
    def test_seat_pays_allowed_with_seat_wallet_id(self, monkeypatch):
        """Funded seat → allowed, price from the shared prices table, wallet_id is
        the SEAT wallet, managed_by_org True, and NONE of the personal-context
        fields (overage / upgrade / reset_date) are set (rule 8)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_seat_wallet(reserve=500)]))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed and r.managed_by_org is True
        assert r.wallet_id == "wallet-seat"
        assert r.price == 3
        assert r.use_overage is False and r.overage_available is False
        assert r.upgrade_required is False
        assert r.reset_date is None
        # Every seat-wallet select filtered owner_type='seat'; the personal wallet
        # (owner_type='user') is NEVER read.
        wallet_queries = sb._log.get("credit_wallets", [])
        assert wallet_queries
        for preds in wallet_queries:
            assert ("eq", "owner_type", "seat") in preds
            assert ("eq", "owner_type", "user") not in preds
        # The org seat path consulted NO personal subscription row.
        assert "subscriptions" not in sb._log

    def test_seat_pays_then_debit_targets_seat_wallet(self, monkeypatch):
        """The end-to-end money path: check → build grant from the result → debit.
        The debit RPC receives the SEAT wallet id, and the debit path adds no
        personal-wallet select (rule 9)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_seat_wallet(reserve=500)]))
        svc = EntitlementsService(sb)

        r = svc.check_credits(CTX_USER, "zoe_message")
        grant = CreditGrant(
            request_id="req-seat",
            action="zoe_message",
            price=r.price,
            kind="debit",
            enabled=True,
            wallet_id=r.wallet_id,
        )
        svc.debit_for_action(CTX_USER, grant)

        debit_calls = [c for c in sb.rpc.call_args_list if c.args and c.args[0] == "debit_credits"]
        assert len(debit_calls) == 1
        assert debit_calls[0].args[1]["p_wallet_id"] == "wallet-seat"
        # No personal-wallet ('user') select anywhere across check + debit.
        for preds in sb._log.get("credit_wallets", []):
            assert ("eq", "owner_type", "user") not in preds

    def test_seat_dry_402_shape(self, monkeypatch):
        """Empty seat → 402-shaped result: managed_by_org True, the ask-your-admin
        reason, and NO overage / upgrade / reset_date fields (rule 8)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_seat_wallet(reserve=0, bundle=0)]))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is False and r.managed_by_org is True
        assert r.reason == "You've used the credits your organization allocated. Ask your admin for more."
        assert r.overage_available is False and r.use_overage is False
        assert r.upgrade_required is False
        assert r.reset_date is None
        assert r.degraded is False  # a dry seat is a legitimate wall, not an outage

    def test_missing_seat_wallet_lazy_creates_zero_then_402(self, monkeypatch):
        """No seat wallet ROW at all → lazy-create at zero → 402 (rule 8's carve-out:
        a missing row is NOT the READ-ERROR fail-open case)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([]))  # zero wallets

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is False and r.managed_by_org is True
        assert r.degraded is False
        assert r.reason == "You've used the credits your organization allocated. Ask your admin for more."

    def test_seat_wallet_read_error_fails_open_uncharged(self, monkeypatch):
        """A seat-wallet READ EXCEPTION fails OPEN uncharged (price 0, degraded),
        like the paid personal tier (spec §12)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        table_fn, _log, _updates, _store = _ctx_store(_org_check_data([_seat_wallet(reserve=500)]))
        sb = MagicMock()

        def _table(name):
            if name == "credit_wallets":
                raise RuntimeError("seat wallet read exploded")
            return table_fn(name)

        sb.table.side_effect = _table
        sb.rpc.return_value.execute.return_value = MagicMock(data=True)

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is True and r.degraded is True and r.price == 0
        assert r.managed_by_org is True

    def test_suspended_org_context_uses_personal_wallet(self, monkeypatch):
        """A suspended org is a DEAD reference → personal check path, personal
        wallet id, no managed_by_org."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_personal_via_dead_org_data(org_status="suspended"))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is True and r.managed_by_org is False
        assert r.wallet_id == "wallet-personal"

    def test_pending_org_context_uses_personal_wallet(self, monkeypatch):
        """A pending org confers nothing yet (rule 7) → personal check path."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_personal_via_dead_org_data(org_status="pending"))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is True and r.managed_by_org is False
        assert r.wallet_id == "wallet-personal"


class TestDebitFollowsCheck:
    def test_debit_targets_grant_wallet_id_ignoring_context_switch(self, monkeypatch):
        """Rule 9: a grant carrying a seat wallet_id debits THAT wallet directly,
        even if the caller's billing context switched after the check. The debit
        path re-resolves NOTHING — no profiles / org / personal-wallet reads."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = MagicMock()
        sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})
        grant = CreditGrant(
            request_id="req-x",
            action="zoe_message",
            price=3,
            kind="debit",
            enabled=True,
            wallet_id="wallet-seat",
        )

        EntitlementsService(sb).debit_for_action(CTX_USER, grant)

        debit_calls = [c for c in sb.rpc.call_args_list if c.args and c.args[0] == "debit_credits"]
        assert len(debit_calls) == 1
        assert debit_calls[0].args[1]["p_wallet_id"] == "wallet-seat"
        # No re-resolution: the wallet_id-bearing grant reads NO table at all.
        sb.table.assert_not_called()

    def test_legacy_grant_without_wallet_id_resolves_personal(self, monkeypatch):
        """A wallet_id=None (legacy) grant falls back to today's personal resolve."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        grant = CreditGrant(request_id="req-legacy", action="zoe_message", price=3, kind="debit", enabled=True)
        EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        debit_calls = [c for c in sb.rpc.call_args_list if c.args and c.args[0] == "debit_credits"]
        assert debit_calls[0].args[1]["p_wallet_id"] == "w-default"  # personal wallet id


class TestPersonalAllowedCarriesWalletId:
    """Regression for rule 9's personal side: allowed personal results now carry
    the personal wallet id so the same debit-follows-check invariant holds."""

    def test_sufficient_balance_carries_personal_wallet_id(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        r = EntitlementsService(_paid_supabase(bundle=100)).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.managed_by_org is False
        assert r.wallet_id == "w-default"

    def test_overage_allowed_carries_personal_wallet_id(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=0, overage_enabled=True)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "oneclick_run")
        assert r.allowed and r.use_overage and r.wallet_id == "w-default"


# ---------------------------------------------------------------------------
# Licensing Phase C — Task 6: resource-derived billing + owner-aware dry-seat
# wall in check_credits. Resolution order is derived-resource org → ambient →
# personal (rule 5: resource WINS over ambient context). Uses test_billing_
# context's FILTER-AWARE mock so the seat/user wallets, seats, and the
# deny-path project_members owner read all resolve correctly (and can be PROVEN
# absent on the allow path — rule: lazy, deny-only ownership check).
# ---------------------------------------------------------------------------

DERIV_PROJECT = "9b1d0000-0000-0000-0000-0000000000aa"
DERIV_PROJECT_2 = "9b1d0000-0000-0000-0000-0000000000bb"
DERIV_C1 = "f11e0000-0000-0000-0000-0000000000a1"
DERIV_C2 = "f11e0000-0000-0000-0000-0000000000b2"

ORG_A = "0rg0000a-0000-0000-0000-000000000001"
ORG_B = "0rg0000b-0000-0000-0000-000000000001"
MEMBER_A = "mem0000a-0000-0000-0000-000000000001"
MEMBER_B = "mem0000b-0000-0000-0000-000000000001"


def _link_row(project_id, org_id=ORG):
    return {"project_id": project_id, "org_id": org_id}


def _member_row(member_id, org_id, role="member", status="active"):
    return {"id": member_id, "org_id": org_id, "user_id": CTX_USER, "role": role, "status": status}


def _org_row(org_id, status="active", name="Org", archived_at=None):
    return {"id": org_id, "name": name, "status": status, "archived_at": archived_at}


def _seat_wallet_for(member_id, wallet_id, *, reserve=500, bundle=0):
    return {
        "id": wallet_id,
        "owner_type": "seat",
        "owner_id": member_id,
        "bundle_balance": bundle,
        "reserve_balance": reserve,
        "overage_this_period": 0,
        "period_start": None,
        "period_end": None,
    }


def _owner_pm_row(role="owner", project_id=DERIV_PROJECT):
    return {"project_id": project_id, "user_id": CTX_USER, "role": role}


def _derived_single_org_data(seat_wallets, *, project_members=None, org_status="active", member_status="active"):
    """A resource (DERIV_PROJECT) linked to ORG where CTX_USER holds a seat.
    Deliberately OMITS personal subscription/tier/user-wallet rows — the derived
    seat path must never read them."""
    return {
        "profiles": [_profile(context_org=None)],  # ambient personal — derivation must still win
        "org_project_links": [_link_row(DERIV_PROJECT, ORG)],
        "organizations": [_org(status=org_status)],
        "org_members": [_member(status=member_status)],
        "credit_prices": list(PRICES),
        "credit_wallets": list(seat_wallets),
        "project_members": project_members or [],
    }


def _personal_fallback_tables(*, bundle=100):
    """The personal subscription/tier/wallet rows a MISS falls through to."""
    return {
        "subscriptions": [_ctx_sub_row(tier="free")],
        "tier_entitlements": [FREE_TIER_ROW],
        "tier_overrides": [],
        "credit_wallets": [_user_wallet(bundle=bundle, period_end=FAR_FUTURE)],
    }


class TestCheckCreditsResourceDerivation:
    """Rule 5 matrix: resource wins over ambient context; a miss falls through."""

    def test_ambient_personal_resource_seat_pays_and_debit_targets_seat(self, monkeypatch):
        """Ambient = personal, resource linked to org where the caller has a
        FUNDED seat → the SEAT wallet pays, the personal wallet is never read,
        and the debit RPC targets the seat wallet id (rule 6)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_derived_single_org_data([_seat_wallet(reserve=500)]))
        svc = EntitlementsService(sb)

        r = svc.check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed and r.managed_by_org is True
        assert r.wallet_id == "wallet-seat" and r.price == 3
        # The personal (owner_type='user') wallet is NEVER read — seat wins.
        for preds in sb._log.get("credit_wallets", []):
            assert ("eq", "owner_type", "user") not in preds
        assert "subscriptions" not in sb._log

        grant = CreditGrant(
            request_id="req-c", action="zoe_message", price=r.price, kind="debit", enabled=True, wallet_id=r.wallet_id
        )
        svc.debit_for_action(CTX_USER, grant)
        debit_calls = [c for c in sb.rpc.call_args_list if c.args and c.args[0] == "debit_credits"]
        assert len(debit_calls) == 1
        assert debit_calls[0].args[1]["p_wallet_id"] == "wallet-seat"

    def test_ambient_org_a_resource_linked_org_b_pays_org_b(self, monkeypatch):
        """Ambient = org A (seat held), resource linked to org B (seat also
        held) → resource WINS: org B's seat wallet pays, not org A's (rule 5)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [{"id": CTX_USER, "billing_context_org_id": ORG_A, "is_admin": False}],  # ambient = A
            "org_project_links": [_link_row(DERIV_PROJECT, ORG_B)],  # resource → B
            "organizations": [_org_row(ORG_A), _org_row(ORG_B)],
            "org_members": [_member_row(MEMBER_A, ORG_A), _member_row(MEMBER_B, ORG_B)],
            "credit_prices": list(PRICES),
            "credit_wallets": [_seat_wallet_for(MEMBER_B, "wallet-seat-b", reserve=500)],
        }
        sb = _ctx_supabase(data)

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed and r.managed_by_org is True
        assert r.wallet_id == "wallet-seat-b"  # org B, not the ambient org A

    def test_linked_but_no_seat_falls_to_ambient_personal(self, monkeypatch):
        """Resource linked to an org where the caller holds NO seat → derivation
        misses → ambient personal pays; the 402 shape is unchanged and carries
        NO org data (rule 4)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [_profile(context_org=None)],
            "org_project_links": [_link_row(DERIV_PROJECT, ORG)],
            "organizations": [_org(status="active")],
            "org_members": [],  # no seat for the caller
            "credit_prices": list(PRICES),
            **_personal_fallback_tables(bundle=0),  # empty personal wallet → free-tier wall
        }
        sb = _ctx_supabase(data)

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed is False
        assert r.managed_by_org is False  # personal wall — no org data leaks
        assert r.upgrade_required is True
        assert r.owner_can_unlink is False and r.project_id is None
        assert r.wallet_id is None

    def test_mixed_project_contract_list_falls_to_ambient(self, monkeypatch):
        """Two contracts spread across two projects → no unanimity → ambient
        personal pays (rule 5: non-deterministic attribution forbidden)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [_profile(context_org=None)],
            "org_project_links": [_link_row(DERIV_PROJECT, ORG)],
            "organizations": [_org(status="active")],
            "org_members": [_member()],
            "project_files": [
                {"id": DERIV_C1, "project_id": DERIV_PROJECT},
                {"id": DERIV_C2, "project_id": DERIV_PROJECT_2},  # different project
            ],
            "credit_prices": list(PRICES),
            **_personal_fallback_tables(bundle=100),  # funded personal → allowed
        }
        sb = _ctx_supabase(data)

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message", resource_contract_ids=[DERIV_C1, DERIV_C2])

        assert r.allowed is True
        assert r.managed_by_org is False
        assert r.wallet_id == "wallet-personal"

    def test_resolver_exception_falls_to_ambient(self, monkeypatch):
        """If the resource resolver raises, check_credits swallows it and runs
        the ambient/personal path — the request is unaffected (rule 4)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")

        def _boom(*a, **k):
            raise RuntimeError("derivation exploded")

        monkeypatch.setattr(EntitlementsService, "resolve_billing_org_for_resource", _boom)
        data = {
            "profiles": [_profile(context_org=None)],
            "credit_prices": list(PRICES),
            **_personal_fallback_tables(bundle=100),
        }
        sb = _ctx_supabase(data)

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed is True and r.managed_by_org is False
        assert r.wallet_id == "wallet-personal"

    def test_no_resource_kwargs_is_byte_identical_personal(self, monkeypatch):
        """The default (no resource kwargs) call takes the personal path with no
        derivation queries at all — the byte-identical regression guarantee."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [_profile(context_org=None)],
            "credit_prices": list(PRICES),
            **_personal_fallback_tables(bundle=100),
        }
        sb = _ctx_supabase(data)

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is True and r.wallet_id == "wallet-personal"
        # No resource → the resolver short-circuits before any derivation read.
        assert "org_project_links" not in sb._log


class TestOwnerAwareDrySeatWall:
    """Rule 11: a dry seat on a linked project walls EVERYONE, but the OWNER
    additionally gets an unlink CTA — surfaced lazily, DENY-path only."""

    def test_derived_deny_owner_gets_unlink_and_project_id(self, monkeypatch):
        """Derived dry-seat deny + caller OWNS the linked project → owner_can_
        unlink, project_id, appended reason — and managed_by_org STILL present
        (the two CTAs co-occur, round 4)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _derived_single_org_data([_seat_wallet(reserve=0, bundle=0)], project_members=[_owner_pm_row("owner")])
        )

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed is False
        assert r.managed_by_org is True  # co-occurs — never mutually exclusive
        assert r.owner_can_unlink is True
        assert r.project_id == DERIV_PROJECT
        assert "unlink this project" in r.reason.lower()

    def test_derived_deny_non_owner_member_gets_no_owner_fields(self, monkeypatch):
        """Same dry-seat deny, but the caller is a non-owner MEMBER of the
        project → no owner fields, base reason unchanged."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _derived_single_org_data([_seat_wallet(reserve=0, bundle=0)], project_members=[_owner_pm_row("editor")])
        )

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed is False and r.managed_by_org is True
        assert r.owner_can_unlink is False and r.project_id is None
        assert r.reason == "You've used the credits your organization allocated. Ask your admin for more."

    def test_ambient_org_deny_never_gains_owner_fields(self, monkeypatch):
        """An AMBIENT org dry-seat deny (no resource → ctx has no project_id)
        never runs the owner check and never gains owner fields (AC #4)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [_profile(context_org=ORG)],  # ambient org context
            "org_members": [_member(status="active")],
            "organizations": [_org(status="active")],
            "credit_prices": list(PRICES),
            "credit_wallets": [_seat_wallet(reserve=0, bundle=0)],  # dry seat
        }
        sb = _ctx_supabase(data)

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")  # NO resource

        assert r.allowed is False and r.managed_by_org is True
        assert r.owner_can_unlink is False and r.project_id is None
        # The owner read never happened — ambient path carries no project_id.
        assert "project_members" not in sb._log

    def test_ownership_read_only_on_deny_not_allow(self, monkeypatch):
        """THE lazy-deny-only pin: on an ALLOW (funded seat), the project_members
        owner read is NEVER issued — ownership is checked only when a wall is
        already being built (Task 5 skips it on every derivation)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _derived_single_org_data([_seat_wallet(reserve=500)], project_members=[_owner_pm_row("owner")])
        )

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed is True and r.managed_by_org is True
        assert r.owner_can_unlink is False
        # No ownership read on the happy path.
        assert "project_members" not in sb._log
