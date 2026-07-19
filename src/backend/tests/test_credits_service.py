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
