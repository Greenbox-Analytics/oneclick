"""Wallet plumbing + code-level flag retirement in EntitlementsService."""

from unittest.mock import MagicMock

from orgs.storage_guard import TEAM_STORAGE_FULL_MSG
from subscriptions.ai_pricing import credits_for_cost
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
    MEMBER,
    ORG,
    PRICES,
    PRO_TIER_ROW,
    _ctx_store,
    _ctx_supabase,
    _member,
    _org,
    _pool_wallet,
    _profile,
    _usage_row,
    _user_wallet,
)
from tests.test_billing_context import _sub_row as _ctx_sub_row  # avoid shadowing this file's _sub_row(user_id)
from tests.test_billing_sweep import _FilterBuilder
from utils.llm.tracking import TrackedOpenAI, set_llm_context

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
                    }
                ],
                count=1,
            )
        return b

    sb.table.side_effect = side_effect
    sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False, "balance_after": 0})
    return sb


class TestGrandfatheredGrant:
    """Precedence: explicit admin override > grandfathered > tier value.
    The SAME resolution must hold in get_for_user, check_credits and the sweep
    — a mismatch silently grants two different bundles."""

    # Sentinel: "unset" means "far-future when gf is set, else irrelevant" —
    # every EXISTING call site that only passes gf=... keeps testing the
    # unexpired-grandfather path without touching each one individually. Pass
    # gf_until explicitly (an expired ISO string, or None) to test expiry.
    _UNSET = object()

    def _sb(self, *, gf=None, gf_until=_UNSET, override_credits=None, tier="basic", tier_credits=2000):
        sb = _paid_supabase(bundle=0)
        orig = sb.table.side_effect
        until = (FAR_FUTURE if gf is not None else None) if gf_until is self._UNSET else gf_until

        def side_effect(name):
            b = orig(name)
            if name == "subscriptions":
                b.execute.return_value = MagicMock(
                    data=[
                        {
                            "user_id": TEST_USER_ID,
                            "tier": tier,
                            "status": "active",
                            "grandfathered_monthly_credits": gf,
                            "grandfathered_until": until,
                        }
                    ],
                    count=1,
                )
            if name == "tier_entitlements":
                row = dict(_PRO_TIER_ROW, tier=tier, monthly_credits=tier_credits)
                b.execute.return_value = MagicMock(data=[row], count=1)
            if name == "tier_overrides" and override_credits is not None:
                b.execute.return_value = MagicMock(
                    data=[{"user_id": TEST_USER_ID, "monthly_credits": override_credits}], count=1
                )
            return b

        sb.table.side_effect = side_effect
        return sb

    def test_grandfather_beats_tier_value(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(self._sb(gf=3000)).get_for_user(TEST_USER_ID)
        assert ent.credits.monthly_grant == 3000

    def test_admin_override_beats_grandfather(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(self._sb(gf=3000, override_credits=9999)).get_for_user(TEST_USER_ID)
        assert ent.credits.monthly_grant == 9999

    def test_no_grandfather_uses_tier_value(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(self._sb(gf=None)).get_for_user(TEST_USER_ID)
        assert ent.credits.monthly_grant == 2000

    def test_expired_grandfather_uses_tier_value(self, monkeypatch):
        """Owner policy clarification (spec §1): grandfathering expires with
        the already-paid period — a past grandfathered_until reads as
        expired, silently, and falls through to the tier value."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(self._sb(gf=8000, gf_until="2020-01-01T00:00:00+00:00")).get_for_user(TEST_USER_ID)
        assert ent.credits.monthly_grant == 2000

    def test_grandfather_with_no_until_stamped_treated_as_expired(self, monkeypatch):
        """Defensive: a grant with no expiry stamped (should never happen —
        the migration backfill always stamps both together) is treated as
        already expired, never as indefinite."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(self._sb(gf=8000, gf_until=None)).get_for_user(TEST_USER_ID)
        assert ent.credits.monthly_grant == 2000

    def test_get_credit_usage_rolls_stale_wallet_at_grandfathered_grant(self, monkeypatch):
        """get_credit_usage is a FOURTH grant-computation site (spec review
        follow-up): a grandfathered user with a stale wallet hitting the usage
        view must roll at the resolved grant, not the tier default."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = self._sb(gf=8000)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "credit_wallets":
                stale = dict(_DEFAULT_WALLET_ROW, period_end="2020-01-01T00:00:00+00:00")
                b.execute.return_value = MagicMock(data=[stale], count=1)
            return b

        sb.table.side_effect = side_effect

        result = EntitlementsService(sb).get_credit_usage(TEST_USER_ID)

        args = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"][0]
        assert args["p_monthly_grant"] == 8000
        assert result["monthlyGrant"] == 8000

    def test_check_credits_rolls_stale_wallet_at_grandfathered_grant(self, monkeypatch):
        """check_credits is the SAME chokepoint (Task 1 AC) as get_for_user and
        get_credit_usage — clone of the get_credit_usage pin above."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = self._sb(gf=8000)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "credit_wallets":
                stale = dict(_DEFAULT_WALLET_ROW, period_end="2020-01-01T00:00:00+00:00")
                b.execute.return_value = MagicMock(data=[stale], count=1)
            return b

        sb.table.side_effect = side_effect

        EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")

        args = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "rollover_wallet"][0]
        assert args["p_monthly_grant"] == 8000


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

    def test_bypass_paywalls_env_short_circuits_before_wallet(self, monkeypatch):
        """BYPASS_PAYWALLS is the ONLY remaining short-circuit — the ops
        escape hatch. Admin status no longer confers one (owner decision,
        2026-08-15: admins are metered so the system is testable by its own
        operators; they self-grant when low)."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("BYPASS_PAYWALLS", "true")
        sb = _paid_supabase(bundle=0)
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
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

    def test_db_admin_is_metered_like_everyone(self, monkeypatch):
        """profiles.is_admin=True no longer bypasses the credit gate: the check
        runs against the wallet and returns the real price. Admins self-grant
        when they run low — their ledger rows are what make the credit system
        testable by its own operators."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        orig = sb.table.side_effect

        def side_effect(name):
            b = orig(name)
            if name == "profiles":
                b.execute.return_value = MagicMock(data=[{"is_admin": True}], count=1)
            return b

        sb.table.side_effect = side_effect
        r = EntitlementsService(sb).check_credits(TEST_USER_ID, "zoe_message")
        assert r.allowed and r.price == 3
        tables_touched = {c.args[0] for c in sb.table.call_args_list}
        assert "credit_wallets" in tables_touched


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


def _burn(*, model="gpt-5-mini", input_tokens=0, output_tokens=0):
    """Push a real LLM call through the tracking proxy so the current scope's
    spend accumulator moves exactly as it does in production."""
    inner = MagicMock()
    inner.chat.completions.create.return_value = MagicMock(
        usage=MagicMock(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            prompt_tokens_details=None,
        )
    )
    TrackedOpenAI(inner, get_supabase=lambda: MagicMock()).chat.completions.create(model=model)


class TestMeteredDebit:
    """The charge is max(BASE, metered): `grant.price` is the per-action base
    rate and a FLOOR, and the measured spend only decides the amount when a run
    cost more than the base already covers.
    """

    @staticmethod
    def _grant(price=3, kind="debit"):
        return CreditGrant(request_id="req-m", action="zoe_message", price=price, kind=kind, enabled=True)

    @staticmethod
    def _debit_args(sb):
        calls = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "debit_credits"]
        return calls[0] if calls else None

    def test_measured_spend_overrides_flat_price(self, monkeypatch):
        """200k tokens on a 3-credit base: metered 8, and the size tail adds 7
        on top of the base (170k tokens past zoe's 30k allowance) -> 10."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        with set_llm_context(TEST_USER_ID, "zoe"):
            _burn(input_tokens=200_000)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        args = self._debit_args(sb)
        assert args["p_amount"] == 10
        assert args["p_metadata"]["input_tokens"] == 200_000
        assert args["p_metadata"]["estimated"] == 3
        assert args["p_metadata"]["metered"] is True

    def test_cheap_call_still_pays_the_base(self, monkeypatch):
        """A small request meters to 1 credit, but the base rate is a FLOOR."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        with set_llm_context(TEST_USER_ID, "zoe"):
            _burn(input_tokens=2_000)  # $0.0005 -> 1 metered credit
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant(price=21))
        assert self._debit_args(sb)["p_amount"] == 21

    def test_no_llm_call_still_pays_the_base(self, monkeypatch):
        """A cache hit inside a tracked scope burns no tokens -> measures 0 and
        pays exactly the base, because the deliverable is the same one.

        This is the cache-hit rule at the service layer (spec 2026-08-17 §4).
        """
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        with set_llm_context(TEST_USER_ID, "registry"):
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        assert self._debit_args(sb)["p_amount"] == 3

    def test_unpriced_model_falls_back_to_the_estimate(self, monkeypatch):
        """A model missing from MODEL_RATES has unknowable cost — charge the
        flat estimate rather than silently under-charging to zero."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        with set_llm_context(TEST_USER_ID, "zoe"):
            _burn(model="some-unlisted-model", input_tokens=200_000)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        args = self._debit_args(sb)
        assert args["p_amount"] == 3 and args["p_metadata"]["metered"] is False

    def test_no_tracked_scope_falls_back_to_the_estimate(self, monkeypatch):
        """Background jobs and scripts have no scope — behave as before."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        assert self._debit_args(sb)["p_amount"] == 3

    def test_metered_amount_flows_into_overage(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            _burn(input_tokens=2_000)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant(price=21, kind="overage_debit"))
        args = self._debit_args(sb)
        # A normal-sized run: metered 1 and no size tail, so the 21 base is the
        # floor — and the floor applies to overage exactly as it does to bundle.
        assert args["p_kind"] == "overage_debit" and args["p_amount"] == 21


class TestCreditsForCost:
    def test_zero_cost_is_free(self):
        assert credits_for_cost(0) == 0 and credits_for_cost(-1) == 0

    def test_any_real_spend_costs_at_least_one_credit(self):
        assert credits_for_cost(0.000001) == 1

    def test_scales_linearly_with_cost(self):
        assert credits_for_cost(0.20) == 2 * credits_for_cost(0.10)

    def test_no_phantom_credit_from_float_dust(self):
        """0.10 * 3 / 0.02 is 15.000000000000002 in IEEE754; ceil must not
        round that to 16 (the bug the pricing dashboard's selfCheck caught)."""
        assert credits_for_cost(0.10) == 15

    def test_markup_is_tunable(self, monkeypatch):
        monkeypatch.setenv("CREDIT_MARKUP", "6.0")
        assert credits_for_cost(0.10) == 30


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

    def test_paid_under_included_allowed(self, monkeypatch):
        """Counterpart to the denial above: the allowance gate must not fire
        while usage is still under included_storage_bytes. (There is no storage
        pay-per-use — past the allowance the upload simply blocks.)"""
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
                            "total_storage_bytes": 107374182400 - 1024,
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


def _org_check_data(
    pool_wallets, *, org_status="active", member_status="active", monthly_cap=None, cap_used=0, default_member_cap=None
):
    """Data for an org-context check_credits: profiles preference + active
    membership + org + prices + the org's pool wallet. Deliberately OMITS the
    personal wallet / subscription / tier tables — the org path must never read
    them. `monthly_cap`/`cap_used` drive the member-cap pre-check."""
    from datetime import UTC, datetime, timedelta

    future = (datetime.now(UTC) + timedelta(days=10)).isoformat()
    return {
        "profiles": [_profile(context_org=ORG)],
        "org_members": [
            _member(
                status=member_status,
                monthly_cap=monthly_cap,
                cap_used=cap_used,
                cap_period_end=future if cap_used else None,
            )
        ],
        "organizations": [_org(status=org_status, default_member_cap=default_member_cap)],
        "credit_prices": list(PRICES),
        "credit_wallets": list(pool_wallets),
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
    def test_seat_pays_allowed_with_pool_wallet_id(self, monkeypatch):
        """Funded seat → allowed, price from the shared prices table, wallet_id is
        the SEAT wallet, managed_by_org True, and NONE of the personal-context
        fields (overage / upgrade / reset_date) are set (rule 8)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_pool_wallet(reserve=500)]))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed and r.managed_by_org is True
        assert r.wallet_id == "wallet-pool"
        assert r.price == 3
        assert r.use_overage is False and r.overage_available is False
        assert r.upgrade_required is False
        assert r.reset_date is None
        # Every pool select filtered owner_type='org'; the personal wallet
        # (owner_type='user') is NEVER read.
        wallet_queries = sb._log.get("credit_wallets", [])
        assert wallet_queries
        for preds in wallet_queries:
            assert ("eq", "owner_type", "org") in preds
            assert ("eq", "owner_type", "user") not in preds
        # The org seat path consulted NO personal subscription row.
        assert "subscriptions" not in sb._log

    def test_seat_pays_then_debit_targets_pool_wallet(self, monkeypatch):
        """The end-to-end money path: check → build grant from the result → debit.
        The debit RPC receives the SEAT wallet id, and the debit path adds no
        personal-wallet select (rule 9)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_pool_wallet(reserve=500)]))
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
        assert debit_calls[0].args[1]["p_wallet_id"] == "wallet-pool"
        # No personal-wallet ('user') select anywhere across check + debit.
        for preds in sb._log.get("credit_wallets", []):
            assert ("eq", "owner_type", "user") not in preds

    def test_seat_dry_402_shape(self, monkeypatch):
        """Empty seat → 402-shaped result: managed_by_org True, the ask-your-admin
        reason, and NO overage / upgrade / reset_date fields (rule 8)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_pool_wallet(reserve=0, bundle=0)]))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is False and r.managed_by_org is True
        assert r.reason == "Your organization is out of credits. Ask your admin to top up."
        assert r.overage_available is False and r.use_overage is False
        assert r.upgrade_required is False
        assert r.reset_date is None
        assert r.degraded is False  # a dry seat is a legitimate wall, not an outage

    def test_missing_pool_wallet_lazy_creates_zero_then_402(self, monkeypatch):
        """No seat wallet ROW at all → lazy-create at zero → 402 (rule 8's carve-out:
        a missing row is NOT the READ-ERROR fail-open case)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([]))  # zero wallets

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is False and r.managed_by_org is True
        assert r.degraded is False
        assert r.reason == "Your organization is out of credits. Ask your admin to top up."

    def test_pool_wallet_read_error_fails_open_uncharged(self, monkeypatch):
        """A seat-wallet READ EXCEPTION fails OPEN uncharged (price 0, degraded),
        like the paid personal tier (spec §12)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        table_fn, _log, _updates, _store = _ctx_store(_org_check_data([_pool_wallet(reserve=500)]))
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
        """A suspended org is PARKED (preference kept, resumes on reactivation)
        → personal check path meanwhile: personal wallet id, no managed_by_org."""
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
            wallet_id="wallet-pool",
        )

        EntitlementsService(sb).debit_for_action(CTX_USER, grant)

        debit_calls = [c for c in sb.rpc.call_args_list if c.args and c.args[0] == "debit_credits"]
        assert len(debit_calls) == 1
        assert debit_calls[0].args[1]["p_wallet_id"] == "wallet-pool"
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


def _artist_owned_tables(project_id, org_id=ORG, artist_id="art-owned"):
    """The artist edge that replaced org_project_links: project -> artist ->
    team. Two tables where the link table was one row."""
    return {
        "projects": [{"id": project_id, "artist_id": artist_id}],
        "artists": [{"id": artist_id, "team_id": org_id}],
    }


def _member_row(member_id, org_id, role="member", status="active"):
    return {"id": member_id, "org_id": org_id, "user_id": CTX_USER, "role": role, "status": status}


def _org_row(org_id, status="active", name="Org", archived_at=None):
    return {"id": org_id, "name": name, "status": status, "archived_at": archived_at}


def _pool_wallet_for(member_id, wallet_id, *, reserve=500, bundle=0):
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


def _derived_single_org_data(seat_wallets, *, org_status="active", member_status="active"):
    """A resource (DERIV_PROJECT) whose ARTIST is owned by ORG, where CTX_USER
    holds a seat. Deliberately OMITS personal subscription/tier/user-wallet rows
    — the derived seat path must never read them."""
    return {
        "profiles": [_profile(context_org=None)],  # ambient personal — derivation must still win
        "projects": [{"id": DERIV_PROJECT, "artist_id": "art-deriv"}],
        "artists": [{"id": "art-deriv", "team_id": ORG}],
        "organizations": [_org(status=org_status)],
        "org_members": [_member(status=member_status)],
        "credit_prices": list(PRICES),
        "credit_wallets": list(seat_wallets),
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
        sb = _ctx_supabase(_derived_single_org_data([_pool_wallet(reserve=500)]))
        svc = EntitlementsService(sb)

        r = svc.check_credits(CTX_USER, "zoe_message", resource_project_id=DERIV_PROJECT)

        assert r.allowed and r.managed_by_org is True
        assert r.wallet_id == "wallet-pool" and r.price == 3
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
        assert debit_calls[0].args[1]["p_wallet_id"] == "wallet-pool"

    def test_ambient_org_a_resource_linked_org_b_pays_org_b(self, monkeypatch):
        """Ambient = org A (seat held), resource linked to org B (seat also
        held) → resource WINS: org B's seat wallet pays, not org A's (rule 5)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [{"id": CTX_USER, "billing_context_org_id": ORG_A, "is_admin": False}],  # ambient = A
            **_artist_owned_tables(DERIV_PROJECT, ORG_B),  # resource's artist → B
            "organizations": [_org_row(ORG_A), _org_row(ORG_B)],
            "org_members": [_member_row(MEMBER_A, ORG_A), _member_row(MEMBER_B, ORG_B)],
            "credit_prices": list(PRICES),
            "credit_wallets": [_pool_wallet_for(MEMBER_B, "wallet-seat-b", reserve=500)],
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
            **_artist_owned_tables(DERIV_PROJECT, ORG),
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
        assert r.wallet_id is None

    def test_mixed_project_contract_list_falls_to_ambient(self, monkeypatch):
        """Two contracts spread across two projects → no unanimity → ambient
        personal pays (rule 5: non-deterministic attribution forbidden)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        data = {
            "profiles": [_profile(context_org=None)],
            **_artist_owned_tables(DERIV_PROJECT, ORG),
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


class TestFreeTierToolsOpenUnderCredits:
    """Credits ARE the gate: under CREDITS_ENABLED the metered tools are open on
    every tier and the legacy per-period OneClick counter goes unlimited, so a
    Free user meets a credit wall (having used the tool) rather than a feature
    wall (never having seen it). Flag OFF must restore the stored values exactly
    — that's the spec §9 rollback property, and it's what these two pin."""

    FREE_ROW = {
        "tier": "free",
        "max_artists": 3,
        "max_projects": 3,
        "max_tasks": 50,
        "max_storage_bytes": 1073741824,
        "max_split_sheets_per_month": 5,
        "max_oneclick_runs_per_month": 1,
        "zoe_enabled": False,
        "oneclick_enabled": False,
        "registry_enabled": False,
        "integrations_allowed": ["google_drive"],
        "monthly_credits": 150,
        "max_works": 10,
        "included_storage_bytes": 1073741824,
    }

    def _free_supabase(self):
        sb = MagicMock()

        def side_effect(name):
            b = _default_table_side_effect(name)
            if name == "subscriptions":
                b.execute.return_value = MagicMock(
                    data=[{"user_id": TEST_USER_ID, "tier": "free", "status": "active"}], count=1
                )
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[self.FREE_ROW], count=1)
            return b

        sb.table.side_effect = side_effect
        sb.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False, "balance_after": 0})
        return sb

    def test_credits_on_opens_the_metered_tools(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        ent = EntitlementsService(self._free_supabase()).get_for_user(TEST_USER_ID)
        assert (ent.features.zoe_enabled, ent.features.oneclick_enabled, ent.features.registry_enabled) == (
            True,
            True,
            True,
        )
        assert ent.caps.max_oneclick_runs_per_month == -1
        # Split sheets are NOT credit-priced — that cap must survive untouched.
        assert ent.caps.max_split_sheets_per_month == 5

    def test_credits_off_keeps_stored_flags(self, monkeypatch):
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        ent = EntitlementsService(self._free_supabase()).get_for_user(TEST_USER_ID)
        assert (ent.features.zoe_enabled, ent.features.oneclick_enabled, ent.features.registry_enabled) == (
            False,
            False,
            False,
        )
        assert ent.caps.max_oneclick_runs_per_month == 1


class TestMemberCapGate:
    """The org path has TWO ceilings with DIFFERENT remedies: the member's own
    monthly cap (ask the admin to raise it) and the pool balance (only the admin
    buying credits fixes that). `cap_reached` is what lets the 402 say which."""

    def test_under_cap_and_funded_pool_allows(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_pool_wallet(reserve=500)], monthly_cap=100, cap_used=10))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")  # price 3

        assert r.allowed and r.managed_by_org is True
        assert r.cap_reached is False
        assert r.wallet_id == "wallet-pool"
        # The member id rides along so the debit can move the cap counter under
        # the same lock the debit takes.
        assert r.org_member_id == MEMBER

    def test_cap_reached_denies_even_with_a_full_pool(self, monkeypatch):
        """The pool has plenty; the member is at their ceiling. The wall must say
        so — pointing them at "the org is out of credits" would send them to the
        wrong remedy."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_pool_wallet(reserve=100_000)], monthly_cap=100, cap_used=98))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")  # price 3 > 100-98

        assert r.allowed is False
        assert r.cap_reached is True
        assert "monthly credit limit" in r.reason
        assert r.use_overage is False and r.upgrade_required is False

    def test_dry_pool_denies_without_claiming_the_cap(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_pool_wallet(reserve=0)], monthly_cap=1000, cap_used=0))

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is False
        assert r.cap_reached is False
        assert "out of credits" in r.reason

    def test_null_member_cap_falls_through_to_the_org_default(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _org_check_data([_pool_wallet(reserve=500)], monthly_cap=None, cap_used=2, default_member_cap=3)
        )

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")  # price 3, used 2 of 3

        assert r.allowed is False and r.cap_reached is True

    def test_uncapped_member_is_bounded_only_by_the_pool(self, monkeypatch):
        """No member cap and no org default = uncapped. The pool is the limit,
        which is the whole point of allowing caps to overcommit."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _org_check_data([_pool_wallet(reserve=500)], monthly_cap=None, cap_used=0, default_member_cap=None)
        )

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is True and r.cap_reached is False

    def test_sentinel_cap_uncaps_a_member_under_a_capped_org(self, monkeypatch):
        """-1 on the member row beats a real org default — the ONLY way to say
        "no limit" now that every org carries one. Without the normalization
        this would compare against a ceiling of -1 and wall immediately."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _org_check_data([_pool_wallet(reserve=500)], monthly_cap=-1, cap_used=9999, default_member_cap=3)
        )

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is True and r.cap_reached is False

    def test_sentinel_org_default_uncaps_inheriting_members(self, monkeypatch):
        """An admin can lift the ceiling org-wide the same way."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _org_check_data([_pool_wallet(reserve=500)], monthly_cap=None, cap_used=9999, default_member_cap=-1)
        )

        r = EntitlementsService(sb).check_credits(CTX_USER, "zoe_message")

        assert r.allowed is True and r.cap_reached is False

    def test_uncapped_member_reports_no_cap_to_the_ui(self, monkeypatch):
        """A -1 member must surface memberCap=None so the UI says "pulling from
        org credits pool" rather than rendering a nonsense -1 limit."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(
            _org_check_data([_pool_wallet(reserve=500)], monthly_cap=-1, cap_used=0, default_member_cap=2000)
        )

        cap, _used = EntitlementsService(sb)._member_cap(ORG, MEMBER)

        assert cap is None

    def test_debit_threads_the_member_id_into_the_rpc(self, monkeypatch):
        """End-to-end: the grant carries org_member_id, so the RPC gets
        p_member_id and moves the cap counter transactionally."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _ctx_supabase(_org_check_data([_pool_wallet(reserve=500)], monthly_cap=1000))
        svc = EntitlementsService(sb)

        r = svc.check_credits(CTX_USER, "zoe_message")
        grant = CreditGrant(
            request_id="req-cap",
            action="zoe_message",
            price=r.price,
            kind="debit",
            enabled=True,
            wallet_id=r.wallet_id,
            org_member_id=r.org_member_id,
        )
        svc.debit_for_action(CTX_USER, grant)

        debit = [c for c in sb.rpc.call_args_list if c.args and c.args[0] == "debit_credits"][0]
        assert debit.args[1]["p_wallet_id"] == "wallet-pool"
        assert debit.args[1]["p_member_id"] == MEMBER

    def test_personal_debit_never_sends_a_member_id(self, monkeypatch):
        """p_member_id must be ABSENT for personal spend, so the RPC's
        org_members lock never fires for a personal wallet."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100)
        svc = EntitlementsService(sb)

        grant = CreditGrant(
            request_id="req-personal", action="zoe_message", price=3, kind="debit", enabled=True, wallet_id="w-personal"
        )
        svc.debit_for_action(TEST_USER_ID, grant)

        debit = [c for c in sb.rpc.call_args_list if c.args and c.args[0] == "debit_credits"][0]
        assert "p_member_id" not in debit.args[1]


# ===========================================================================
# ARTIST OWNERSHIP — the payer edge (Team-Owned Artists, Task 3)
# ===========================================================================


class TestArtistOwnershipDerivation:
    """Artist ownership is THE payer edge — the only one, since
    org_project_links was retired in 20260804000001."""

    def _sb(self, *, artist_team):
        data = {
            "profiles": [_profile(context_org=None)],
            "org_members": [_member(status="active")],
            "organizations": [_org(status="active")],
            "credit_prices": list(PRICES),
            "credit_wallets": [_pool_wallet(reserve=500)],
            "projects": [{"id": DERIV_PROJECT, "artist_id": "art-1"}],
            "artists": [{"id": "art-1", "team_id": artist_team}],
        }
        return _ctx_supabase(data)

    def test_team_owned_artist_resolves_to_its_org(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        svc = EntitlementsService(self._sb(artist_team=ORG))

        ctx = svc.resolve_billing_org_for_project(CTX_USER, DERIV_PROJECT)

        assert ctx is not None and ctx["org_id"] == ORG
        assert ctx["project_id"] == DERIV_PROJECT

    def test_personal_artist_resolves_to_none(self, monkeypatch):
        """No artist team -> personal billing. There is no second source of
        truth to fall through to."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        svc = EntitlementsService(self._sb(artist_team=None))

        assert svc.resolve_billing_org_for_project(CTX_USER, DERIV_PROJECT) is None

    def test_no_seat_in_the_artists_team_returns_none(self, monkeypatch):
        """Derivation only ever UPGRADES (rule 4): a team the caller is not in
        must never become their payer."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        foreign = "0rg00000-0000-0000-0000-0000000000ee"
        sb = self._sb(artist_team=foreign)

        assert EntitlementsService(sb).resolve_billing_org_for_project(CTX_USER, DERIV_PROJECT) is None

    def test_licensing_off_short_circuits(self, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        sb = self._sb(artist_team=ORG)

        assert EntitlementsService(sb).resolve_billing_org_for_project(CTX_USER, DERIV_PROJECT) is None
        assert "artists" not in sb._log  # no query at all


class TestTeamStorageAttribution:
    """An upload into a team-owned artist's project is measured against the
    TEAM's total and the TEAM's cap (per-seat x active seats), not against the
    uploader's personal counter and not against a single seat."""

    def _sb(self, *, artist_team, team_bytes=0, personal_bytes=0, seats=1):
        data = {
            "profiles": [_profile(context_org=None)],
            "org_members": [_member(status="active") for _ in range(seats)],
            "organizations": [{**_org(status="active"), "storage_bytes": team_bytes}],
            "credit_prices": list(PRICES),
            "credit_wallets": [_pool_wallet(reserve=500)],
            "projects": [{"id": DERIV_PROJECT, "artist_id": "art-1"}],
            "artists": [{"id": "art-1", "team_id": artist_team}],
            "usage_counters": [_usage_row(storage=personal_bytes)],
            "subscriptions": [_ctx_sub_row(tier="pro")],
            "tier_entitlements": [PRO_TIER_ROW],
            "tier_overrides": [],
        }
        return _ctx_supabase(data)

    def _can_upload(self, sb, size=50):
        from subscriptions.models import Action

        return EntitlementsService(sb).can(CTX_USER, Action.UPLOAD_BYTES, size=size, resource_project_id=DERIV_PROJECT)

    def test_team_artist_upload_measures_the_team_total(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", "1000")
        sb = self._sb(artist_team=ORG, team_bytes=990, personal_bytes=0, seats=1)

        r = self._can_upload(sb)

        assert not r.allowed, "990 + 50 exceeds the 1-seat 1000-byte team cap"
        assert "team" in r.reason.lower()

    def test_team_cap_scales_with_active_seats(self, monkeypatch):
        """The seat number is PER PERSON. A ten-person team that paid for ten
        seats does not share one seat's worth of space."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", "1000")
        sb = self._sb(artist_team=ORG, team_bytes=990, seats=10)

        assert self._can_upload(sb).allowed, "990 + 50 is well inside 10 x 1000"

    def test_personal_artist_upload_is_unchanged(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", "1000")
        sb = self._sb(artist_team=None, team_bytes=999_999, personal_bytes=0)

        assert self._can_upload(sb).allowed, "a personal artist must not see the team's usage"


class TestSelfServeOrgUploadGate:
    """Task 12: the can() UPLOAD_BYTES org branch, self_serve side.
    `_team_storage`'s org-row read (in TestTeamStorageAttribution above) now
    also carries `kind`/`covered_by` — a self_serve org routes through the
    shared per-owner pool (orgs/storage_guard.pool_state's math) instead of
    the enterprise per-seat calc, with its own member-facing copy. The
    enterprise branch itself is asserted BYTE-IDENTICAL above; this class only
    adds the new self_serve fork plus the two edges the task calls out
    explicitly: an enterprise org must not pick up an extra tier_entitlements
    read, and a lapsed org must never reach this branch at all."""

    GB = 2**30
    BASIC_TEAM_TIER_ROW = dict(FREE_TIER_ROW, tier="basic", team_storage_bytes=10 * GB)

    def _sb(self, *, org_status="active", org_kind="self_serve", team_bytes=0):
        org = {**_org(status=org_status), "storage_bytes": team_bytes}
        if org_kind is not None:
            org["kind"] = org_kind
            org["covered_by"] = CTX_USER
        data = {
            "profiles": [_profile(context_org=None)],
            "org_members": [_member(status="active")],
            "organizations": [org],
            "projects": [{"id": DERIV_PROJECT, "artist_id": "art-1"}],
            "artists": [{"id": "art-1", "team_id": ORG}],
            "usage_counters": [_usage_row(storage=0)],
            "subscriptions": [_ctx_sub_row(tier="basic" if org_kind == "self_serve" else "pro")],
            "tier_entitlements": [self.BASIC_TEAM_TIER_ROW if org_kind == "self_serve" else PRO_TIER_ROW],
            "tier_overrides": [],
            "credit_prices": list(PRICES),
            "credit_wallets": [],
        }
        return _ctx_supabase(data)

    def _can_upload(self, sb, size):
        from subscriptions.models import Action

        return EntitlementsService(sb).can(CTX_USER, Action.UPLOAD_BYTES, size=size, resource_project_id=DERIV_PROJECT)

    def test_self_serve_pool_denies_with_member_copy(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = self._sb(team_bytes=int(9.5 * self.GB))

        r = self._can_upload(sb, size=int(0.6 * self.GB))

        assert r.allowed is False
        assert r.reason == TEAM_STORAGE_FULL_MSG

    def test_self_serve_pool_allows_within_headroom(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = self._sb(team_bytes=int(9.5 * self.GB))

        r = self._can_upload(sb, size=int(0.4 * self.GB))

        assert r.allowed is True

    def test_landing_exactly_on_the_pool_boundary_is_allowed(self, monkeypatch):
        """used + size == pool is a fit, not an overage (inclusive `<=`)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = self._sb(team_bytes=9 * self.GB)  # basic pool is 10 GiB

        r = self._can_upload(sb, size=1 * self.GB)  # 9 + 1 == 10

        assert r.allowed is True

    def test_enterprise_upload_does_not_add_a_tier_entitlements_read(self, monkeypatch):
        """Enterprise stays on the ORIGINAL per-seat calc: pool_state's
        team_dials_for_user is a SECOND profiles/subscriptions/tier_entitlements
        read on top of get_for_user's own, and it must never fire here."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(10 * self.GB))
        sb = self._sb(org_kind=None, team_bytes=0)  # no `kind` on the row -> enterprise fork

        r = self._can_upload(sb, size=1)

        assert r.allowed is True
        assert len(sb._log.get("tier_entitlements", [])) == 1, "only get_for_user's own personal-tier read"

    def test_lapsed_org_never_reaches_the_pool_branch(self, monkeypatch):
        """The REAL gate for a lapsed org is upstream, at the artist-visibility
        layer (artist_access.live_org_ids excludes status='lapsed' — see
        tests/test_artist_access.py, not duplicated here). This pins the
        can()-level consequence: resolve_billing_org_for_project requires
        status='active' (_org_context_for), so a lapsed org's project never
        derives — the pool/_team_storage branch is unreached and the request
        falls through to the caller's own personal cap, never a wall
        pretending the (here: 999 GB) org total is theirs (review r2)."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        sb = self._sb(org_status="lapsed", team_bytes=999 * self.GB)

        assert EntitlementsService(sb).resolve_billing_org_for_project(CTX_USER, DERIV_PROJECT) is None

        r = self._can_upload(sb, size=1)

        assert r.allowed is True, "must fall through to the tiny personal cap, not the lapsed org's 999GB total"


class TestBaseRateCharge:
    """charge = max(base, metered, base + size tail).

    Base rates: spec 2026-08-17 §2. Size tail added 2026-08-27 — a run gets a
    per-action free token allowance (~10 pages for document tools) and pays for
    what it burns past it, so a 60-page contract no longer costs what a 3-page
    one does.
    """

    @staticmethod
    def _grant(price=30, kind="debit"):
        return CreditGrant(request_id="req-b", action="oneclick_run", price=price, kind=kind, enabled=True)

    @staticmethod
    def _debit_args(sb):
        calls = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "debit_credits"]
        return calls[0] if calls else None

    def test_a_normal_sized_run_charges_exactly_the_base(self, monkeypatch):
        """4,800 tokens is the MEASURED median OneClick run (ai_usage_log,
        2026-08-27). It fits inside the 6,500-token allowance and meters to 1,
        so the advertised 30 is the real price — which is the whole point of
        publishing a base rate."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=1000)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            _burn(input_tokens=4_800)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        args = self._debit_args(sb)
        assert args["p_amount"] == 30
        assert args["p_metadata"]["base"] == 30
        assert args["p_metadata"]["measurable"] is True
        assert args["p_metadata"]["tail_credits"] == 0
        assert args["p_metadata"]["free_tokens"] == 6_500
        assert args["p_metadata"]["metered"] is False  # nothing beat the base

    def test_an_oversized_run_pays_base_plus_the_excess(self, monkeypatch):
        """1M tokens is ~1,500 pages. Metered alone would say 38; base + the
        tail on the 993.5k tokens past the allowance says 68, and the max()
        takes it. Under the OLD two-term rule this was 38."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=1000)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            _burn(input_tokens=1_000_000)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        args = self._debit_args(sb)
        assert args["p_amount"] == 68
        assert args["p_metadata"]["metered_credits"] == 38
        assert args["p_metadata"]["tail_credits"] == 38
        assert args["p_metadata"]["metered"] is True

    def test_cache_hit_measures_zero_and_still_charges_base(self, monkeypatch):
        """THE behaviour change: a tracked scope that burns no tokens still pays
        the base, because the user received the same deliverable."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=1000)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        assert self._debit_args(sb)["p_amount"] == 30

    def test_unmeasurable_charges_base(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=1000)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            _burn(model="some-unlisted-model", input_tokens=1_000_000)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant())
        args = self._debit_args(sb)
        assert args["p_amount"] == 30
        assert args["p_metadata"]["measurable"] is False
        assert args["p_metadata"]["metered_credits"] is None
        assert args["p_metadata"]["metered"] is False

    def test_zero_price_grant_still_charges_nothing(self, monkeypatch):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=1000)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            _burn(input_tokens=1_000_000)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, self._grant(price=0))
        assert self._debit_args(sb) is None


class TestSizeTail:
    """The size curve the /docs page promises, pinned end to end.

    These are the numbers a user can check against their own invoice, so they
    are asserted as whole charges rather than as internals. Pages are converted
    at ~650 tokens/page, the figure the pricing model uses.
    """

    PAGE = 650

    @staticmethod
    def _debit_args(sb):
        calls = [c.args[1] for c in sb.rpc.call_args_list if c.args[0] == "debit_credits"]
        return calls[0] if calls else None

    def _charge(self, monkeypatch, *, pages, action="oneclick_run", price=30, tool="oneclick"):
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=100_000)
        grant = CreditGrant(request_id=f"req-{pages}", action=action, price=price, kind="debit", enabled=True)
        with set_llm_context(TEST_USER_ID, tool):
            _burn(input_tokens=pages * self.PAGE)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        return self._debit_args(sb)["p_amount"]

    def test_ten_pages_is_the_advertised_price(self, monkeypatch):
        """The allowance is set AT the promise: 10 pages costs exactly 30."""
        assert self._charge(monkeypatch, pages=10) == 30

    def test_under_the_allowance_is_the_advertised_price(self, monkeypatch):
        assert self._charge(monkeypatch, pages=1) == 30
        assert self._charge(monkeypatch, pages=9) == 30

    def test_the_curve_is_monotonic_in_size(self, monkeypatch):
        """The defect this whole change fixes: a bigger run must never cost the
        same as a smaller one once past the allowance."""
        charges = [self._charge(monkeypatch, pages=p) for p in (10, 30, 60, 120)]
        assert charges == sorted(charges)
        assert len(set(charges)) == len(charges), f"sizes collapsed to one price: {charges}"

    def test_a_cache_hit_never_tails(self, monkeypatch):
        """Zero tokens burned means zero excess, however big the document was.
        Consistent with charge-for-the-deliverable — and the reason the tail is
        keyed on tokens BURNED rather than document size."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=1000)
        grant = CreditGrant(request_id="req-c", action="oneclick_run", price=30, kind="debit", enabled=True)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        assert self._debit_args(sb)["p_amount"] == 30

    def test_zoe_keeps_its_own_larger_allowance(self, monkeypatch):
        """Zoe burns ~6x a document run because retrieval reads the corpus. One
        global allowance would have tripled the price of a median question, so
        the allowance is per-action. 27k tokens is Zoe's MEASURED median."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        sb = _paid_supabase(bundle=1000)
        grant = CreditGrant(request_id="req-z", action="zoe_message", price=5, kind="debit", enabled=True)
        with set_llm_context(TEST_USER_ID, "zoe"):
            _burn(input_tokens=27_000)
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        assert self._debit_args(sb)["p_amount"] == 5

    def test_the_charge_never_dips_below_real_cost(self, monkeypatch):
        """The safety property that makes the allowance safe to mis-tune: set it
        absurdly high and the charge falls back to measured cost, never below
        it. Mis-tuning this dial can cost margin; it can never cost money."""
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        monkeypatch.setenv("CREDIT_TAIL_FREE_TOKENS", "99999999")
        sb = _paid_supabase(bundle=100_000)
        grant = CreditGrant(request_id="req-s", action="oneclick_run", price=30, kind="debit", enabled=True)
        with set_llm_context(TEST_USER_ID, "oneclick"):
            _burn(input_tokens=1_000_000)  # $0.25 -> 38 metered credits
            EntitlementsService(sb).debit_for_action(TEST_USER_ID, grant)
        assert self._debit_args(sb)["p_amount"] == 38
