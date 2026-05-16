"""Unit tests for EntitlementsService. Mock-based, matches existing test pattern."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from tests.conftest import TEST_USER_ID, MockQueryBuilder

# ---------------------------------------------------------------------------
# Sample DB rows
# ---------------------------------------------------------------------------

FREE_TIER_ROW = {
    "tier": "free",
    "max_artists": 3,
    "max_projects": 3,
    "max_boards": 3,
    "max_tasks": 50,
    "max_storage_bytes": 1073741824,
    "max_split_sheets_per_month": 5,
    "max_oneclick_runs_per_month": 1,
    "zoe_enabled": False,
    "oneclick_enabled": True,
    "registry_enabled": False,
    "integrations_allowed": ["google_drive"],
    "updated_at": "2026-05-09T00:00:00+00:00",
}
PRO_TIER_ROW = {
    "tier": "pro",
    "max_artists": -1,
    "max_projects": -1,
    "max_boards": -1,
    "max_tasks": -1,
    "max_storage_bytes": -1,
    "max_split_sheets_per_month": -1,
    "max_oneclick_runs_per_month": -1,
    "zoe_enabled": True,
    "oneclick_enabled": True,
    "registry_enabled": True,
    "integrations_allowed": ["google_drive", "slack", "notion", "monday"],
    "updated_at": "2026-05-09T00:00:00+00:00",
}
FREE_SUB_ROW = {
    "id": "00000000-0000-0000-0000-000000000aaa",
    "user_id": TEST_USER_ID,
    "tier": "free",
    "status": "active",
    "stripe_customer_id": None,
    "stripe_subscription_id": None,
    "stripe_price_id": None,
    "current_period_start": None,
    "current_period_end": None,
    "cancel_at_period_end": False,
    "canceled_at": None,
    "created_at": "2026-05-09T00:00:00+00:00",
    "updated_at": "2026-05-09T00:00:00+00:00",
}
PRO_SUB_ROW = {**FREE_SUB_ROW, "tier": "pro"}
ZERO_USAGE_ROW = {
    "user_id": TEST_USER_ID,
    "total_storage_bytes": 0,
    "split_sheets_this_period": 0,
    "zoe_queries_this_period": 0,
    "oneclick_runs_this_period": 0,
    "period_start": "2026-05-09T00:00:00+00:00",
    "period_end": "2099-05-09T00:00:00+00:00",  # far future so no rollover
    "updated_at": "2026-05-09T00:00:00+00:00",
}

# Pro host fixtures (used by host-wins tests in Task 5 — declared here so future
# appends to this file can reuse without re-declaring).
HOST_USER_ID = "00000000-0000-0000-0000-000000000099"
PRO_HOST_SUB_ROW = {**PRO_SUB_ROW, "user_id": HOST_USER_ID}
HOST_USAGE_ROW = {**ZERO_USAGE_ROW, "user_id": HOST_USER_ID}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_supabase_with_rows(rows_by_table: dict[str, list[dict]]):
    """Mock Supabase client whose .table(name).execute() returns rows_by_table[name]."""
    mock = MagicMock()

    def _table(name):
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=rows_by_table.get(name, []), count=len(rows_by_table.get(name, [])))
        return b

    mock.table.side_effect = _table
    return mock


def _make_supabase_per_user_rows(per_user: dict[str, dict[str, list[dict]]]):
    """Mock that picks rows based on the user_id arg passed to .eq('user_id', X).

    `per_user` is {user_id: {table_name: rows}}. The SAME tier_entitlements set is
    used for everyone (we look it up by tier, not user). subscription/usage_counters/
    tier_overrides are looked up by user_id.
    """
    mock = MagicMock()
    state = {"current_user_id": None, "tier_for_query": None}

    def _table(name):
        b = MockQueryBuilder()
        original_eq = b.eq

        def _capture_eq(field, value):
            if field == "user_id":
                state["current_user_id"] = value
            elif field == "tier":
                state["tier_for_query"] = value
            return original_eq(field, value)

        b.eq = _capture_eq

        def _execute_capture(*a, **kw):
            uid = state["current_user_id"]
            if name == "tier_entitlements":
                tier = state["tier_for_query"]
                rows = [r for r in per_user.get(uid, {}).get(name, []) if r["tier"] == tier]
                if not rows:
                    for u in per_user.values():
                        for r in u.get(name, []):
                            if r["tier"] == tier:
                                return MagicMock(data=[r], count=1)
                return MagicMock(data=rows, count=len(rows))
            rows = per_user.get(uid, {}).get(name, [])
            return MagicMock(data=rows, count=len(rows))

        b.execute = _execute_capture
        return b

    mock.table.side_effect = _table
    return mock


# ---------------------------------------------------------------------------
# Task 3 tests — basic reads + override merge
# ---------------------------------------------------------------------------


class TestFreeUserDefaults:
    def test_free_user_default_entitlements(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)

        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.tier == "free"
        assert ent.status == "active"
        assert ent.caps.max_artists == 3
        assert ent.features.zoe_enabled is False
        assert ent.features.integrations_allowed == ["google_drive"]
        assert ent.has_overrides is False


class TestProUserDefaults:
    def test_pro_user_default_entitlements(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [PRO_SUB_ROW],
                "tier_entitlements": [PRO_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)

        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.tier == "pro"
        assert ent.caps.max_artists == -1
        assert ent.features.zoe_enabled is True
        assert "slack" in ent.features.integrations_allowed
        assert ent.has_overrides is False


class TestOverrideMerging:
    def test_override_per_field_merge(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [
                    {
                        "user_id": TEST_USER_ID,
                        "max_artists": None,
                        "max_projects": None,
                        "max_boards": None,
                        "max_tasks": None,
                        "max_storage_bytes": None,
                        "max_split_sheets_per_month": None,
                        "zoe_enabled": True,
                        "oneclick_enabled": None,
                        "registry_enabled": None,
                        "integrations_allowed": None,
                        "reason": "Beta tester",
                        "granted_at": "2026-05-09T00:00:00+00:00",
                        "expires_at": None,
                    }
                ],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)

        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.features.zoe_enabled is True
        assert ent.caps.max_artists == 3
        assert ent.has_overrides is True

    def test_override_replaces_integrations_array(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [
                    {
                        "user_id": TEST_USER_ID,
                        "max_artists": None,
                        "max_projects": None,
                        "max_boards": None,
                        "max_tasks": None,
                        "max_storage_bytes": None,
                        "max_split_sheets_per_month": None,
                        "zoe_enabled": None,
                        "oneclick_enabled": None,
                        "registry_enabled": None,
                        "integrations_allowed": ["slack"],
                        "reason": None,
                        "granted_at": "2026-05-09T00:00:00+00:00",
                        "expires_at": None,
                    }
                ],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)

        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.features.integrations_allowed == ["slack"]
        assert "google_drive" not in ent.features.integrations_allowed

    def test_expired_override_ignored(self):
        from subscriptions.service import EntitlementsService

        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [
                    {
                        "user_id": TEST_USER_ID,
                        "max_artists": None,
                        "max_projects": None,
                        "max_boards": None,
                        "max_tasks": None,
                        "max_storage_bytes": None,
                        "max_split_sheets_per_month": None,
                        "zoe_enabled": True,
                        "oneclick_enabled": None,
                        "registry_enabled": None,
                        "integrations_allowed": None,
                        "reason": "expired comp",
                        "granted_at": "2026-01-01T00:00:00+00:00",
                        "expires_at": past,
                    }
                ],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)

        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.features.zoe_enabled is False
        assert ent.has_overrides is False


class TestMissingRows:
    def test_missing_subscription_auto_creates_free(self):
        from subscriptions.service import EntitlementsService

        call_state = {"subscriptions_reads": 0}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                if call_state["subscriptions_reads"] == 0:
                    b.execute.return_value = MagicMock(data=[], count=0)
                    call_state["subscriptions_reads"] += 1
                else:
                    b.execute.return_value = MagicMock(data=[FREE_SUB_ROW], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "usage_counters":
                b.execute.return_value = MagicMock(data=[ZERO_USAGE_ROW], count=1)
            return b

        sb.table.side_effect = _table
        svc = EntitlementsService(sb)

        ent = svc.get_for_user(TEST_USER_ID)
        assert ent.tier == "free"
        assert ent.status == "active"

    def test_missing_usage_counter_auto_creates(self):
        from subscriptions.service import EntitlementsService

        call_state = {"usage_reads": 0}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[FREE_SUB_ROW], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "usage_counters":
                if call_state["usage_reads"] == 0:
                    b.execute.return_value = MagicMock(data=[], count=0)
                    call_state["usage_reads"] += 1
                else:
                    b.execute.return_value = MagicMock(data=[ZERO_USAGE_ROW], count=1)
            return b

        sb.table.side_effect = _table
        svc = EntitlementsService(sb)

        ent = svc.get_for_user(TEST_USER_ID)
        assert ent.usage.total_storage_bytes == 0

    def test_missing_tier_entitlements_raises(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)

        with pytest.raises(RuntimeError, match="tier_entitlements"):
            svc.get_for_user(TEST_USER_ID)


# ---------------------------------------------------------------------------
# Task 4 tests — lazy period rollover (race-fixed)
# ---------------------------------------------------------------------------


class TestPeriodRollover:
    def test_lazy_period_rollover_one_period(self):
        """period_end was yesterday → counter resets, period advances to a future date."""
        from subscriptions.service import EntitlementsService

        usage_expired = {
            **ZERO_USAGE_ROW,
            "split_sheets_this_period": 4,
            "period_start": (datetime.now(UTC) - timedelta(days=31)).isoformat(),
            "period_end": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        }

        call_state = {"usage_reads": 0}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[FREE_SUB_ROW], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "usage_counters":
                if call_state["usage_reads"] == 0:
                    b.execute.return_value = MagicMock(data=[usage_expired], count=1)
                    call_state["usage_reads"] += 1
                else:
                    rolled = {
                        **ZERO_USAGE_ROW,
                        "split_sheets_this_period": 0,
                        "period_end": (datetime.now(UTC) + timedelta(days=15)).isoformat(),
                    }
                    b.execute.return_value = MagicMock(data=[rolled], count=1)
            return b

        sb.table.side_effect = _table
        svc = EntitlementsService(sb)
        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.usage.split_sheets_this_period == 0
        assert ent.usage.period_end > datetime.now(UTC)

    def test_lazy_period_rollover_long_gap(self):
        """period_end was ~6 months ago → rolls forward enough times to land in the future."""
        from subscriptions.service import EntitlementsService

        usage_long_expired = {
            **ZERO_USAGE_ROW,
            "split_sheets_this_period": 5,
            "period_start": (datetime.now(UTC) - timedelta(days=230)).isoformat(),
            "period_end": (datetime.now(UTC) - timedelta(days=200)).isoformat(),
        }
        call_state = {"usage_reads": 0}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "usage_counters":
                if call_state["usage_reads"] == 0:
                    b.execute.return_value = MagicMock(data=[usage_long_expired], count=1)
                    call_state["usage_reads"] += 1
                else:
                    rolled = {
                        **ZERO_USAGE_ROW,
                        "split_sheets_this_period": 0,
                        "period_end": (datetime.now(UTC) + timedelta(days=10)).isoformat(),
                    }
                    b.execute.return_value = MagicMock(data=[rolled], count=1)
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[FREE_SUB_ROW], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb.table.side_effect = _table
        svc = EntitlementsService(sb)
        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.usage.split_sheets_this_period == 0
        assert ent.usage.period_end > datetime.now(UTC)

    def test_rollover_update_uses_race_fix_where_clause(self):
        """The UPDATE chain must include lt('period_end', new_period_end) to win exactly one racer."""
        from subscriptions.service import EntitlementsService

        usage_expired = {
            **ZERO_USAGE_ROW,
            "split_sheets_this_period": 1,
            "period_end": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
        }
        captured = {"lt_calls": []}
        call_state = {"usage_reads": 0}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "usage_counters":
                if call_state["usage_reads"] == 0:
                    b.execute.return_value = MagicMock(data=[usage_expired], count=1)
                    call_state["usage_reads"] += 1
                else:
                    b.execute.return_value = MagicMock(
                        data=[
                            {
                                **usage_expired,
                                "split_sheets_this_period": 0,
                                "period_end": (datetime.now(UTC) + timedelta(days=29)).isoformat(),
                            }
                        ],
                        count=1,
                    )
                # Capture lt() calls to verify race-fix clause
                original_lt = b.lt

                def _capture_lt(field, value):
                    captured["lt_calls"].append((field, value))
                    return original_lt(field, value)

                b.lt = _capture_lt
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[FREE_SUB_ROW], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb.table.side_effect = _table
        svc = EntitlementsService(sb)
        svc.get_for_user(TEST_USER_ID)

        # Service should have called .lt('period_end', <new_period_end>) on the rollover UPDATE
        period_end_lts = [c for c in captured["lt_calls"] if c[0] == "period_end"]
        assert len(period_end_lts) >= 1, "Race-fix WHERE period_end < new_period_end clause missing"


# ---------------------------------------------------------------------------
# Task 5 tests — can() chokepoint, host-wins, degraded fallback
# ---------------------------------------------------------------------------


def _free_acting_pro_host_supabase():
    """Acting user is Free; host (HOST_USER_ID) is Pro. Both tier rows available to merge."""
    return _make_supabase_per_user_rows(
        {
            TEST_USER_ID: {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            },
            HOST_USER_ID: {
                "subscriptions": [PRO_HOST_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [HOST_USAGE_ROW],
            },
        }
    )


class TestCanCreate:
    def test_can_create_artist_under_cap(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.CREATE_ARTIST, current_count=2)
        assert result.allowed is True

    def test_can_create_artist_at_cap(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.CREATE_ARTIST, current_count=3)
        assert result.allowed is False
        assert result.upgrade_required is True

    def test_can_create_artist_pro_unlimited(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [PRO_SUB_ROW],
                "tier_entitlements": [PRO_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.CREATE_ARTIST, current_count=999)
        assert result.allowed is True

    def test_can_create_ignores_host(self):
        """Cap actions: host-wins does NOT apply. Free user at cap = blocked even on a Pro host's project."""
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _free_acting_pro_host_supabase()
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.CREATE_ARTIST, current_count=3, host_user_id=HOST_USER_ID)
        assert result.allowed is False  # Cap is on YOUR own artists; host irrelevant.


class TestCanFeatureHostWins:
    def test_can_use_zoe_blocked_on_free(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_ZOE)
        assert result.allowed is False

    def test_can_use_zoe_allowed_via_pro_host(self):
        """Host-wins: Free user can use Zoe on a Pro host's project."""
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _free_acting_pro_host_supabase()
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_ZOE, host_user_id=HOST_USER_ID)
        assert result.allowed is True

    def test_can_use_zoe_blocked_when_host_also_free(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_per_user_rows(
            {
                TEST_USER_ID: {
                    "subscriptions": [FREE_SUB_ROW],
                    "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                    "tier_overrides": [],
                    "usage_counters": [ZERO_USAGE_ROW],
                },
                HOST_USER_ID: {
                    "subscriptions": [{**FREE_SUB_ROW, "user_id": HOST_USER_ID}],
                    "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                    "tier_overrides": [],
                    "usage_counters": [HOST_USAGE_ROW],
                },
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_ZOE, host_user_id=HOST_USER_ID)
        assert result.allowed is False

    def test_can_use_integration_drive_on_free(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_INTEGRATION, name="google_drive")
        assert result.allowed is True

    def test_can_use_integration_slack_blocked_on_free(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [ZERO_USAGE_ROW],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_INTEGRATION, name="slack")
        assert result.allowed is False

    def test_can_use_integration_slack_allowed_via_pro_host(self):
        """Host-wins: Free user can use Slack integration on Pro host's project."""
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _free_acting_pro_host_supabase()
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_INTEGRATION, name="slack", host_user_id=HOST_USER_ID)
        assert result.allowed is True


class TestCanUpload:
    def test_can_upload_uses_acting_user_when_no_host(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [{**ZERO_USAGE_ROW, "total_storage_bytes": 500_000_000}],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.UPLOAD_BYTES, size=100_000)
        assert result.allowed is True

    def test_can_upload_blocks_at_cap_on_own_project(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [{**ZERO_USAGE_ROW, "total_storage_bytes": 1023 * 1024 * 1024}],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.UPLOAD_BYTES, size=2 * 1024 * 1024)
        assert result.allowed is False
        assert "storage" in result.reason.lower()

    def test_can_upload_uses_host_when_host_provided(self):
        """Owner-scoped storage: uploading to a Pro host's project never blocks (host has unlimited)."""
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_per_user_rows(
            {
                TEST_USER_ID: {
                    "subscriptions": [FREE_SUB_ROW],
                    "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                    "tier_overrides": [],
                    "usage_counters": [{**ZERO_USAGE_ROW, "total_storage_bytes": 1024 * 1024 * 1024}],
                },
                HOST_USER_ID: {
                    "subscriptions": [PRO_HOST_SUB_ROW],
                    "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                    "tier_overrides": [],
                    "usage_counters": [HOST_USAGE_ROW],
                },
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.UPLOAD_BYTES, size=10 * 1024 * 1024, host_user_id=HOST_USER_ID)
        assert result.allowed is True


class TestCanSplitSheet:
    def test_can_generate_split_sheet_at_cap(self):
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [{**ZERO_USAGE_ROW, "split_sheets_this_period": 5}],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.GENERATE_SPLIT_SHEET)
        assert result.allowed is False


class TestCanOneClick:
    def test_free_user_under_cap_allowed(self):
        """Free user with 0 runs this period and cap=1 should be allowed."""
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [{**ZERO_USAGE_ROW, "oneclick_runs_this_period": 0}],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_ONECLICK)
        assert result.allowed is True

    def test_free_user_at_cap_denied(self):
        """Free user with 1 run this period and cap=1 should be denied."""
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [FREE_SUB_ROW],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [{**ZERO_USAGE_ROW, "oneclick_runs_this_period": 1}],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_ONECLICK)
        assert result.allowed is False
        assert result.upgrade_required is True
        assert "OneClick" in result.reason

    def test_pro_user_unlimited(self):
        """Pro user with cap=-1 should always be allowed regardless of run count."""
        from subscriptions.models import Action
        from subscriptions.service import EntitlementsService

        sb = _make_supabase_with_rows(
            {
                "subscriptions": [PRO_SUB_ROW],
                "tier_entitlements": [PRO_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [{**ZERO_USAGE_ROW, "oneclick_runs_this_period": 999}],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.can(TEST_USER_ID, Action.USE_ONECLICK)
        assert result.allowed is True


class TestDegradedFallback:
    def test_get_for_user_safe_returns_degraded_on_db_error(self):
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        sb.table.side_effect = RuntimeError("DB unreachable")
        svc = EntitlementsService(sb)

        ent = svc.get_for_user_safe(TEST_USER_ID)
        assert ent.degraded is True
        assert ent.tier == "free"
        assert ent.features.zoe_enabled is False
        assert ent.caps.max_artists == 3


# ---------------------------------------------------------------------------
# SP2 tests — increment_usage + extended rollover
# ---------------------------------------------------------------------------


class TestIncrementUsage:
    def test_calls_rpc_with_correct_args(self):
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        rpc_calls = []
        sb.rpc.side_effect = lambda fn, args: rpc_calls.append((fn, args)) or MagicMock(
            execute=MagicMock(return_value=MagicMock(data=None))
        )
        svc = EntitlementsService(sb)

        svc.increment_usage(TEST_USER_ID, "zoe_queries_this_period")
        assert rpc_calls == [
            (
                "increment_usage_counter",
                {"p_user_id": TEST_USER_ID, "p_counter_name": "zoe_queries_this_period"},
            )
        ]

    def test_invalid_counter_does_not_call_rpc(self):
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        svc = EntitlementsService(sb)

        svc.increment_usage(TEST_USER_ID, "bogus_counter")
        sb.rpc.assert_not_called()

    def test_swallows_rpc_exception(self):
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        sb.rpc.side_effect = RuntimeError("DB unreachable")
        svc = EntitlementsService(sb)

        svc.increment_usage(TEST_USER_ID, "zoe_queries_this_period")


class TestRolloverResetsNewCounters:
    def test_rollover_resets_zoe_and_oneclick_counters(self):
        """The lazy rollover UPDATE payload must include zero-resets for the new counters."""
        from subscriptions.service import EntitlementsService

        usage_expired = {
            **ZERO_USAGE_ROW,
            "split_sheets_this_period": 4,
            "zoe_queries_this_period": 7,
            "oneclick_runs_this_period": 3,
            "period_start": (datetime.now(UTC) - timedelta(days=31)).isoformat(),
            "period_end": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        }

        captured_updates = []
        call_state = {"usage_reads": 0}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "usage_counters":
                if call_state["usage_reads"] == 0:
                    b.execute.return_value = MagicMock(data=[usage_expired], count=1)
                    call_state["usage_reads"] += 1
                else:
                    rolled = {
                        **ZERO_USAGE_ROW,
                        "period_end": (datetime.now(UTC) + timedelta(days=15)).isoformat(),
                    }
                    b.execute.return_value = MagicMock(data=[rolled], count=1)
                original_update = b.update

                def _capture_update(payload, *a, **kw):
                    captured_updates.append(payload)
                    return original_update(payload, *a, **kw)

                b.update = _capture_update
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[FREE_SUB_ROW], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[FREE_TIER_ROW], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb.table.side_effect = _table
        svc = EntitlementsService(sb)
        svc.get_for_user(TEST_USER_ID)

        assert any(
            u.get("zoe_queries_this_period") == 0 and u.get("oneclick_runs_this_period") == 0 for u in captured_updates
        ), f"rollover UPDATE should reset zoe + oneclick counters; saw: {captured_updates}"


# ---------------------------------------------------------------------------
# Beta bypass + admin entitlement tests (Task 3 of Beta+Tester plan)
# ---------------------------------------------------------------------------


def _make_free_supabase():
    """Returns a mock Supabase client configured with Free-tier rows for TEST_USER_ID."""
    return _make_supabase_with_rows(
        {
            "subscriptions": [FREE_SUB_ROW],
            "tier_entitlements": [FREE_TIER_ROW],
            "tier_overrides": [],
            "usage_counters": [ZERO_USAGE_ROW],
        }
    )


class TestBypassPaywalls:
    def test_bypass_off_normal_user_normal_caps(self, monkeypatch):
        """With BYPASS_PAYWALLS unset/false, a Free user gets normal Free caps."""
        from subscriptions.service import EntitlementsService

        monkeypatch.delenv("BYPASS_PAYWALLS", raising=False)

        svc = EntitlementsService(_make_free_supabase())
        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.caps.max_artists == 3
        assert ent.caps.max_projects == 3
        assert ent.features.zoe_enabled is False
        assert ent.features.registry_enabled is False

    def test_bypass_on_normal_user_max_caps(self, monkeypatch):
        """With BYPASS_PAYWALLS=true, a Free user receives all caps=-1 and all features=true."""
        from subscriptions.service import EntitlementsService

        monkeypatch.setenv("BYPASS_PAYWALLS", "true")

        svc = EntitlementsService(_make_free_supabase())
        ent = svc.get_for_user(TEST_USER_ID)

        assert ent.caps.max_artists == -1
        assert ent.caps.max_projects == -1
        assert ent.caps.max_tasks == -1
        assert ent.caps.max_storage_bytes == -1
        assert ent.caps.max_split_sheets_per_month == -1
        assert ent.caps.max_oneclick_runs_per_month == -1
        assert ent.features.zoe_enabled is True
        assert ent.features.oneclick_enabled is True
        assert ent.features.registry_enabled is True
        assert set(ent.features.integrations_allowed) == {"google_drive", "slack", "notion", "monday"}
        # Usage, tier string, status, user_id must be preserved
        assert ent.tier == "free"
        assert ent.user_id == TEST_USER_ID
        assert ent.usage.total_storage_bytes == 0

    def test_admin_user_gets_max_caps(self, monkeypatch):
        """is_admin=True gives a Free user Pro-shaped entitlements even when bypass is off."""
        from subscriptions.service import EntitlementsService

        monkeypatch.delenv("BYPASS_PAYWALLS", raising=False)

        svc = EntitlementsService(_make_free_supabase())
        ent = svc.get_for_user(TEST_USER_ID, is_admin=True)

        assert ent.caps.max_artists == -1
        assert ent.caps.max_oneclick_runs_per_month == -1
        assert ent.features.zoe_enabled is True
        assert ent.features.registry_enabled is True
        assert "slack" in ent.features.integrations_allowed

    def test_non_admin_user_normal_caps_when_bypass_off(self, monkeypatch):
        """is_admin=False + BYPASS_PAYWALLS unset → normal Free caps (no accidental elevation)."""
        from subscriptions.service import EntitlementsService

        monkeypatch.delenv("BYPASS_PAYWALLS", raising=False)

        svc = EntitlementsService(_make_free_supabase())
        ent = svc.get_for_user(TEST_USER_ID, is_admin=False)

        assert ent.caps.max_artists == 3
        assert ent.features.zoe_enabled is False
        assert ent.features.registry_enabled is False
