"""Unit tests for EntitlementsService.bulk_get_for_users."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder

# Reuse fixtures from test_entitlements_service.py shape conventions
FREE_TIER_ROW = {
    "tier": "free",
    "max_artists": 3,
    "max_projects": 3,
    "max_boards": -1,
    "max_tasks": 50,
    "max_storage_bytes": 1073741824,
    "max_split_sheets_per_month": 5,
    "zoe_enabled": False,
    "oneclick_enabled": False,
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
    "zoe_enabled": True,
    "oneclick_enabled": True,
    "registry_enabled": True,
    "integrations_allowed": ["google_drive", "slack", "notion", "monday"],
    "updated_at": "2026-05-09T00:00:00+00:00",
}
USER_A = "00000000-0000-0000-0000-00000000000a"
USER_B = "00000000-0000-0000-0000-00000000000b"


def _make_supabase(rows_by_table: dict[str, list[dict]]):
    mock = MagicMock()

    def _table(name):
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(
            data=rows_by_table.get(name, []),
            count=len(rows_by_table.get(name, [])),
        )
        return b

    mock.table.side_effect = _table
    return mock


class TestBulkGetForUsers:
    def test_empty_list_returns_empty_dict_no_db_calls(self):
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([])
        assert result == {}
        sb.table.assert_not_called()

    def test_single_user_returns_entitlements(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase(
            {
                "subscriptions": [{"user_id": USER_A, "tier": "free", "status": "active"}],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [
                    {
                        "user_id": USER_A,
                        "total_storage_bytes": 100,
                        "split_sheets_this_period": 0,
                        "zoe_queries_this_period": 0,
                        "oneclick_runs_this_period": 0,
                        "period_start": "2026-05-09T00:00:00+00:00",
                        "period_end": "2099-05-09T00:00:00+00:00",
                    }
                ],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([USER_A])

        assert USER_A in result
        ent = result[USER_A]
        assert ent.tier == "free"
        assert ent.caps.max_artists == 3
        assert ent.usage.total_storage_bytes == 100

    def test_multiple_users_mixed_tiers(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase(
            {
                "subscriptions": [
                    {"user_id": USER_A, "tier": "free", "status": "active"},
                    {"user_id": USER_B, "tier": "pro", "status": "active"},
                ],
                "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [
                    {
                        "user_id": USER_A,
                        "total_storage_bytes": 0,
                        "split_sheets_this_period": 0,
                        "zoe_queries_this_period": 0,
                        "oneclick_runs_this_period": 0,
                        "period_start": "2026-05-09T00:00:00+00:00",
                        "period_end": "2099-05-09T00:00:00+00:00",
                    },
                    {
                        "user_id": USER_B,
                        "total_storage_bytes": 0,
                        "split_sheets_this_period": 0,
                        "zoe_queries_this_period": 0,
                        "oneclick_runs_this_period": 0,
                        "period_start": "2026-05-09T00:00:00+00:00",
                        "period_end": "2099-05-09T00:00:00+00:00",
                    },
                ],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([USER_A, USER_B])

        assert result[USER_A].tier == "free"
        assert result[USER_A].features.zoe_enabled is False
        assert result[USER_B].tier == "pro"
        assert result[USER_B].features.zoe_enabled is True

    def test_active_override_applied(self):
        from subscriptions.service import EntitlementsService

        sb = _make_supabase(
            {
                "subscriptions": [{"user_id": USER_A, "tier": "free", "status": "active"}],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [
                    {
                        "user_id": USER_A,
                        "max_artists": 10,
                        "max_projects": None,
                        "max_boards": None,
                        "max_tasks": None,
                        "max_storage_bytes": None,
                        "max_split_sheets_per_month": None,
                        "zoe_enabled": True,
                        "oneclick_enabled": None,
                        "registry_enabled": None,
                        "integrations_allowed": None,
                        "reason": None,
                        "granted_at": "2026-05-09T00:00:00+00:00",
                        "expires_at": None,
                    }
                ],
                "usage_counters": [
                    {
                        "user_id": USER_A,
                        "total_storage_bytes": 0,
                        "split_sheets_this_period": 0,
                        "zoe_queries_this_period": 0,
                        "oneclick_runs_this_period": 0,
                        "period_start": "2026-05-09T00:00:00+00:00",
                        "period_end": "2099-05-09T00:00:00+00:00",
                    }
                ],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([USER_A])

        assert result[USER_A].caps.max_artists == 10
        assert result[USER_A].features.zoe_enabled is True
        assert result[USER_A].has_overrides is True

    def test_expired_override_ignored(self):
        from subscriptions.service import EntitlementsService

        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        sb = _make_supabase(
            {
                "subscriptions": [{"user_id": USER_A, "tier": "free", "status": "active"}],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [
                    {
                        "user_id": USER_A,
                        "zoe_enabled": True,
                        "max_artists": None,
                        "max_projects": None,
                        "max_boards": None,
                        "max_tasks": None,
                        "max_storage_bytes": None,
                        "max_split_sheets_per_month": None,
                        "oneclick_enabled": None,
                        "registry_enabled": None,
                        "integrations_allowed": None,
                        "reason": None,
                        "granted_at": "2026-05-09T00:00:00+00:00",
                        "expires_at": past,
                    }
                ],
                "usage_counters": [
                    {
                        "user_id": USER_A,
                        "total_storage_bytes": 0,
                        "split_sheets_this_period": 0,
                        "zoe_queries_this_period": 0,
                        "oneclick_runs_this_period": 0,
                        "period_start": "2026-05-09T00:00:00+00:00",
                        "period_end": "2099-05-09T00:00:00+00:00",
                    }
                ],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([USER_A])

        assert result[USER_A].features.zoe_enabled is False
        assert result[USER_A].has_overrides is False

    def test_missing_user_falls_back_to_free(self):
        """User in user_ids but no subscriptions row → defaults to Free."""
        from subscriptions.service import EntitlementsService

        sb = _make_supabase(
            {
                "subscriptions": [],  # empty — user not in subscriptions
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([USER_A])

        assert USER_A in result
        assert result[USER_A].tier == "free"
        assert result[USER_A].caps.max_artists == 3

    def test_missing_tier_entitlements_skips_user(self):
        """User's tier has no row in tier_entitlements → user skipped."""
        from subscriptions.service import EntitlementsService

        sb = _make_supabase(
            {
                "subscriptions": [{"user_id": USER_A, "tier": "free", "status": "active"}],
                "tier_entitlements": [],  # no rows
                "tier_overrides": [],
                "usage_counters": [],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([USER_A])

        assert USER_A not in result

    def test_mixed_tier_with_missing_subscription_row_falls_back_to_free(self):
        """Regression: USER_A has tier=pro, USER_B has no subscriptions row.
        Without the fix, the tier-fetch set is just {'pro'}, so 'free' is
        never queried, USER_B's tier_row lookup returns None, and USER_B
        gets dropped silently. With the fix (`{...} | {'free'}`), USER_B
        falls back to Free correctly."""
        from subscriptions.service import EntitlementsService

        sb = _make_supabase(
            {
                "subscriptions": [{"user_id": USER_A, "tier": "pro", "status": "active"}],
                "tier_entitlements": [FREE_TIER_ROW, PRO_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [],
            }
        )
        svc = EntitlementsService(sb)
        result = svc.bulk_get_for_users([USER_A, USER_B])

        # Both users present in result
        assert USER_A in result
        assert USER_B in result
        # USER_A is Pro; USER_B falls back to Free
        assert result[USER_A].tier == "pro"
        assert result[USER_B].tier == "free"
        assert result[USER_B].caps.max_artists == 3

    def test_does_not_trigger_period_rollover(self):
        """User with period_end < now → bulk-resolve returns the stale period without rolling forward."""
        from subscriptions.service import EntitlementsService

        past_period_end = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        sb = _make_supabase(
            {
                "subscriptions": [{"user_id": USER_A, "tier": "free", "status": "active"}],
                "tier_entitlements": [FREE_TIER_ROW],
                "tier_overrides": [],
                "usage_counters": [
                    {
                        "user_id": USER_A,
                        "total_storage_bytes": 0,
                        "split_sheets_this_period": 4,
                        "zoe_queries_this_period": 0,
                        "oneclick_runs_this_period": 0,
                        "period_start": "2026-04-09T00:00:00+00:00",
                        "period_end": past_period_end,
                    }
                ],
            }
        )
        svc = EntitlementsService(sb)

        # Track UPDATE calls — bulk should NOT update usage_counters
        update_call_count = {"count": 0}
        original_table = sb.table.side_effect

        def _track(name):
            b = original_table(name)
            original_update = b.update

            def _capture(*a, **kw):
                update_call_count["count"] += 1
                return original_update(*a, **kw)

            b.update = _capture
            return b

        sb.table.side_effect = _track

        result = svc.bulk_get_for_users([USER_A])

        # Stale split_sheets returned (not reset)
        assert result[USER_A].usage.split_sheets_this_period == 4
        # No UPDATE was issued
        assert update_call_count["count"] == 0
        # Defensive: bulk_get_for_users must make EXACTLY 4 table calls
        # (subscriptions, tier_entitlements, tier_overrides, usage_counters).
        # Any extra call would indicate an accidental N+1 or rollover side effect.
        assert sb.table.call_count == 4, f"Expected exactly 4 table() calls, got {sb.table.call_count}"
