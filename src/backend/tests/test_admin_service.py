"""Unit tests for AdminService — admin operations on subscriptions and overrides."""

from datetime import UTC
from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder


def _wire_supabase(rows_by_table: dict[str, list[dict]]):
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


class TestSetTier:
    def test_set_tier_pro_upserts_subscriptions(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    captured["on_conflict"] = kw.get("on_conflict")
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.set_tier(TEST_USER_ID, "pro")
        assert captured["payload"]["tier"] == "pro"
        assert captured["payload"]["user_id"] == TEST_USER_ID

    def test_set_tier_free(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.set_tier(TEST_USER_ID, "free")
        assert captured["payload"]["tier"] == "free"


class TestApplyOverride:
    def test_only_supplied_fields_in_upsert(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.apply_override(TEST_USER_ID, {"max_artists": 10, "zoe_enabled": True, "reason": "Beta"})

        payload = captured["payload"]
        assert payload["max_artists"] == 10
        assert payload["zoe_enabled"] is True
        assert payload["reason"] == "Beta"
        assert "max_projects" not in payload

    def test_does_not_include_granted_by(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.apply_override(TEST_USER_ID, {"max_artists": 10})
        assert "granted_by" not in captured["payload"]

    def test_expires_days_converted_to_expires_at(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.apply_override(TEST_USER_ID, {"expires_days": 30})

        assert "expires_days" not in captured["payload"]
        assert "expires_at" in captured["payload"]
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        exp = _dt.fromisoformat(captured["payload"]["expires_at"])
        delta = exp - _dt.now(UTC)
        assert _td(days=29) < delta < _td(days=31)


class TestClearOverride:
    def test_deletes_by_user_id(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original = b.eq

                def _capture(field, value):
                    if field == "user_id":
                        captured["user_id"] = value
                    return original(field, value)

                b.eq = _capture
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.clear_override(TEST_USER_ID)
        assert captured["user_id"] == TEST_USER_ID


class TestListProRequests:
    def test_no_filter_returns_all(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        sb = _wire_supabase(
            {
                "pro_requests": [
                    {
                        "id": "p1",
                        "email": "a@x.com",
                        "status": "new",
                        "message": None,
                        "user_id": None,
                        "created_at": "2026-05-09T00:00:00+00:00",
                    },
                    {
                        "id": "p2",
                        "email": "b@x.com",
                        "status": "contacted",
                        "message": "hi",
                        "user_id": None,
                        "created_at": "2026-05-09T00:00:00+00:00",
                    },
                ],
            }
        )
        svc = AdminService(sb, EntitlementsService(sb))

        result = svc.list_pro_requests()
        assert len(result) == 2

    def test_filter_by_status(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "pro_requests":
                original = b.eq

                def _capture(field, value):
                    captured.setdefault("eq", []).append((field, value))
                    return original(field, value)

                b.eq = _capture
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.list_pro_requests(status="new")
        assert ("status", "new") in captured["eq"]


class TestListUsers:
    def test_returns_paginated_shape(self):
        """Returns dict with keys: users, page, per_page, has_more."""
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[{"user_id": TEST_USER_ID, "tier": "free"}], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb.table.side_effect = _table
        sb.auth.admin.list_users.return_value = [
            MagicMock(id=TEST_USER_ID, email="test@example.com", created_at="2026-05-01T00:00:00+00:00"),
        ]

        svc = AdminService(sb, EntitlementsService(sb))
        result = svc.list_users(search="", page=1, per_page=25)

        assert "users" in result
        assert "page" in result
        assert "per_page" in result
        assert "has_more" in result
        assert result["page"] == 1
        assert result["per_page"] == 25
        assert result["has_more"] is False


class TestGetUserDetail:
    def test_returns_user_plus_entitlements_plus_raw_override(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        sb.auth.admin.get_user_by_id.return_value = MagicMock(
            user=MagicMock(id=TEST_USER_ID, email="test@example.com", created_at="2026-05-01T00:00:00+00:00"),
        )

        free_sub = {
            "user_id": TEST_USER_ID,
            "tier": "free",
            "status": "active",
            "id": "s1",
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "stripe_price_id": None,
            "current_period_start": None,
            "current_period_end": None,
            "cancel_at_period_end": False,
            "canceled_at": None,
            "created_at": "2026-05-01T00:00:00+00:00",
            "updated_at": "2026-05-01T00:00:00+00:00",
        }
        free_tier = {
            "tier": "free",
            "max_artists": 3,
            "max_projects": 3,
            "max_boards": 3,
            "max_tasks": 50,
            "max_storage_bytes": 1073741824,
            "max_split_sheets_per_month": 5,
            "zoe_enabled": False,
            "oneclick_enabled": False,
            "registry_enabled": False,
            "integrations_allowed": ["google_drive"],
            "updated_at": "2026-05-09T00:00:00+00:00",
        }
        zero_usage = {
            "user_id": TEST_USER_ID,
            "total_storage_bytes": 0,
            "split_sheets_this_period": 0,
            "zoe_queries_this_period": 0,
            "oneclick_runs_this_period": 0,
            "period_start": "2026-05-09T00:00:00+00:00",
            "period_end": "2099-05-09T00:00:00+00:00",
            "updated_at": "2026-05-09T00:00:00+00:00",
        }
        raw_override = {
            "user_id": TEST_USER_ID,
            "max_artists": 10,
            "zoe_enabled": True,
            "max_projects": None,
            "max_boards": None,
            "max_tasks": None,
            "max_storage_bytes": None,
            "max_split_sheets_per_month": None,
            "oneclick_enabled": None,
            "registry_enabled": None,
            "integrations_allowed": None,
            "reason": "Beta",
            "granted_at": "2026-05-09T00:00:00+00:00",
            "expires_at": None,
        }

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[free_sub], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[free_tier], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[raw_override], count=1)
            elif name == "usage_counters":
                b.execute.return_value = MagicMock(data=[zero_usage], count=1)
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        result = svc.get_user_detail(TEST_USER_ID)
        assert result["user"]["email"] == "test@example.com"
        assert result["entitlements"]["tier"] == "free"
        assert result["override"] is not None
        assert result["override"]["max_artists"] == 10
        assert result["override"]["zoe_enabled"] is True
