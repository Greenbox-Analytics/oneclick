"""Endpoint tests for /admin/*. Exercises require_admin + delegation to AdminService."""

from unittest.mock import MagicMock

import pytest

from tests.conftest import TEST_USER_ID, MockQueryBuilder

ADMIN_EMAIL = "admin@example.com"
NON_ADMIN_EMAIL = "user@example.com"


@pytest.fixture(autouse=True)
def _set_admin_emails(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", ADMIN_EMAIL)
    yield


@pytest.fixture(autouse=True)
def _reset_admin_service_singleton():
    from subscriptions import admin_router as r

    r._admin_service = None
    yield
    r._admin_service = None


@pytest.fixture
def admin_client(mock_supabase):
    """FastAPI TestClient where get_current_user_email returns ADMIN_EMAIL."""
    from fastapi.testclient import TestClient

    import main
    from auth import get_current_user_email, get_current_user_id

    main.get_supabase_client = lambda: mock_supabase
    main.supabase = mock_supabase

    async def _admin_email():
        return ADMIN_EMAIL

    async def _admin_uid():
        return TEST_USER_ID

    main.app.dependency_overrides[get_current_user_email] = _admin_email
    main.app.dependency_overrides[get_current_user_id] = _admin_uid

    with TestClient(main.app) as tc:
        yield tc

    main.app.dependency_overrides.clear()


@pytest.fixture
def non_admin_client(mock_supabase):
    """FastAPI TestClient where get_current_user_email returns a non-admin email."""
    from fastapi.testclient import TestClient

    import main
    from auth import get_current_user_email, get_current_user_id

    main.get_supabase_client = lambda: mock_supabase
    main.supabase = mock_supabase

    async def _user_email():
        return NON_ADMIN_EMAIL

    async def _user_uid():
        return TEST_USER_ID

    main.app.dependency_overrides[get_current_user_email] = _user_email
    main.app.dependency_overrides[get_current_user_id] = _user_uid

    with TestClient(main.app) as tc:
        yield tc

    main.app.dependency_overrides.clear()


class TestAdminMe:
    def test_returns_200_for_admin(self, admin_client):
        resp = admin_client.get("/admin/me")
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == ADMIN_EMAIL
        assert body["isAdmin"] is True

    def test_returns_403_for_non_admin(self, non_admin_client):
        resp = non_admin_client.get("/admin/me")
        assert resp.status_code == 403

    def test_returns_500_when_admin_emails_unset(self, admin_client, monkeypatch):
        monkeypatch.setenv("ADMIN_EMAILS", "")
        resp = admin_client.get("/admin/me")
        assert resp.status_code == 500

    def test_email_match_is_case_insensitive(self, admin_client, monkeypatch):
        monkeypatch.setenv("ADMIN_EMAILS", ADMIN_EMAIL.upper())
        resp = admin_client.get("/admin/me")
        assert resp.status_code == 200


class TestListUsers:
    def test_admin_can_list(self, admin_client, mock_supabase):
        mock_supabase.auth.admin.list_users.return_value = [
            MagicMock(id=TEST_USER_ID, email="a@example.com", created_at="2026-05-01T00:00:00+00:00"),
        ]

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[{"user_id": TEST_USER_ID, "tier": "free"}], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.get("/admin/users")
        assert resp.status_code == 200
        body = resp.json()
        assert "users" in body
        assert len(body["users"]) == 1
        assert body["users"][0]["tier"] == "free"

    def test_non_admin_blocked(self, non_admin_client):
        resp = non_admin_client.get("/admin/users")
        assert resp.status_code == 403


class TestGetUserDetail:
    def test_returns_user_plus_entitlements_plus_override(self, admin_client, mock_supabase):
        mock_supabase.auth.admin.get_user_by_id.return_value = MagicMock(
            user=MagicMock(
                id=TEST_USER_ID,
                email="a@example.com",
                created_at="2026-05-01T00:00:00+00:00",
            ),
        )
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

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(data=[free_sub], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(data=[free_tier], count=1)
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[], count=0)
            elif name == "usage_counters":
                b.execute.return_value = MagicMock(data=[zero_usage], count=1)
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.get(f"/admin/users/{TEST_USER_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user"]["email"] == "a@example.com"
        assert body["entitlements"]["tier"] == "free"
        # Raw override row included for editor pre-fill (None when no override exists)
        assert body["override"] is None


class TestGrantRevoke:
    def test_grant_returns_ok(self, admin_client, mock_supabase):
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.post(f"/admin/users/{TEST_USER_ID}/grant")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert captured["payload"]["tier"] == "pro"

    def test_revoke_returns_ok(self, admin_client, mock_supabase):
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.post(f"/admin/users/{TEST_USER_ID}/revoke")
        assert resp.status_code == 200
        assert captured["payload"]["tier"] == "free"

    def test_grant_unknown_user_returns_400(self, admin_client, mock_supabase):
        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                # Simulate FK violation
                b.execute.side_effect = RuntimeError("violates foreign key constraint subscriptions_user_id_fkey")
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.post("/admin/users/00000000-0000-0000-0000-000000000bad/grant")
        assert resp.status_code == 400


class TestOverride:
    def test_apply_override_returns_ok(self, admin_client, mock_supabase):
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.post(
            f"/admin/users/{TEST_USER_ID}/override",
            json={"max_artists": 10, "zoe_enabled": True, "reason": "Beta"},
        )
        assert resp.status_code == 200
        assert captured["payload"]["max_artists"] == 10
        assert captured["payload"]["zoe_enabled"] is True

    def test_apply_override_validates_negative_expires_days(self, admin_client):
        resp = admin_client.post(
            f"/admin/users/{TEST_USER_ID}/override",
            json={"expires_days": -5},
        )
        assert resp.status_code == 422

    def test_clear_override_returns_ok(self, admin_client, mock_supabase):
        captured = {}

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

        mock_supabase.table.side_effect = _table

        resp = admin_client.delete(f"/admin/users/{TEST_USER_ID}/override")
        assert resp.status_code == 200
        assert captured["user_id"] == TEST_USER_ID


class TestProRequests:
    def test_list_returns_rows(self, admin_client, mock_supabase):
        def _table(name):
            b = MockQueryBuilder()
            if name == "pro_requests":
                b.execute.return_value = MagicMock(
                    data=[{"id": "p1", "email": "a@x.com", "status": "new"}],
                    count=1,
                )
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.get("/admin/pro-requests")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 1


class TestNonAdminBlocked:
    """Parameterized: every /admin/* route returns 403 for non-admin."""

    ROUTES = [
        ("GET", "/admin/me"),
        ("GET", "/admin/users"),
        ("GET", f"/admin/users/{TEST_USER_ID}"),
        ("POST", f"/admin/users/{TEST_USER_ID}/grant"),
        ("POST", f"/admin/users/{TEST_USER_ID}/revoke"),
        ("POST", f"/admin/users/{TEST_USER_ID}/override"),
        ("DELETE", f"/admin/users/{TEST_USER_ID}/override"),
        ("GET", "/admin/pro-requests"),
    ]

    @pytest.mark.parametrize("method,path", ROUTES)
    def test_non_admin_blocked(self, method, path, non_admin_client):
        resp = non_admin_client.request(method, path, json={} if method in ("POST",) else None)
        assert resp.status_code == 403, f"{method} {path} should be 403 for non-admin"
