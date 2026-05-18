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

    def test_returns_200_with_isAdmin_false_for_non_admin(self, non_admin_client):
        """/admin/me is a status probe — non-admins get 200 + isAdmin: false,
        NOT a 403. The 403 produced console noise for every non-admin user on
        every page load. Other admin endpoints (grant, override, list_users)
        still 403 — only this status check is open."""
        resp = non_admin_client.get("/admin/me")
        assert resp.status_code == 200
        body = resp.json()
        assert body["isAdmin"] is False

    def test_returns_isAdmin_false_when_admin_emails_unset(self, admin_client, monkeypatch):
        """Operator misconfig (no admins) → /admin/me returns isAdmin: false.
        Fail-loud-on-misconfig is still in effect for protected admin actions
        (those raise 500 via require_admin); /admin/me itself is now a
        non-blocking status probe."""
        monkeypatch.setenv("ADMIN_EMAILS", "")
        resp = admin_client.get("/admin/me")
        assert resp.status_code == 200
        assert resp.json()["isAdmin"] is False

    def test_email_match_is_case_insensitive(self, admin_client, monkeypatch):
        monkeypatch.setenv("ADMIN_EMAILS", ADMIN_EMAIL.upper())
        resp = admin_client.get("/admin/me")
        assert resp.status_code == 200


class TestRequireAdminDbPath:
    """require_admin must accept users whose profiles.is_admin = true even
    if their email is NOT in ADMIN_EMAILS."""

    def _install_profiles_side_effect(self, mock_supabase, profiles_builder):
        """Install a side_effect that returns *profiles_builder* for the
        'profiles' table and delegates to the default for everything else."""
        original_side = mock_supabase.table.side_effect

        def _side(name):
            if name == "profiles":
                return profiles_builder
            return original_side(name)

        mock_supabase.table.side_effect = _side

    def test_db_admin_with_empty_env_can_access_admin_me(self, mock_supabase, monkeypatch):
        from fastapi.testclient import TestClient

        import main
        from auth import get_current_user_email, get_current_user_id

        monkeypatch.setenv("ADMIN_EMAILS", "")  # no env admins at all
        DB_ADMIN_EMAIL = "dbadmin@example.com"
        DB_ADMIN_UID = "11111111-1111-1111-1111-111111111111"

        # Profile lookup for the caller returns is_admin=true. The bootstrap
        # check (any admin exists) ALSO needs to see at least one row, so the
        # same builder serves both calls — its .execute returns the same data
        # each time (the chain returns self).
        profiles_builder = MockQueryBuilder()
        profiles_builder.execute.return_value = MagicMock(data=[{"is_admin": True}])
        self._install_profiles_side_effect(mock_supabase, profiles_builder)

        main.get_supabase_client = lambda: mock_supabase
        main.supabase = mock_supabase

        async def _email():
            return DB_ADMIN_EMAIL

        async def _uid():
            return DB_ADMIN_UID

        main.app.dependency_overrides[get_current_user_email] = _email
        main.app.dependency_overrides[get_current_user_id] = _uid

        with TestClient(main.app) as tc:
            r = tc.get("/admin/me")
        main.app.dependency_overrides.clear()

        assert r.status_code == 200
        assert r.json() == {"email": DB_ADMIN_EMAIL, "isAdmin": True}

    def test_non_admin_with_empty_env_via_admin_me_returns_isAdmin_false(self, mock_supabase, monkeypatch):
        """/admin/me is a status probe → returns isAdmin: false even when env
        is empty + no DB admin (the operator-misconfig 500 still fires for
        protected actions like /admin/users — see TestNonAdminBlocked)."""
        from fastapi.testclient import TestClient

        import main
        from auth import get_current_user_email, get_current_user_id

        monkeypatch.setenv("ADMIN_EMAILS", "")
        NON_ADMIN_UID = "22222222-2222-2222-2222-222222222222"

        profiles_builder = MockQueryBuilder()
        profiles_builder.execute.return_value = MagicMock(data=[])
        self._install_profiles_side_effect(mock_supabase, profiles_builder)

        main.get_supabase_client = lambda: mock_supabase
        main.supabase = mock_supabase

        async def _email():
            return "user@example.com"

        async def _uid():
            return NON_ADMIN_UID

        main.app.dependency_overrides[get_current_user_email] = _email
        main.app.dependency_overrides[get_current_user_id] = _uid

        with TestClient(main.app) as tc:
            r = tc.get("/admin/me")
        main.app.dependency_overrides.clear()

        assert r.status_code == 200
        assert r.json()["isAdmin"] is False

    def test_non_admin_with_env_configured_via_admin_me_returns_isAdmin_false(self, mock_supabase, monkeypatch):
        """Env configured, caller not admin via either path → /admin/me
        returns isAdmin: false (200). Other admin endpoints still 403."""
        from fastapi.testclient import TestClient

        import main
        from auth import get_current_user_email, get_current_user_id

        monkeypatch.setenv("ADMIN_EMAILS", "root@example.com")
        NON_ADMIN_UID = "44444444-4444-4444-4444-444444444444"

        profiles_builder = MockQueryBuilder()
        profiles_builder.execute.return_value = MagicMock(data=[])
        self._install_profiles_side_effect(mock_supabase, profiles_builder)

        main.get_supabase_client = lambda: mock_supabase
        main.supabase = mock_supabase

        async def _email():
            return "user@example.com"

        async def _uid():
            return NON_ADMIN_UID

        main.app.dependency_overrides[get_current_user_email] = _email
        main.app.dependency_overrides[get_current_user_id] = _uid

        with TestClient(main.app) as tc:
            r = tc.get("/admin/me")
        main.app.dependency_overrides.clear()

        assert r.status_code == 200
        assert r.json()["isAdmin"] is False

        # And the protected endpoints DO still 403:
        main.app.dependency_overrides[get_current_user_email] = _email
        main.app.dependency_overrides[get_current_user_id] = _uid
        with TestClient(main.app) as tc:
            r2 = tc.get("/admin/users")
        main.app.dependency_overrides.clear()
        assert r2.status_code == 403

    def test_env_admin_works_even_when_db_lookup_fails(self, mock_supabase, monkeypatch):
        """If profiles lookup raises, env-admin path still works."""
        from fastapi.testclient import TestClient

        import main
        from auth import get_current_user_email, get_current_user_id

        monkeypatch.setenv("ADMIN_EMAILS", "root@example.com")

        profiles_builder = MockQueryBuilder()
        profiles_builder.execute.side_effect = Exception("DB unreachable")
        self._install_profiles_side_effect(mock_supabase, profiles_builder)

        main.get_supabase_client = lambda: mock_supabase
        main.supabase = mock_supabase

        async def _email():
            return "root@example.com"

        async def _uid():
            return "33333333-3333-3333-3333-333333333333"

        main.app.dependency_overrides[get_current_user_email] = _email
        main.app.dependency_overrides[get_current_user_id] = _uid

        with TestClient(main.app) as tc:
            r = tc.get("/admin/me")
        main.app.dependency_overrides.clear()

        assert r.status_code == 200


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
            "max_oneclick_runs_per_month": 1,
            "zoe_enabled": False,
            "oneclick_enabled": True,
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
        builders = {}

        def _table(name):
            b = MockQueryBuilder()
            builders[name] = b
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.delete(f"/admin/users/{TEST_USER_ID}/override")
        assert resp.status_code == 200
        # Verify delete().eq("user_id", ...) was called with the correct user_id
        b = builders["tier_overrides"]
        b.delete.return_value.eq.assert_called_with("user_id", TEST_USER_ID)


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


class TestTesterGrantEndpoints:
    def test_list_returns_200_for_admin(self, admin_client, mock_supabase):
        grant_row = {
            "user_id": TEST_USER_ID,
            "reason": "tester",
            "expires_at": None,
            "granted_at": "2026-05-01T00:00:00+00:00",
        }

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[grant_row], count=1)
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.get("/admin/tester-grants")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["user_id"] == TEST_USER_ID

    def test_list_returns_403_for_non_admin(self, non_admin_client):
        resp = non_admin_client.get("/admin/tester-grants")
        assert resp.status_code == 403

    def test_create_returns_404_when_user_not_found(self, admin_client, mock_supabase):
        mock_supabase.auth.admin.list_users.return_value = []

        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "nobody@example.com"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_create_returns_422_invalid_email(self, admin_client):
        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "not-an-email"},
        )
        assert resp.status_code == 422

    def test_create_calls_service_with_correct_args(self, admin_client, mock_supabase):
        mock_supabase.auth.admin.list_users.return_value = [
            MagicMock(id=TEST_USER_ID, email="tester@example.com"),
        ]
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
            "/admin/tester-grants",
            json={"email": "tester@example.com", "expires_at": "2027-01-01T00:00:00+00:00", "reason": "tester-beta"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "tester@example.com"
        assert body["reason"] == "tester-beta"
        assert body["expires_at"] == "2027-01-01T00:00:00+00:00"

    def test_delete_returns_204(self, admin_client, mock_supabase):
        builders = {}

        def _table(name):
            b = MockQueryBuilder()
            builders[name] = b
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.delete(f"/admin/tester-grants/{TEST_USER_ID}")
        assert resp.status_code == 204
        assert resp.content == b""
        b = builders["tier_overrides"]
        b.delete.return_value.eq.assert_called_with("user_id", TEST_USER_ID)


class TestPromoteDemote:
    """POST /admin/users/{id}/promote and /demote endpoints."""

    def test_promote_returns_ok(self, admin_client, mock_supabase):
        target_uid = "77777777-7777-7777-7777-777777777777"
        r = admin_client.post(f"/admin/users/{target_uid}/promote")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_demote_returns_ok_for_other_user(self, admin_client, mock_supabase):
        target_uid = "88888888-8888-8888-8888-888888888888"
        mock_user = MagicMock()
        mock_user.email = "other@example.com"
        mock_supabase.auth.admin.get_user_by_id.return_value = MagicMock(user=mock_user)

        r = admin_client.post(f"/admin/users/{target_uid}/demote")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_demote_self_returns_400(self, admin_client):
        r = admin_client.post(f"/admin/users/{TEST_USER_ID}/demote")
        assert r.status_code == 400
        assert "yourself" in r.json()["detail"].lower()

    def test_demote_env_admin_returns_400(self, admin_client, mock_supabase):
        target_uid = "99999999-9999-9999-9999-999999999999"
        mock_user = MagicMock()
        mock_user.email = ADMIN_EMAIL
        mock_supabase.auth.admin.get_user_by_id.return_value = MagicMock(user=mock_user)

        r = admin_client.post(f"/admin/users/{target_uid}/demote")
        assert r.status_code == 400
        assert "env" in r.json()["detail"].lower()

    def test_demote_fails_closed_when_email_lookup_returns_none(self, admin_client, mock_supabase):
        """If the auth lookup can't find the user (deleted, API hiccup),
        we MUST refuse rather than silently demote — the env-admin check
        depends on a real email, and a silent demote would lie to the UI."""
        target_uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaa01"
        mock_supabase.auth.admin.get_user_by_id.side_effect = Exception("user not found")

        r = admin_client.post(f"/admin/users/{target_uid}/demote")
        assert r.status_code == 400
        assert "verify" in r.json()["detail"].lower()

    def test_non_admin_cannot_promote(self, non_admin_client):
        r = non_admin_client.post("/admin/users/00000000-0000-0000-0000-000000000099/promote")
        assert r.status_code == 403


class TestNonAdminBlocked:
    """Parameterized: every protected /admin/* route returns 403 for non-admin.

    NOTE: /admin/me is intentionally EXCLUDED — it's a status probe that
    returns 200 + isAdmin:false for non-admins (see TestAdminMe). All other
    admin endpoints still enforce 403 via require_admin."""

    ROUTES = [
        ("GET", "/admin/users"),
        ("GET", f"/admin/users/{TEST_USER_ID}"),
        ("POST", f"/admin/users/{TEST_USER_ID}/grant"),
        ("POST", f"/admin/users/{TEST_USER_ID}/revoke"),
        ("POST", f"/admin/users/{TEST_USER_ID}/override"),
        ("DELETE", f"/admin/users/{TEST_USER_ID}/override"),
        ("GET", "/admin/pro-requests"),
        ("GET", "/admin/tester-grants"),
        ("DELETE", f"/admin/tester-grants/{TEST_USER_ID}"),
    ]

    @pytest.mark.parametrize("method,path", ROUTES)
    def test_non_admin_blocked(self, method, path, non_admin_client):
        resp = non_admin_client.request(method, path, json={} if method in ("POST",) else None)
        assert resp.status_code == 403, f"{method} {path} should be 403 for non-admin"
