"""Endpoint tests for /admin/*. Exercises require_admin + delegation to AdminService."""

from unittest.mock import MagicMock

import pytest

from tests.conftest import TEST_USER_ID, MockQueryBuilder

ADMIN_EMAIL = "admin@example.com"
NON_ADMIN_EMAIL = "user@example.com"
ORG_ID = "20000000-0000-0000-0000-000000000001"


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
        Protected admin actions deny with 403 (and log an ERROR for the
        operator); /admin/me itself is a non-blocking status probe."""
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
        is empty + no DB admin (protected actions like /admin/users deny with
        403 — see TestNonAdminBlocked and TestNoAdminsConfigured)."""
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
        assert captured["payload"]["tier"] == "basic"

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

    def test_create_unknown_email_returns_pending(self, admin_client, mock_supabase):
        """create_tester_grant no longer 404s for an unknown email — it parks
        a pending pre-signup designation, claimed on first verified sign-in
        (deliberate behavior change, admin credits & testers spec 2026-08-08)."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "nobody@example.com"},
        )
        assert resp.status_code == 200
        assert resp.json()["pending"] is True

    def test_create_returns_422_invalid_email(self, admin_client):
        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "not-an-email"},
        )
        assert resp.status_code == 422

    def test_create_calls_service_with_correct_args(self, admin_client, mock_supabase):
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {"id": TEST_USER_ID, "email": "tester@example.com", "created_at": "2026-01-01"},
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
        """DELETE /admin/tester-grants/{id} returns 204 and writes a sticky
        'tester_revoked' marker via UPSERT (not a hard DELETE) so that
        /me/bootstrap-tester won't auto-re-grant on next sign-in for users
        in TESTER_EMAILS."""
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original_upsert = b.upsert

                def _capture_upsert(payload, *a, **kw):
                    captured["payload"] = payload
                    captured["on_conflict"] = kw.get("on_conflict")
                    return original_upsert(payload, *a, **kw)

                b.upsert = _capture_upsert
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.delete(f"/admin/tester-grants/{TEST_USER_ID}")
        assert resp.status_code == 204
        assert resp.content == b""
        assert captured["payload"]["user_id"] == TEST_USER_ID
        assert captured["payload"]["reason"] == "tester_revoked"
        assert captured["on_conflict"] == "user_id"


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


class TestRecalcStorage:
    """POST /admin/users/{id}/recalc-storage endpoint."""

    def test_recalc_returns_freshly_computed_total(self, admin_client, mock_supabase):
        """Endpoint calls recalc_user_storage RPC, then re-reads usage_counters
        and returns the freshly-computed total."""

        # mock_supabase.rpc(...).execute() — the default MagicMock chain is fine
        # (returns MagicMock). We just need the table read to return our value.
        def _table(name):
            b = MockQueryBuilder()
            if name == "usage_counters":
                b.execute.return_value = MagicMock(data=[{"total_storage_bytes": 123456}], count=1)
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.post(f"/admin/users/{TEST_USER_ID}/recalc-storage")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"user_id": TEST_USER_ID, "total_storage_bytes": 123456}

        # And the RPC was actually called with the right shape.
        mock_supabase.rpc.assert_any_call("recalc_user_storage", {"p_user_id": TEST_USER_ID})

    def test_recalc_returns_zero_when_no_usage_row(self, admin_client, mock_supabase):
        """If the re-read returns no rows (edge case — counter row truly absent),
        the endpoint responds with total=0 rather than crashing on rows[0]."""

        def _table(name):
            b = MockQueryBuilder()
            if name == "usage_counters":
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.post(f"/admin/users/{TEST_USER_ID}/recalc-storage")
        assert resp.status_code == 200
        assert resp.json() == {"user_id": TEST_USER_ID, "total_storage_bytes": 0}

    def test_recalc_rpc_failure_returns_500(self, admin_client, mock_supabase):
        """RPC failures bubble up as 500 — the caller needs to know the recalc
        didn't happen, not see a stale total."""
        mock_supabase.rpc.return_value.execute.side_effect = RuntimeError("rpc down")

        resp = admin_client.post(f"/admin/users/{TEST_USER_ID}/recalc-storage")
        assert resp.status_code == 500
        assert "Recalc failed" in resp.json()["detail"]

    def test_recalc_non_admin_blocked(self, non_admin_client):
        resp = non_admin_client.post(f"/admin/users/{TEST_USER_ID}/recalc-storage")
        assert resp.status_code == 403


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
        ("POST", f"/admin/users/{TEST_USER_ID}/recalc-storage"),
        ("POST", f"/admin/users/{TEST_USER_ID}/override"),
        ("DELETE", f"/admin/users/{TEST_USER_ID}/override"),
        ("GET", "/admin/pro-requests"),
        ("GET", "/admin/tester-grants"),
        ("DELETE", f"/admin/tester-grants/{TEST_USER_ID}"),
        ("POST", f"/admin/orgs/{ORG_ID}/suspend"),
        ("POST", f"/admin/orgs/{ORG_ID}/reactivate"),
        ("POST", f"/admin/orgs/{ORG_ID}/pool/clawback"),
        ("GET", f"/admin/orgs/{ORG_ID}/pool"),
    ]

    @pytest.mark.parametrize("method,path", ROUTES)
    def test_non_admin_blocked(self, method, path, non_admin_client):
        resp = non_admin_client.request(method, path, json={} if method in ("POST",) else None)
        assert resp.status_code == 403, f"{method} {path} should be 403 for non-admin"


class TestNoAdminsConfigured:
    """An environment with NO admins at all — empty ADMIN_EMAILS and no
    profiles.is_admin row — must deny with 403, never 500.

    It used to 500 as a fail-loud operator signal, which made every /admin/*
    route look like a server fault on a fresh environment and misdirected anyone
    debugging it. The signal now lives where operators actually look: an ERROR
    log. The caller sees the same 403 any other non-admin sees.
    """

    def test_denies_with_403_not_500(self, non_admin_client, mock_supabase, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("ADMIN_EMAILS", "")
        # profiles: no is_admin row anywhere → the bootstrap probe finds nothing.
        mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = (
            MagicMock(data=[])
        )

        with caplog.at_level(logging.ERROR):
            resp = non_admin_client.get("/admin/users")

        assert resp.status_code == 403
        assert resp.json()["detail"] == "Admin access required"
        assert any("No admins configured" in r.message for r in caplog.records), (
            "the operator signal must survive as an ERROR log"
        )

    def test_probe_failure_still_denies_quietly(self, non_admin_client, mock_supabase, monkeypatch):
        """The bootstrap probe is diagnostics only — if the read itself raises,
        the caller still gets the same 403 rather than a 500 leaking out."""
        monkeypatch.setenv("ADMIN_EMAILS", "")
        mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.side_effect = (
            Exception("db down")
        )

        resp = non_admin_client.get("/admin/users")

        assert resp.status_code == 403


class TestOrgSuspendReactivate:
    """POST /admin/orgs/{id}/suspend and /reactivate (Licensing Phase B).

    Deliberately on THIS router (not orgs/router.py) and flag-INDEPENDENT: a
    platform admin may need to suspend/reactivate an org precisely when
    LICENSING_ENABLED is off. None of these tests set the flag."""

    @staticmethod
    def _install_org_table(mock_supabase, org_row: dict | None, captured: dict | None = None):
        """Wire mock_supabase.table('organizations') so a .select()...execute()
        returns *org_row* as a single dict (mirrors real .maybe_single()
        semantics used throughout this codebase) and a subsequent
        .update(...)...execute() returns [merged_row] as a LIST (mirrors
        supabase-py's list-shaped update response), capturing the update
        payload into *captured* if given."""

        def _table(name):
            b = MockQueryBuilder()
            if name != "organizations":
                return b

            def _select(*a, **kw):
                b.execute.return_value = MagicMock(data=org_row)
                return b

            def _update(payload, *a, **kw):
                if captured is not None:
                    captured["payload"] = payload
                merged = {**org_row, **payload} if org_row else None
                b.execute.return_value = MagicMock(data=[merged] if merged else [])
                return b

            b.select = _select
            b.update = _update
            return b

        mock_supabase.table.side_effect = _table

    def test_suspend_active_org_sets_suspended(self, admin_client, mock_supabase, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        captured = {}
        self._install_org_table(mock_supabase, {"id": ORG_ID, "status": "active"}, captured)

        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/suspend")
        assert resp.status_code == 200
        assert captured["payload"] == {"status": "suspended"}
        assert resp.json()["status"] == "suspended"

    def test_suspend_pending_org_409(self, admin_client, mock_supabase):
        self._install_org_table(mock_supabase, {"id": ORG_ID, "status": "pending"})
        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/suspend")
        assert resp.status_code == 409

    def test_suspend_already_suspended_org_409(self, admin_client, mock_supabase):
        self._install_org_table(mock_supabase, {"id": ORG_ID, "status": "suspended"})
        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/suspend")
        assert resp.status_code == 409

    def test_suspend_unknown_org_404(self, admin_client, mock_supabase):
        self._install_org_table(mock_supabase, None)
        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/suspend")
        assert resp.status_code == 404

    def test_reactivate_suspended_org_sets_active(self, admin_client, mock_supabase):
        captured = {}
        self._install_org_table(mock_supabase, {"id": ORG_ID, "status": "suspended"}, captured)

        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/reactivate")
        assert resp.status_code == 200
        assert captured["payload"] == {"status": "active"}
        assert resp.json()["status"] == "active"

    def test_reactivate_pending_org_409_not_activated(self, admin_client, mock_supabase):
        self._install_org_table(mock_supabase, {"id": ORG_ID, "status": "pending"})
        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/reactivate")
        assert resp.status_code == 409
        assert "not been activated" in resp.json()["detail"]

    def test_reactivate_active_org_409(self, admin_client, mock_supabase):
        self._install_org_table(mock_supabase, {"id": ORG_ID, "status": "active"})
        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/reactivate")
        assert resp.status_code == 409

    def test_reactivate_unknown_org_404(self, admin_client, mock_supabase):
        self._install_org_table(mock_supabase, None)
        resp = admin_client.post(f"/admin/orgs/{ORG_ID}/reactivate")
        assert resp.status_code == 404

    def test_non_admin_blocked_regardless_of_flag(self, non_admin_client, monkeypatch):
        """Flag-independence, negative direction: even with the flag fully
        unset, a non-admin still gets 403 (not a licensing-related 404) —
        confirms these routes are gated ONLY by platform require_admin."""
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = non_admin_client.post(f"/admin/orgs/{ORG_ID}/suspend")
        assert resp.status_code == 403


class TestOrgPoolEndpointsFlagIndependence:
    """POST /admin/orgs/{id}/pool/clawback and GET /admin/orgs/{id}/pool
    (follow-ups plan 2026-07-22, Task 2) placement mirrors
    TestOrgSuspendReactivate exactly: PLATFORM require_admin, flag-
    INDEPENDENT. Full behavioral coverage (RPC shape, 404s, GET shape,
    cumulativePurchased) lives in tests/test_admin_credits.py alongside the
    per-user credit tooling it mirrors; this class only pins the
    admin-gating + flag-independence contract at the router level, same as
    the suspend/reactivate class above."""

    def test_non_admin_blocked_regardless_of_flag(self, non_admin_client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = non_admin_client.post(
            f"/admin/orgs/{ORG_ID}/pool/clawback",
            json={"amount": 100, "reason": "x", "idempotency_key": "k1"},
        )
        assert resp.status_code == 403

        resp = non_admin_client.get(f"/admin/orgs/{ORG_ID}/pool")
        assert resp.status_code == 403

    def test_admin_reaches_handler_with_flag_unset(self, admin_client, mock_supabase, monkeypatch):
        """Positive direction: with LICENSING_ENABLED fully unset, an admin
        still reaches the handler (not blocked by some latent flag check) —
        a 404 here (unknown org, since mock_supabase's default table stub
        has no 'organizations' row) proves the handler ran, as opposed to a
        403/500 that would indicate an accidental flag gate."""
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        resp = admin_client.get(f"/admin/orgs/{ORG_ID}/pool")
        assert resp.status_code == 404


class TestCreateEnterpriseOrg:
    """POST /admin/orgs — Msanii-admin-only enterprise org creation (standing
    core, spec 2026-08-15 §2, review r2 hole 5). Post-migration this (and
    PUT /admin/orgs/{id}/kind's flip-to-enterprise path) are the ONLY
    producers of kind='enterprise' rows — org creation everywhere else
    (POST /orgs) is self-serve + slot-gated (orgs.service.create_org)."""

    CUSTOMER_ID = "55555555-5555-5555-5555-555555555555"

    def test_non_admin_blocked(self, non_admin_client):
        resp = non_admin_client.post("/admin/orgs", json={"name": "Acme Inc", "admin_email": "boss@acme.com"})
        assert resp.status_code == 403

    def test_creates_org_with_customer_as_created_by(self, admin_client, mock_supabase):
        """created_by must be the CUSTOMER's id, never the calling admin's —
        the auto_create_org_admin trigger seats created_by as the org's sole
        admin, so an operator-id row would make Msanii the customer's admin."""
        mock_supabase.rpc.return_value.execute.return_value.data = self.CUSTOMER_ID
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "organizations":
                original_insert = b.insert

                def _capture_insert(payload, *a, **kw):
                    captured["payload"] = payload
                    return original_insert(payload, *a, **kw)

                b.insert = _capture_insert
                b.execute.return_value = MagicMock(
                    data=[{"id": ORG_ID, "name": "Acme Inc", "created_by": self.CUSTOMER_ID, "kind": "enterprise"}],
                    count=1,
                )
            return b

        mock_supabase.table.side_effect = _table

        resp = admin_client.post("/admin/orgs", json={"name": "Acme Inc", "admin_email": "boss@acme.com"})
        assert resp.status_code == 200
        assert captured["payload"] == {"name": "Acme Inc", "created_by": self.CUSTOMER_ID, "kind": "enterprise"}
        assert resp.json()["created_by"] == self.CUSTOMER_ID

    def test_unknown_email_returns_404(self, admin_client, mock_supabase):
        """Enterprise onboarding requires an existing account — no
        invite-by-email path, unlike orgs.service.invite_member."""
        mock_supabase.rpc.return_value.execute.return_value.data = None
        resp = admin_client.post("/admin/orgs", json={"name": "Acme Inc", "admin_email": "nobody@acme.com"})
        assert resp.status_code == 404


class TestSetOrgKind:
    """PUT /admin/orgs/{org_id}/kind — flip between 'enterprise' and
    'self_serve' (standing core, Task 4)."""

    COVERER_ID = "66666666-6666-6666-6666-666666666666"

    def test_non_admin_blocked(self, non_admin_client):
        resp = non_admin_client.put(f"/admin/orgs/{ORG_ID}/kind", json={"kind": "self_serve"})
        assert resp.status_code == 403

    def test_self_serve_without_coverer_returns_422(self, admin_client, mock_supabase):
        """Without a coverer the org would sit uncovered and lapse in the
        grace window — rejected at the router before any DB read."""
        resp = admin_client.put(f"/admin/orgs/{ORG_ID}/kind", json={"kind": "self_serve"})
        assert resp.status_code == 422

    def test_unknown_org_returns_404(self, admin_client, mock_supabase):
        def _table(name):
            b = MockQueryBuilder()
            if name == "organizations":
                b.execute.return_value = MagicMock(data=None, count=0)
            return b

        mock_supabase.table.side_effect = _table
        resp = admin_client.put(
            f"/admin/orgs/{ORG_ID}/kind",
            json={"kind": "self_serve", "covered_by_user_id": self.COVERER_ID},
        )
        assert resp.status_code == 404

    @staticmethod
    def _basic_tier_dials_side(name):
        """team_dials_for_user's reads for a plain 'basic' coverer with a
        single team slot — shared by every self_serve-flip test below."""
        b = MockQueryBuilder()
        if name == "profiles":
            b.execute.return_value = MagicMock(data=[{"is_admin": False}], count=1)
        elif name == "subscriptions":
            b.execute.return_value = MagicMock(data=[{"tier": "basic"}], count=1)
        elif name == "tier_entitlements":
            b.execute.return_value = MagicMock(
                data=[{"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 1}],
                count=1,
            )
        return b

    @classmethod
    def _table_with_org_reads(cls, org_row: dict, covered_count: int, on_update=None):
        """The self_serve-flip path makes exactly 3 reads against
        'organizations' before its final update — the router's existence
        check, set_org_kind's archived/dissolved guard, and
        standing.count_covered_orgs — satisfied here by the SAME row + count
        (which columns were selected doesn't matter to the mock). The 4th
        call is the update; `on_update` (if given) receives the payload."""
        calls = {"n": 0}

        def _table(name):
            if name != "organizations":
                return cls._basic_tier_dials_side(name)
            b = MockQueryBuilder()
            calls["n"] += 1
            if calls["n"] <= 3:
                b.execute.return_value = MagicMock(data=[org_row], count=covered_count)
            else:
                original_update = b.update

                def _capture_update(payload, *a, **kw):
                    if on_update is not None:
                        on_update(payload)
                    return original_update(payload, *a, **kw)

                b.update = _capture_update
                b.execute.return_value = MagicMock(data=[{"id": ORG_ID}], count=1)
            return b

        return _table

    def test_coverer_without_slot_returns_402(self, admin_client, mock_supabase, monkeypatch):
        from orgs import authz as org_authz

        is_admin_calls = []
        monkeypatch.setattr(
            org_authz,
            "is_org_admin",
            lambda db, user_id, org_id: is_admin_calls.append((user_id, org_id)) or True,
        )

        # Already at the wall: covered_count=1 >= max_teams=1.
        mock_supabase.table.side_effect = self._table_with_org_reads({"id": ORG_ID}, covered_count=1)
        resp = admin_client.put(
            f"/admin/orgs/{ORG_ID}/kind",
            json={"kind": "self_serve", "covered_by_user_id": self.COVERER_ID},
        )
        assert resp.status_code == 402
        assert resp.json()["detail"]["upgradeRequired"] is True
        # is_org_admin must be checked BEFORE the slot — and with (coverer,
        # org) in that order, matching orgs.authz.is_org_admin's signature.
        assert is_admin_calls == [(self.COVERER_ID, ORG_ID)]

    def test_non_admin_coverer_returns_400(self, admin_client, mock_supabase, monkeypatch):
        """A coverer who isn't an active admin of THIS org must be rejected
        even before the slot check runs — never stub is_org_admin blindly."""
        from orgs import authz as org_authz

        monkeypatch.setattr(org_authz, "is_org_admin", lambda *a: False)

        mock_supabase.table.side_effect = self._table_with_org_reads({"id": ORG_ID}, covered_count=0)
        resp = admin_client.put(
            f"/admin/orgs/{ORG_ID}/kind",
            json={"kind": "self_serve", "covered_by_user_id": self.COVERER_ID},
        )
        assert resp.status_code == 400
        assert "active admin" in resp.json()["detail"]

    def test_archived_org_returns_400(self, admin_client, mock_supabase, monkeypatch):
        """Mirrors orgs.service.claim_coverage's guard — coverage must never
        be assigned onto a team that can't be reactivated this way."""
        from orgs import authz as org_authz

        monkeypatch.setattr(org_authz, "is_org_admin", lambda *a: True)

        mock_supabase.table.side_effect = self._table_with_org_reads(
            {"id": ORG_ID, "archived_at": "2026-07-01T00:00:00+00:00", "dissolved_at": None}, covered_count=0
        )
        resp = admin_client.put(
            f"/admin/orgs/{ORG_ID}/kind",
            json={"kind": "self_serve", "covered_by_user_id": self.COVERER_ID},
        )
        assert resp.status_code == 400
        assert "archived" in resp.json()["detail"].lower()

    def test_dissolved_org_returns_400(self, admin_client, mock_supabase, monkeypatch):
        from orgs import authz as org_authz

        monkeypatch.setattr(org_authz, "is_org_admin", lambda *a: True)

        mock_supabase.table.side_effect = self._table_with_org_reads(
            {"id": ORG_ID, "archived_at": None, "dissolved_at": "2026-07-01T00:00:00+00:00"}, covered_count=0
        )
        resp = admin_client.put(
            f"/admin/orgs/{ORG_ID}/kind",
            json={"kind": "self_serve", "covered_by_user_id": self.COVERER_ID},
        )
        assert resp.status_code == 400

    def test_self_serve_success_stamps_coverage_and_activates(self, admin_client, mock_supabase, monkeypatch):
        from orgs import authz as org_authz

        is_admin_calls = []
        monkeypatch.setattr(
            org_authz,
            "is_org_admin",
            lambda db, user_id, org_id: is_admin_calls.append((user_id, org_id)) or True,
        )
        captured = {}

        mock_supabase.table.side_effect = self._table_with_org_reads(
            {"id": ORG_ID}, covered_count=0, on_update=lambda payload: captured.update(payload=payload)
        )
        resp = admin_client.put(
            f"/admin/orgs/{ORG_ID}/kind",
            json={"kind": "self_serve", "covered_by_user_id": self.COVERER_ID},
        )
        assert resp.status_code == 200
        assert is_admin_calls == [(self.COVERER_ID, ORG_ID)]

        payload = captured["payload"]
        assert payload["kind"] == "self_serve"
        assert payload["covered_by"] == self.COVERER_ID
        assert payload["covered_at"]
        # create_org's invariant: the slot IS the activation — a flip must
        # land an org in the same state a fresh self-serve create would.
        assert payload["status"] == "active"
        # An enterprise dispersal surviving the flip would mint free monthly
        # credits into a self-serve pool forever after (the sweep grants
        # monthly_dispersal_credits to ANY org regardless of kind).
        assert payload["monthly_dispersal_credits"] == 0

    def test_flip_to_enterprise_clears_grace_started_at(self, admin_client, mock_supabase):
        """Standing (the sweep) ignores enterprise orgs entirely, so a stale
        grace-start timestamp from a prior self_serve stint must not linger
        and confuse a later flip back."""
        captured = {}
        org_calls = {"n": 0}

        def _table(name):
            b = MockQueryBuilder()
            if name == "organizations":
                org_calls["n"] += 1
                if org_calls["n"] == 1:
                    b.execute.return_value = MagicMock(data={"id": ORG_ID}, count=1)
                else:
                    original_update = b.update

                    def _capture_update(payload, *a, **kw):
                        captured["payload"] = payload
                        return original_update(payload, *a, **kw)

                    b.update = _capture_update
                    b.execute.return_value = MagicMock(data=[{"id": ORG_ID, "kind": "enterprise"}], count=1)
            return b

        mock_supabase.table.side_effect = _table
        resp = admin_client.put(f"/admin/orgs/{ORG_ID}/kind", json={"kind": "enterprise"})
        assert resp.status_code == 200
        assert captured["payload"] == {"kind": "enterprise", "grace_started_at": None}
