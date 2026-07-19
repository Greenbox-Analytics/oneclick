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

        builders = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            builders[name] = b
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.clear_override(TEST_USER_ID)
        # Verify delete().eq("user_id", ...) was called with the correct user_id
        b = builders["tier_overrides"]
        b.delete.return_value.eq.assert_called_with("user_id", TEST_USER_ID)


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

    def test_list_users_with_search_calls_rpc(self):
        """When search is non-empty, list_users must call the
        admin_search_users_by_email RPC and NOT the paginated auth admin API.

        Regression guard for the original bug: the old impl fetched a single
        page of auth.admin.list_users() and filtered in Python, silently
        missing any matching user past the first page.
        """
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        sb = MagicMock()

        # RPC returns one matching row in the dict shape Supabase emits.
        rpc_builder = MagicMock()
        rpc_builder.execute.return_value = MagicMock(
            data=[
                {
                    "id": TEST_USER_ID,
                    "email": "match@example.com",
                    "created_at": "2026-05-01T00:00:00+00:00",
                },
            ]
        )
        sb.rpc.return_value = rpc_builder

        def _table(name):
            b = MockQueryBuilder()
            if name in ("subscriptions", "tier_overrides", "profiles"):
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb.table.side_effect = _table

        svc = AdminService(sb, EntitlementsService(sb))
        result = svc.list_users(search="match", page=1, per_page=25)

        # RPC called with the right name + params
        sb.rpc.assert_called_once_with(
            "admin_search_users_by_email",
            {"p_search": "match", "p_limit": 25},
        )
        # The paginated auth admin API must NOT have been called on the search path
        sb.auth.admin.list_users.assert_not_called()

        # Result is built from the RPC row
        assert len(result["users"]) == 1
        assert result["users"][0]["id"] == TEST_USER_ID
        assert result["users"][0]["email"] == "match@example.com"

    def test_list_users_without_search_uses_paginated_admin_api(self):
        """When search is empty, the existing auth.admin.list_users(page=, per_page=)
        path is used and the RPC is NOT called."""
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        sb = MagicMock()

        sb.auth.admin.list_users.return_value = [
            MagicMock(id=TEST_USER_ID, email="test@example.com", created_at="2026-05-01T00:00:00+00:00"),
        ]

        def _table(name):
            b = MockQueryBuilder()
            if name in ("subscriptions", "tier_overrides", "profiles"):
                b.execute.return_value = MagicMock(data=[], count=0)
            return b

        sb.table.side_effect = _table

        svc = AdminService(sb, EntitlementsService(sb))
        svc.list_users(search="", page=2, per_page=25)

        sb.auth.admin.list_users.assert_called_once_with(page=2, per_page=25)
        sb.rpc.assert_not_called()


class TestTesterGrants:
    def test_list_returns_only_tester_grants_active(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        # The DB-side LIKE filter returns only tester* rows; Python-side filter
        # removes expired ones. Simulate what the DB returns after LIKE 'tester%':
        # 2 rows matching reason prefix — 1 active, 1 expired.
        tester_rows = [
            {"user_id": "uid-1", "reason": "tester", "expires_at": None, "granted_at": "2026-05-01T00:00:00+00:00"},
            {
                "user_id": "uid-3",
                "reason": "tester-beta",
                "expires_at": "2020-01-01T00:00:00+00:00",
                "granted_at": "2019-12-01T00:00:00+00:00",
            },
        ]
        sb = _wire_supabase({"tier_overrides": tester_rows})
        svc = AdminService(sb, EntitlementsService(sb))

        result = svc.list_tester_grants()
        # Only the active tester row should survive the expiry filter
        assert len(result) == 1
        assert result[0]["user_id"] == "uid-1"

    def test_list_enriches_with_email_and_name(self):
        """list_tester_grants should populate email (auth.users) and name
        (profiles.full_name) on every active row. Rows without a matching
        profile row should still have email populated and name=None."""
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        # Two active tester rows for two distinct users.
        tester_rows = [
            {"user_id": "uid-1", "reason": "tester", "expires_at": None, "granted_at": "2026-05-01T00:00:00+00:00"},
            {
                "user_id": "uid-2",
                "reason": "tester-beta",
                "expires_at": None,
                "granted_at": "2026-05-02T00:00:00+00:00",
            },
        ]
        # Only uid-1 has a profile row with full_name.
        profile_rows = [{"id": "uid-1", "full_name": "Alice Tester"}]

        sb = MagicMock()

        email_by_id = {
            "uid-1": "alice@example.com",
            "uid-2": "bob@example.com",
        }

        def _get_user_by_id(uid):
            user_obj = MagicMock()
            user_obj.email = email_by_id.get(uid, "")
            user_obj.id = uid
            resp = MagicMock()
            resp.user = user_obj
            return resp

        sb.auth.admin.get_user_by_id.side_effect = _get_user_by_id

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                b.execute.return_value = MagicMock(data=tester_rows, count=len(tester_rows))
            elif name == "profiles":
                b.execute.return_value = MagicMock(data=profile_rows, count=len(profile_rows))
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        result = svc.list_tester_grants()
        by_uid = {row["user_id"]: row for row in result}
        assert len(result) == 2
        assert by_uid["uid-1"]["email"] == "alice@example.com"
        assert by_uid["uid-1"]["name"] == "Alice Tester"
        assert by_uid["uid-2"]["email"] == "bob@example.com"
        assert by_uid["uid-2"]["name"] is None

    def test_create_full_pro_override(self):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()
        sb.auth.admin.list_users.return_value = [
            MagicMock(id="uid-10", email="tester@example.com"),
        ]

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original = b.upsert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    captured["on_conflict"] = kw.get("on_conflict")
                    return original(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        result = svc.create_tester_grant(email="tester@example.com", expires_at="2027-01-01T00:00:00+00:00")

        p = captured["payload"]
        assert p["max_artists"] == -1
        assert p["max_projects"] == -1
        assert p["max_tasks"] == -1
        assert p["max_storage_bytes"] == -1
        assert p["max_split_sheets_per_month"] == -1
        assert p["max_oneclick_runs_per_month"] == -1
        assert p["zoe_enabled"] is True
        assert p["oneclick_enabled"] is True
        assert p["registry_enabled"] is True
        assert p["integrations_allowed"] == ["google_drive", "slack"]
        assert p["reason"] == "tester"
        assert p["expires_at"] == "2027-01-01T00:00:00+00:00"
        assert captured["on_conflict"] == "user_id"

        assert result["user_id"] == "uid-10"
        assert result["email"] == "tester@example.com"

    def test_create_tester_grant_grants_initial_credits(self):
        """create_tester_grant should delegate initial credit allocation to
        grant_initial_tester_credits, passing (supabase, user_id, credits)."""
        from unittest.mock import patch

        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        sb.auth.admin.list_users.return_value = [
            MagicMock(id="uid-10", email="tester@example.com"),
        ]

        def _table(name):
            return MockQueryBuilder()

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        with patch("subscriptions.admin_service.grant_initial_tester_credits") as gitc:
            svc.create_tester_grant(email="tester@example.com", credits=750)

        gitc.assert_called_once()
        args = gitc.call_args[0]
        assert args[0] is sb
        assert args[1] == "uid-10"
        assert args[2] == 750  # (supabase, user_id, credits)

    def test_create_tester_grant_normalizes_reason(self):
        """`reason` must always satisfy LIKE 'tester%' so list_tester_grants
        returns the row. Empty → 'tester'. Already-prefixed (case-insensitive)
        → leading 'tester' forced to lowercase, suffix preserved. Custom →
        wrapped as 'tester (whatever)'."""
        from subscriptions.admin_service import _normalize_tester_reason

        assert _normalize_tester_reason(None) == "tester"
        assert _normalize_tester_reason("") == "tester"
        assert _normalize_tester_reason("   ") == "tester"
        assert _normalize_tester_reason("tester") == "tester"
        assert _normalize_tester_reason("tester_qa") == "tester_qa"
        # Leading 'tester' prefix is forced to lowercase so SQL LIKE 'tester%'
        # matches; the rest of the string is preserved.
        assert _normalize_tester_reason("Tester special") == "tester special"
        assert _normalize_tester_reason("beta access") == "tester (beta access)"
        assert _normalize_tester_reason("Internal QA") == "tester (Internal QA)"

    def test_create_tester_grant_persists_normalized_reason(self):
        """End-to-end: a custom reason like 'beta access' should be wrapped to
        'tester (beta access)' in both the upserted payload AND the returned dict,
        so list_tester_grants (which filters by LIKE 'tester%') sees the row."""
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        captured = {}
        sb = MagicMock()
        sb.auth.admin.list_users.return_value = [
            MagicMock(id="uid-11", email="custom@example.com"),
        ]

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

        result = svc.create_tester_grant(email="custom@example.com", reason="beta access")

        assert captured["payload"]["reason"] == "tester (beta access)"
        assert result["reason"] == "tester (beta access)"

    def test_create_user_not_found_raises(self):
        import pytest

        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        sb = MagicMock()
        sb.auth.admin.list_users.return_value = []
        svc = AdminService(sb, EntitlementsService(sb))

        with pytest.raises(ValueError, match="User not found"):
            svc.create_tester_grant(email="nobody@example.com")

    def test_revoke_writes_sticky_marker_not_delete(self):
        """revoke_tester_grant should UPSERT with reason='tester_revoked' and null caps,
        NOT delete the row. This prevents bootstrap-tester from re-creating the grant
        on the next sign-in for users in TESTER_EMAILS."""
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService
        from tests.conftest import TEST_USER_ID

        captured = {}
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original_upsert = b.upsert
                original_delete = b.delete

                def _capture_upsert(payload, *a, **kw):
                    captured["upsert_payload"] = payload
                    captured["upsert_on_conflict"] = kw.get("on_conflict")
                    return original_upsert(payload, *a, **kw)

                def _capture_delete():
                    captured["delete_called"] = True
                    return original_delete()

                b.upsert = _capture_upsert
                b.delete = _capture_delete
            return b

        sb.table.side_effect = _table
        svc = AdminService(sb, EntitlementsService(sb))

        svc.revoke_tester_grant(TEST_USER_ID)

        # Must have done an upsert, not a delete
        assert "upsert_payload" in captured
        assert captured.get("delete_called") is not True
        assert captured["upsert_payload"]["reason"] == "tester_revoked"
        assert captured["upsert_payload"]["user_id"] == TEST_USER_ID
        assert captured["upsert_payload"]["max_artists"] is None
        assert captured["upsert_payload"]["max_oneclick_runs_per_month"] is None
        assert captured["upsert_payload"]["zoe_enabled"] is None
        assert captured["upsert_on_conflict"] == "user_id"

    def test_list_excludes_revoked_rows(self):
        """list_tester_grants should NOT return rows where reason='tester_revoked'."""
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        rows = [
            {"user_id": "u1", "reason": "tester", "expires_at": None, "granted_at": "2026-01-01T00:00:00Z"},
            {"user_id": "u2", "reason": "tester_env", "expires_at": None, "granted_at": "2026-01-01T00:00:00Z"},
            {"user_id": "u3", "reason": "tester_revoked", "expires_at": None, "granted_at": "2026-01-01T00:00:00Z"},
        ]
        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=rows if name == "tier_overrides" else [])
            return b

        sb.table.side_effect = _table

        # Skip the email/name enrichment — irrelevant to this test
        sb.auth.admin.get_user_by_id.return_value = MagicMock(user=MagicMock(email="x@y.com", id="x"))

        svc = AdminService(sb, EntitlementsService(sb))
        result = svc.list_tester_grants()
        user_ids = [r["user_id"] for r in result]
        assert "u1" in user_ids
        assert "u2" in user_ids
        assert "u3" not in user_ids  # revoked row excluded

    def test_list_uses_ilike_so_capital_T_rows_surface(self):
        """A row with reason='Tester' (legacy / weird casing) should still
        appear in the active tester list. This is the bug that hid Yash's grant."""
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        rows = [
            {"user_id": "u1", "reason": "Tester", "expires_at": None, "granted_at": "2026-01-01T00:00:00Z"},
            {"user_id": "u2", "reason": "tester_env", "expires_at": None, "granted_at": "2026-01-01T00:00:00Z"},
        ]
        sb = MagicMock()
        captured_filter = {}

        def _table(name):
            b = MockQueryBuilder()
            original_ilike = b.ilike

            def _capture_ilike(col, pat):
                captured_filter["col"] = col
                captured_filter["pat"] = pat
                return original_ilike(col, pat)

            b.ilike = _capture_ilike
            b.execute.return_value = MagicMock(data=rows if name == "tier_overrides" else [])
            return b

        sb.table.side_effect = _table
        sb.auth.admin.get_user_by_id.return_value = MagicMock(user=MagicMock(email="x@y.com", id="x"))

        svc = AdminService(sb, EntitlementsService(sb))
        result = svc.list_tester_grants()
        assert captured_filter == {"col": "reason", "pat": "tester%"}
        user_ids = [r["user_id"] for r in result]
        assert "u1" in user_ids  # 'Tester' must appear via ILIKE
        assert "u2" in user_ids


class TestNormalizeTesterReason:
    def test_empty_input_returns_bare_tester(self):
        from subscriptions.admin_service import _normalize_tester_reason

        assert _normalize_tester_reason(None) == "tester"
        assert _normalize_tester_reason("") == "tester"
        assert _normalize_tester_reason("   ") == "tester"

    def test_lowercase_prefix_unchanged(self):
        from subscriptions.admin_service import _normalize_tester_reason

        assert _normalize_tester_reason("tester") == "tester"
        assert _normalize_tester_reason("tester_qa") == "tester_qa"
        assert _normalize_tester_reason("tester (Yash)") == "tester (Yash)"

    def test_uppercase_prefix_forced_to_lowercase(self):
        from subscriptions.admin_service import _normalize_tester_reason

        # Capital T case-insensitively matches, but we force the prefix to
        # lowercase so SQL LIKE 'tester%' filter still finds it. Suffix is
        # preserved with original casing.
        assert _normalize_tester_reason("Tester") == "tester"
        assert _normalize_tester_reason("Tester QA") == "tester QA"
        assert _normalize_tester_reason("TESTER_qa") == "tester_qa"

    def test_non_tester_input_wrapped(self):
        from subscriptions.admin_service import _normalize_tester_reason

        assert _normalize_tester_reason("beta access") == "tester (beta access)"
        assert _normalize_tester_reason("Internal QA") == "tester (Internal QA)"


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
            "max_oneclick_runs_per_month": 1,
            "zoe_enabled": False,
            "oneclick_enabled": True,
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


class TestPromoteDemote:
    """profiles.is_admin toggle via AdminService.

    Note: mock_supabase.table.side_effect is _default_table_side_effect which
    returns a FRESH MockQueryBuilder per call. To assert on the builder used
    inside the service call, we register a custom side_effect that returns a
    shared, pre-spied builder for the 'profiles' table.
    """

    def _profiles_builder_with_spy(self, mock_supabase):
        """Install a shared MockQueryBuilder for the 'profiles' table with
        update() spied. Returns (builder, update_spy)."""
        builder = MockQueryBuilder()
        update_spy = MagicMock(return_value=builder)
        builder.update = update_spy

        original_side = mock_supabase.table.side_effect

        def _side(name):
            if name == "profiles":
                return builder
            return original_side(name)

        mock_supabase.table.side_effect = _side
        return builder, update_spy

    def test_promote_user_calls_update_with_is_admin_true(self, mock_supabase):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        builder, update_spy = self._profiles_builder_with_spy(mock_supabase)
        target_uid = "44444444-4444-4444-4444-444444444444"

        svc = AdminService(mock_supabase, EntitlementsService(mock_supabase))
        svc.promote_user(target_uid)

        update_spy.assert_called_once_with({"is_admin": True})
        assert builder.execute.called

    def test_demote_user_calls_update_with_is_admin_false(self, mock_supabase):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        builder, update_spy = self._profiles_builder_with_spy(mock_supabase)
        target_uid = "55555555-5555-5555-5555-555555555555"

        svc = AdminService(mock_supabase, EntitlementsService(mock_supabase))
        svc.demote_user(target_uid)

        update_spy.assert_called_once_with({"is_admin": False})
        assert builder.execute.called

    def test_get_email_for_user_id_returns_email_when_found(self, mock_supabase):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        mock_user = MagicMock()
        mock_user.email = "target@example.com"
        mock_supabase.auth.admin.get_user_by_id.return_value = MagicMock(user=mock_user)

        svc = AdminService(mock_supabase, EntitlementsService(mock_supabase))
        result = svc.get_email_for_user_id("66666666-6666-6666-6666-666666666666")
        assert result == "target@example.com"

    def test_get_email_for_user_id_returns_none_on_lookup_failure(self, mock_supabase):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        mock_supabase.auth.admin.get_user_by_id.side_effect = Exception("404")
        svc = AdminService(mock_supabase, EntitlementsService(mock_supabase))
        assert svc.get_email_for_user_id("nope") is None


class TestAdminFlagsInResponses:
    def test_list_users_includes_is_admin_and_is_env_admin(self, mock_supabase, monkeypatch):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        monkeypatch.setenv("ADMIN_EMAILS", "env@example.com")

        u1 = MagicMock(id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", email="env@example.com", created_at="2026-01-01")
        u2 = MagicMock(id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", email="db@example.com", created_at="2026-01-02")
        mock_supabase.auth.admin.list_users.return_value = [u1, u2]

        def _table(name):
            b = MockQueryBuilder()
            if name == "profiles":
                b.execute.return_value = MagicMock(
                    data=[
                        {"id": str(u1.id), "is_admin": False},
                        {"id": str(u2.id), "is_admin": True},
                    ]
                )
            elif name in ("subscriptions", "tier_overrides"):
                b.execute.return_value = MagicMock(data=[])
            return b

        original_side = mock_supabase.table.side_effect

        def _side(name):
            if name in ("profiles", "subscriptions", "tier_overrides"):
                return _table(name)
            return original_side(name)

        mock_supabase.table.side_effect = _side
        svc = AdminService(mock_supabase, EntitlementsService(mock_supabase))
        result = svc.list_users()

        rows = {r["id"]: r for r in result["users"]}
        assert rows[str(u1.id)]["is_env_admin"] is True
        assert rows[str(u1.id)]["is_admin"] is False
        assert rows[str(u2.id)]["is_env_admin"] is False
        assert rows[str(u2.id)]["is_admin"] is True

    def test_get_user_detail_includes_admin_flags(self, mock_supabase, monkeypatch):
        from subscriptions.admin_service import AdminService
        from subscriptions.service import EntitlementsService

        monkeypatch.setenv("ADMIN_EMAILS", "env@example.com")
        uid = "cccccccc-cccc-cccc-cccc-cccccccccccc"

        mock_user = MagicMock()
        mock_user.id = uid
        mock_user.email = "env@example.com"
        mock_user.created_at = "2026-01-01"
        mock_supabase.auth.admin.get_user_by_id.return_value = MagicMock(user=mock_user)

        def _table(name):
            b = MockQueryBuilder()
            if name == "profiles":
                b.execute.return_value = MagicMock(data=[{"is_admin": False}])
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(data=[])
            return b

        original_side = mock_supabase.table.side_effect

        def _side(name):
            if name in ("profiles", "tier_overrides"):
                return _table(name)
            return original_side(name)

        mock_supabase.table.side_effect = _side
        svc = AdminService(mock_supabase, EntitlementsService(mock_supabase))
        result = svc.get_user_detail(uid)

        assert result["user"]["is_env_admin"] is True
        assert result["user"]["is_admin"] is False
