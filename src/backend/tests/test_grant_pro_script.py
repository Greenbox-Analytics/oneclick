"""Tests for the grant_pro CLI admin script."""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

USER_EMAIL = "tech@example.com"


def _wire_user_lookup(supabase_mock, user_id=TEST_USER_ID, email=USER_EMAIL, page_size=50):
    """Single-page user lookup: returns one user on page 1, empty on page 2+."""
    pages = {1: [MagicMock(id=user_id, email=email)], 2: []}

    def _list_users(page=1, per_page=page_size):
        return pages.get(page, [])

    supabase_mock.auth.admin.list_users.side_effect = _list_users


class TestGrantCommand:
    def test_grant_calls_subscriptions_upsert_with_pro(self, monkeypatch):
        from scripts import grant_pro

        sb = MagicMock()
        _wire_user_lookup(sb)
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                original_upsert = b.upsert

                def _capture(payload, *a, **kw):
                    captured["upsert_payload"] = payload
                    return original_upsert(payload, *a, **kw)

                b.upsert = _capture
                b.execute.return_value = MagicMock(data=[{"user_id": TEST_USER_ID, "tier": "pro"}], count=1)
            return b

        sb.table.side_effect = _table
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        exit_code = grant_pro.main(["grant", USER_EMAIL])
        assert exit_code == 0
        assert captured["upsert_payload"]["tier"] == "pro"
        assert captured["upsert_payload"]["user_id"] == TEST_USER_ID

    def test_grant_unknown_email_exits_nonzero(self, monkeypatch, capsys):
        from scripts import grant_pro

        sb = MagicMock()
        sb.auth.admin.list_users.return_value = []
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        exit_code = grant_pro.main(["grant", "noone@nowhere.com"])
        assert exit_code != 0
        out = capsys.readouterr().err
        assert "not found" in out.lower() or "no user" in out.lower()


class TestRevokeCommand:
    def test_revoke_sets_tier_free(self, monkeypatch):
        from scripts import grant_pro

        sb = MagicMock()
        _wire_user_lookup(sb)
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                original_upsert = b.upsert

                def _capture(payload, *a, **kw):
                    captured["upsert_payload"] = payload
                    return original_upsert(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        exit_code = grant_pro.main(["revoke", USER_EMAIL])
        assert exit_code == 0
        assert captured["upsert_payload"]["tier"] == "free"


class TestOverrideCommand:
    def test_override_only_supplied_fields(self, monkeypatch):
        from scripts import grant_pro

        sb = MagicMock()
        _wire_user_lookup(sb)
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original_upsert = b.upsert

                def _capture(payload, *a, **kw):
                    captured["upsert_payload"] = payload
                    return original_upsert(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        exit_code = grant_pro.main(
            [
                "override",
                USER_EMAIL,
                "--max-artists",
                "10",
                "--zoe-enabled",
                "--reason",
                "Beta tester",
            ]
        )

        assert exit_code == 0
        payload = captured["upsert_payload"]
        assert payload["user_id"] == TEST_USER_ID
        assert payload["max_artists"] == 10
        assert payload["zoe_enabled"] is True
        assert payload["reason"] == "Beta tester"
        # Fields NOT supplied should be absent
        assert "max_projects" not in payload

    def test_override_does_not_write_granted_by(self, monkeypatch):
        """granted_by column is removed from schema; CLI must not include it."""
        from scripts import grant_pro

        sb = MagicMock()
        _wire_user_lookup(sb)
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original_upsert = b.upsert

                def _capture(payload, *a, **kw):
                    captured["upsert_payload"] = payload
                    return original_upsert(payload, *a, **kw)

                b.upsert = _capture
            return b

        sb.table.side_effect = _table
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        grant_pro.main(["override", USER_EMAIL, "--max-artists", "10"])
        assert "granted_by" not in captured["upsert_payload"]


class TestClearOverrideCommand:
    def test_clear_override_deletes_row(self, monkeypatch):
        from scripts import grant_pro

        sb = MagicMock()
        _wire_user_lookup(sb)
        deleted_for = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "tier_overrides":
                original_eq = b.eq

                def _capture_eq(field, value):
                    if field == "user_id":
                        deleted_for["user_id"] = value
                    return original_eq(field, value)

                b.eq = _capture_eq
            return b

        sb.table.side_effect = _table
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        exit_code = grant_pro.main(["clear-override", USER_EMAIL])
        assert exit_code == 0
        assert deleted_for["user_id"] == TEST_USER_ID


class TestListCommand:
    def test_list_outputs_table(self, monkeypatch, capsys):
        from scripts import grant_pro

        sb = MagicMock()

        def _table(name):
            b = MockQueryBuilder()
            if name == "subscriptions":
                b.execute.return_value = MagicMock(
                    data=[{"user_id": TEST_USER_ID, "tier": "pro", "status": "active"}],
                    count=1,
                )
            elif name == "tier_overrides":
                b.execute.return_value = MagicMock(
                    data=[{"user_id": TEST_USER_ID, "zoe_enabled": True, "reason": "Beta"}],
                    count=1,
                )
            return b

        sb.table.side_effect = _table
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        exit_code = grant_pro.main(["list"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert TEST_USER_ID in out or "pro" in out.lower()


class TestPagination:
    def test_resolve_user_id_paginates_past_first_page(self, monkeypatch):
        """User on page 2 (past default 50) should still be found."""
        from scripts import grant_pro

        sb = MagicMock()
        page1_users = [MagicMock(id=f"id-{i}", email=f"u{i}@x.com") for i in range(50)]
        page2_users = [MagicMock(id=TEST_USER_ID, email=USER_EMAIL)]

        def _list_users(page=1, per_page=50):
            if page == 1:
                return page1_users
            if page == 2:
                return page2_users
            return []

        sb.auth.admin.list_users.side_effect = _list_users
        monkeypatch.setattr(grant_pro, "_get_supabase", lambda: sb)

        resolved = grant_pro._resolve_user_id(sb, USER_EMAIL)
        assert resolved == TEST_USER_ID
        # Confirm pagination loop fired more than once
        assert sb.auth.admin.list_users.call_count >= 2
