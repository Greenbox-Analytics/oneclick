"""Pending tester grants: POST unknown email, list, revoke (admin routes)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def admin_client(client, monkeypatch):
    import main
    from subscriptions.admin_auth import require_admin

    async def _pass():
        return "admin@example.com"

    main.app.dependency_overrides[require_admin] = _pass
    yield client
    main.app.dependency_overrides.pop(require_admin, None)


@pytest.fixture
def svc_mock(monkeypatch):
    """Fresh AdminService wired to a bare MagicMock supabase.

    AdminService takes TWO args (supabase, entitlements_service) — mirror
    test_admin_service.py's construction exactly.
    """
    import subscriptions.admin_router as ar
    from subscriptions.admin_service import AdminService
    from subscriptions.service import EntitlementsService

    sb = MagicMock()
    monkeypatch.setattr(ar, "_admin_service", AdminService(sb, EntitlementsService(sb)))
    return sb


class TestCreateTesterGrantPending:
    def test_unknown_email_creates_pending_row(self, admin_client, svc_mock):
        svc_mock.rpc.return_value.execute.return_value.data = []  # no user match
        # No existing pending row for this email (the claimed-row recovery
        # check in create_tester_grant's not-matched branch).
        svc_mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "  New.Tester@Example.COM ", "grant_duration_days": 30, "credits": 500},
        )
        assert resp.status_code == 200
        assert resp.json()["pending"] is True

        # search RPC called with generous limit (default 10 can push the exact
        # match out of the window)
        rpc_args = svc_mock.rpc.call_args
        assert rpc_args.args[0] == "admin_search_users_by_email"
        assert rpc_args.args[1]["p_limit"] == 100

        upsert_args = svc_mock.table.return_value.upsert.call_args
        row = upsert_args.args[0]
        assert row["email"] == "new.tester@example.com"  # lowercased + trimmed
        assert row["grant_duration_days"] == 30
        assert row["created_by"] == "admin@example.com"
        assert "claimed_at" not in row  # a re-POST must never clobber a claim
        assert "claimed_user_id" not in row

    def test_known_email_takes_active_path_with_derived_expiry(self, admin_client, svc_mock):
        svc_mock.rpc.return_value.execute.return_value.data = [
            {"id": "user-1", "email": "known@example.com", "created_at": "2026-01-01"}
        ]
        # wallet lookup inside grant_initial_tester_credits
        svc_mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {"id": "w-1"}
        ]
        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "known@example.com", "grant_duration_days": 30},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("pending") is not True
        expires = datetime.fromisoformat(body["expires_at"])
        delta = expires - datetime.now(UTC)
        assert timedelta(days=30) - timedelta(minutes=5) < delta < timedelta(days=30) + timedelta(minutes=5)

    def test_days_wins_over_legacy_expires_at(self, admin_client, svc_mock):
        """When both grant_duration_days and the legacy expires_at are sent,
        days wins — expires_at ends up ~now+days, not the legacy value."""
        svc_mock.rpc.return_value.execute.return_value.data = [
            {"id": "user-1", "email": "known@example.com", "created_at": "2026-01-01"}
        ]
        svc_mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {"id": "w-1"}
        ]
        resp = admin_client.post(
            "/admin/tester-grants",
            json={
                "email": "known@example.com",
                "grant_duration_days": 30,
                "expires_at": "2027-01-01T00:00:00+00:00",  # legacy value; days must win
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["expires_at"] != "2027-01-01T00:00:00+00:00"
        expires = datetime.fromisoformat(body["expires_at"])
        delta = expires - datetime.now(UTC)
        assert timedelta(days=30) - timedelta(minutes=5) < delta < timedelta(days=30) + timedelta(minutes=5)

    def test_known_email_clears_stale_pending_row(self, admin_client, svc_mock):
        svc_mock.rpc.return_value.execute.return_value.data = [
            {"id": "user-1", "email": "known@example.com", "created_at": "2026-01-01"}
        ]
        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "Known@example.com"},
        )
        assert resp.status_code == 200
        delete_chain = svc_mock.table.return_value.delete.return_value
        delete_chain.eq.assert_called_with("email", "known@example.com")
        delete_chain.eq.return_value.is_.assert_called_with("claimed_at", "null")


class TestClaimedRowRecovery:
    """create_tester_grant's not-matched branch, when a CLAIMED pending row
    already exists for the email (RPC false-negative from an email change or
    a >100-row search window, or the claimed user's account was deleted)."""

    def test_claimed_row_with_live_user_takes_active_path(self, admin_client, svc_mock):
        svc_mock.rpc.return_value.execute.return_value.data = []  # RPC finds nobody
        svc_mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"claimed_at": "2026-07-01T00:00:00+00:00", "claimed_user_id": "uid-claimed"}
        ]
        svc_mock.auth.admin.get_user_by_id.return_value = MagicMock(user=MagicMock(id="uid-claimed"))

        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "reclaimed@example.com", "grant_duration_days": 30},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("pending") is not True
        assert body["user_id"] == "uid-claimed"
        assert body["expires_at"] is not None

        # Exactly one upsert happened — the active tier_overrides grant — and
        # the pending row was NOT re-upserted (which would have clobbered a
        # claimed row's terms).
        assert svc_mock.table.return_value.upsert.call_count == 1
        upsert_payload = svc_mock.table.return_value.upsert.call_args.args[0]
        assert upsert_payload["user_id"] == "uid-claimed"

    def test_claimed_row_with_deleted_user_reopens_designation(self, admin_client, svc_mock):
        svc_mock.rpc.return_value.execute.return_value.data = []  # RPC finds nobody
        svc_mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"claimed_at": "2026-07-01T00:00:00+00:00", "claimed_user_id": "uid-gone"}
        ]
        svc_mock.auth.admin.get_user_by_id.side_effect = Exception("not found")

        resp = admin_client.post(
            "/admin/tester-grants",
            json={"email": "gone@example.com", "grant_duration_days": 14},
        )
        assert resp.status_code == 200
        assert resp.json()["pending"] is True

        upsert_args = svc_mock.table.return_value.upsert.call_args
        row = upsert_args.args[0]
        assert row["email"] == "gone@example.com"
        # ONLY this branch may re-open a claim by nulling these columns.
        assert row["claimed_at"] is None
        assert row["claimed_user_id"] is None


class TestListTesterGrantsPending:
    def test_list_returns_pending_only_when_no_active(self, admin_client, svc_mock):
        """GET /admin/tester-grants with zero active tier_overrides rows must
        still surface unclaimed pending designations (the early-return path)."""
        svc_mock.table.return_value.select.return_value.ilike.return_value.execute.return_value.data = []
        svc_mock.table.return_value.select.return_value.is_.return_value.execute.return_value.data = [
            {
                "email": "p1@example.com",
                "reason": "tester",
                "created_at": "2026-08-01T00:00:00+00:00",
                "grant_duration_days": 14,
            }
        ]
        resp = admin_client.get("/admin/tester-grants")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        row = body[0]
        assert row["user_id"] is None
        assert row["pending"] is True
        assert row["email"] == "p1@example.com"
        assert row["grant_duration_days"] == 14
        assert row["name"] is None

    def test_list_merges_active_and_pending(self, admin_client, svc_mock):
        svc_mock.table.return_value.select.return_value.ilike.return_value.execute.return_value.data = [
            {"user_id": "u1", "reason": "tester", "expires_at": None, "granted_at": "2026-08-01T00:00:00+00:00"}
        ]
        svc_mock.table.return_value.select.return_value.is_.return_value.execute.return_value.data = [
            {
                "email": "p2@example.com",
                "reason": "tester",
                "created_at": "2026-08-02T00:00:00+00:00",
                "grant_duration_days": None,
            }
        ]
        svc_mock.table.return_value.select.return_value.in_.return_value.execute.return_value.data = []
        svc_mock.auth.admin.get_user_by_id.return_value = MagicMock(user=MagicMock(email="active@example.com"))

        resp = admin_client.get("/admin/tester-grants")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2

        pending_rows = [r for r in body if r.get("pending") is True]
        assert len(pending_rows) == 1
        assert pending_rows[0]["user_id"] is None
        assert pending_rows[0]["email"] == "p2@example.com"

        active_rows = [r for r in body if not r.get("pending")]
        assert len(active_rows) == 1
        assert active_rows[0]["user_id"] == "u1"
        assert active_rows[0]["email"] == "active@example.com"


class TestPendingRevoke:
    def test_delete_pending_by_email(self, admin_client, svc_mock):
        resp = admin_client.delete("/admin/tester-grants/pending", params={"email": "X@Example.com"})
        assert resp.status_code == 204
        delete_chain = svc_mock.table.return_value.delete.return_value
        delete_chain.eq.assert_called_with("email", "x@example.com")
        # only unclaimed rows are deletable
        delete_chain.eq.return_value.is_.assert_called_with("claimed_at", "null")

    def test_non_admin_rejected(self, client):
        resp = client.delete("/admin/tester-grants/pending", params={"email": "a@b.com"})
        assert resp.status_code in (401, 403)
