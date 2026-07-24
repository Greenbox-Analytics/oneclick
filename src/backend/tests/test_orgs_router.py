"""Endpoint contract tests for the orgs router (uses the shared `client`
fixture). Mirrors tests/test_teams_router.py's idioms.

LICENSING_ENABLED is OFF by default in tests (nothing sets it); this module
enables it via an autouse fixture so most tests exercise the real routes,
and TestFlagGate explicitly turns it back off/false to prove the router-level
gate makes the entire surface disappear."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from orgs import service
from tests.conftest import MockQueryBuilder

ORG_ID = "20000000-0000-0000-0000-000000000001"
MEMBER_ID = "40000000-0000-0000-0000-000000000001"
INVITE_ID = "60000000-0000-0000-0000-000000000001"
TOKEN = "30000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def _licensing_on_by_default(monkeypatch):
    monkeypatch.setenv("LICENSING_ENABLED", "true")


class TestFlagGate:
    """Every /orgs/* route 404s when LICENSING_ENABLED is off — the whole
    surface is a no-op for rollback, independent of auth or service state."""

    ROUTES = [
        ("GET", "/orgs", None),
        ("POST", "/orgs", {"name": "Acme"}),
        ("GET", f"/orgs/{ORG_ID}", None),
        ("PUT", f"/orgs/{ORG_ID}", {"name": "Acme"}),
        ("POST", f"/orgs/{ORG_ID}/archive", None),
        ("GET", f"/orgs/{ORG_ID}/usage", None),
        ("POST", f"/orgs/invites/{TOKEN}/accept", None),
        ("POST", f"/orgs/invites/{TOKEN}/decline", None),
        ("PUT", f"/orgs/{ORG_ID}/members/{MEMBER_ID}/role", {"role": "admin"}),
        ("POST", f"/orgs/{ORG_ID}/members/{MEMBER_ID}/suspend", None),
        ("DELETE", f"/orgs/{ORG_ID}/members/{MEMBER_ID}", None),
        ("POST", f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reactivate", None),
        ("POST", f"/orgs/{ORG_ID}/invites", {"email": "a@b.com", "role": "member"}),
        ("GET", f"/orgs/{ORG_ID}/invites", None),
        ("DELETE", f"/orgs/{ORG_ID}/invites/{INVITE_ID}", None),
    ]

    @pytest.mark.parametrize("method,path,body", ROUTES)
    def test_route_404_when_flag_unset(self, method, path, body, client, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        kwargs = {"json": body} if body is not None else {}
        resp = client.request(method, path, **kwargs)
        assert resp.status_code == 404

    @pytest.mark.parametrize("method,path,body", ROUTES)
    def test_route_404_when_flag_explicitly_false(self, method, path, body, client, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "false")
        kwargs = {"json": body} if body is not None else {}
        resp = client.request(method, path, **kwargs)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /orgs
# ---------------------------------------------------------------------------


def test_create_org_ok(client):
    with patch("orgs.router.service.create_org", new=AsyncMock(return_value={"id": "o1", "my_role": "admin"})):
        resp = client.post("/orgs", json={"name": "Acme"})
    assert resp.status_code == 200
    assert resp.json()["id"] == "o1"
    assert resp.json()["my_role"] == "admin"


def test_create_org_fires_org_created_analytics(client):
    with (
        patch("orgs.router.service.create_org", new=AsyncMock(return_value={"id": "o1"})),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post("/orgs", json={"name": "Acme"})
    assert resp.status_code == 200
    mock_capture.assert_called_once()
    args = mock_capture.call_args.args
    assert args[1] == "org_created"


def test_create_org_rejects_empty_name(client):
    resp = client.post("/orgs", json={"name": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /orgs
# ---------------------------------------------------------------------------


def test_list_orgs_ok(client):
    with patch("orgs.router.service.list_my_orgs", new=AsyncMock(return_value=[{"id": "o1", "my_role": "admin"}])):
        resp = client.get("/orgs")
    assert resp.status_code == 200
    assert resp.json() == {"organizations": [{"id": "o1", "my_role": "admin"}]}


# ---------------------------------------------------------------------------
# GET /orgs/{org_id}
# ---------------------------------------------------------------------------


def test_get_org_ok(client):
    with patch("orgs.router.service.get_org", new=AsyncMock(return_value={"id": ORG_ID, "my_role": "admin"})):
        resp = client.get(f"/orgs/{ORG_ID}")
    assert resp.status_code == 200
    assert resp.json()["id"] == ORG_ID


def test_get_org_not_member_404(client):
    """authz.require_member raises HTTPException(404) directly from inside
    orgs.service.get_org; this asserts the router surfaces it unmodified."""
    with patch(
        "orgs.router.service.get_org",
        new=AsyncMock(side_effect=HTTPException(status_code=404, detail="Organization not found")),
    ):
        resp = client.get(f"/orgs/{ORG_ID}")
    assert resp.status_code == 404


def test_get_org_missing_row_maps_value_error_to_404(client):
    with patch("orgs.router.service.get_org", new=AsyncMock(side_effect=ValueError("Organization not found"))):
        resp = client.get(f"/orgs/{ORG_ID}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /orgs/{org_id}
# ---------------------------------------------------------------------------


def test_update_org_ok_and_forwards_only_set_fields(client):
    with patch(
        "orgs.router.service.update_org", new=AsyncMock(return_value={"id": ORG_ID, "name": "New"})
    ) as mock_update:
        resp = client.put(f"/orgs/{ORG_ID}", json={"name": "New"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"
    # default_seat_allowance was NOT in the request body → must not be forwarded.
    forwarded_fields = mock_update.call_args.args[-1]
    assert forwarded_fields == {"name": "New"}


def test_update_org_forwards_explicit_null_for_default_seat_allowance(client):
    with patch("orgs.router.service.update_org", new=AsyncMock(return_value={"id": ORG_ID})) as mock_update:
        resp = client.put(f"/orgs/{ORG_ID}", json={"default_seat_allowance": None})
    assert resp.status_code == 200
    forwarded_fields = mock_update.call_args.args[-1]
    assert forwarded_fields == {"default_seat_allowance": None}


def test_update_org_denied_for_non_admin_403(client):
    with patch(
        "orgs.router.service.update_org",
        new=AsyncMock(side_effect=HTTPException(status_code=403, detail="Admin access required")),
    ):
        resp = client.put(f"/orgs/{ORG_ID}", json={"name": "New"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /orgs/{org_id}/archive
# ---------------------------------------------------------------------------


def test_archive_org_ok(client):
    with patch("orgs.router.service.archive_org", new=AsyncMock(return_value={"id": ORG_ID, "archived_at": "now"})):
        resp = client.post(f"/orgs/{ORG_ID}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived_at"] == "now"


def test_archive_org_409_when_seat_balance_nonzero(client):
    from orgs.service import SeatBalanceNotZeroError

    with patch(
        "orgs.router.service.archive_org",
        new=AsyncMock(side_effect=SeatBalanceNotZeroError("Reclaim all seat credits first")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/archive")
    assert resp.status_code == 409


def test_archive_org_denied_for_non_admin_403(client):
    with patch(
        "orgs.router.service.archive_org",
        new=AsyncMock(side_effect=HTTPException(status_code=403, detail="Admin access required")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/archive")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /orgs/{org_id}/usage (Task 7 — router wiring only; seat-math coverage
# lives in TestGetOrgUsageService below, which calls orgs.service.get_org_usage
# directly against a table-mocked db).
# ---------------------------------------------------------------------------


def test_get_org_usage_ok(client):
    payload = {
        "poolBalance": 4000,
        "cumulativePurchased": 5000,
        "seats": [{"orgMemberId": MEMBER_ID, "seatBalance": 300}],
    }
    with patch("orgs.router.service.get_org_usage", new=AsyncMock(return_value=payload)):
        resp = client.get(f"/orgs/{ORG_ID}/usage")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_get_org_usage_denied_for_non_admin_403(client):
    with patch(
        "orgs.router.service.get_org_usage",
        new=AsyncMock(side_effect=HTTPException(status_code=403, detail="Admin access required")),
    ):
        resp = client.get(f"/orgs/{ORG_ID}/usage")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /orgs/invites/{token}/accept, /decline
# ---------------------------------------------------------------------------


def test_accept_org_invite_ok_and_fires_analytics(client):
    with (
        patch("orgs.router.service.accept_invite", new=AsyncMock(return_value={"type": "accepted", "org_id": ORG_ID})),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/invites/{TOKEN}/accept")
    assert resp.status_code == 200
    assert resp.json() == {"type": "accepted", "org_id": ORG_ID}
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "org_license_accepted"


def test_accept_org_invite_already_accepted_replay_does_not_fire_analytics(client):
    """Duplicate-gating discipline: an 'already_accepted' replay must NOT
    re-fire org_license_accepted."""
    with (
        patch(
            "orgs.router.service.accept_invite",
            new=AsyncMock(return_value={"type": "already_accepted", "org_id": ORG_ID}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/invites/{TOKEN}/accept")
    assert resp.status_code == 200
    mock_capture.assert_not_called()


def test_accept_org_invite_email_mismatch_403(client):
    with patch(
        "orgs.router.service.accept_invite",
        new=AsyncMock(side_effect=PermissionError("This invite was sent to a different email")),
    ):
        resp = client.post(f"/orgs/invites/{TOKEN}/accept")
    assert resp.status_code == 403


def test_accept_org_invite_expired_410(client):
    from orgs.service import InviteInvalidError

    with patch("orgs.router.service.accept_invite", new=AsyncMock(side_effect=InviteInvalidError("expired"))):
        resp = client.post(f"/orgs/invites/{TOKEN}/accept")
    assert resp.status_code == 410


def test_accept_org_invite_not_found_404(client):
    with patch("orgs.router.service.accept_invite", new=AsyncMock(side_effect=ValueError("Invite not found"))):
        resp = client.post(f"/orgs/invites/{TOKEN}/accept")
    assert resp.status_code == 404


def test_decline_org_invite_ok(client):
    with patch(
        "orgs.router.service.decline_invite", new=AsyncMock(return_value={"type": "declined", "org_id": ORG_ID})
    ):
        resp = client.post(f"/orgs/invites/{TOKEN}/decline")
    assert resp.status_code == 200
    assert resp.json() == {"type": "declined", "org_id": ORG_ID}


def test_decline_org_invite_not_found_404(client):
    with patch("orgs.router.service.decline_invite", new=AsyncMock(side_effect=ValueError("Invite not found"))):
        resp = client.post(f"/orgs/invites/{TOKEN}/decline")
    assert resp.status_code == 404


def test_decline_org_invite_email_mismatch_403(client):
    with patch(
        "orgs.router.service.decline_invite",
        new=AsyncMock(side_effect=PermissionError("This invite was sent to a different email")),
    ):
        resp = client.post(f"/orgs/invites/{TOKEN}/decline")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST/GET/DELETE /orgs/{org_id}/invites
# ---------------------------------------------------------------------------


def test_invite_member_ok_schedules_email_and_fires_analytics(client):
    """The background email task is patched out (it opens its own
    create_client to the REAL Supabase URL/key from env, which isn't set in
    tests) — this asserts scheduling happened with the right shape, not the
    email content itself (that's emails.py's own concern)."""
    with (
        patch(
            "orgs.router.service.invite_member",
            new=AsyncMock(return_value={"type": "invited", "invite": {"id": "i1"}, "notify_user_id": None}),
        ),
        patch("orgs.router._send_org_invite_email_background") as mock_bg,
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{ORG_ID}/invites", json={"email": "a@b.com", "role": "member"})
    assert resp.status_code == 200
    assert resp.json()["type"] == "invited"
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "org_license_invited"
    mock_bg.assert_called_once()
    assert mock_bg.call_args.kwargs["existing_user"] is False
    assert mock_bg.call_args.kwargs["org_id"] == ORG_ID
    assert mock_bg.call_args.kwargs["email"] == "a@b.com"


def test_invite_member_existing_user_schedules_existing_user_email(client):
    with (
        patch(
            "orgs.router.service.invite_member",
            new=AsyncMock(return_value={"type": "invited", "invite": {"id": "i1"}, "notify_user_id": "u-existing"}),
        ),
        patch("orgs.router._send_org_invite_email_background") as mock_bg,
        patch("orgs.router.analytics_capture"),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/invites", json={"email": "existing@b.com", "role": "member"})
    assert resp.status_code == 200
    assert mock_bg.call_args.kwargs["existing_user"] is True


def test_invite_member_duplicate_409(client):
    from orgs.service import DuplicateInviteError

    with patch("orgs.router.service.invite_member", new=AsyncMock(side_effect=DuplicateInviteError("dup"))):
        resp = client.post(f"/orgs/{ORG_ID}/invites", json={"email": "a@b.com", "role": "member"})
    assert resp.status_code == 409


def test_invite_member_invalid_role_400(client):
    with patch("orgs.router.service.invite_member", new=AsyncMock(side_effect=ValueError("Invalid role"))):
        resp = client.post(f"/orgs/{ORG_ID}/invites", json={"email": "a@b.com", "role": "owner"})
    assert resp.status_code == 400


def test_invite_member_rejects_malformed_email_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/invites", json={"email": "not-an-email", "role": "member"})
    assert resp.status_code == 422


def test_list_invites_ok(client):
    with patch("orgs.router.service.get_pending_invites", new=AsyncMock(return_value=[{"id": "i1"}])):
        resp = client.get(f"/orgs/{ORG_ID}/invites")
    assert resp.status_code == 200
    assert resp.json() == {"invites": [{"id": "i1"}]}


def test_list_invites_denied_for_non_admin_403(client):
    with patch(
        "orgs.router.service.get_pending_invites",
        new=AsyncMock(side_effect=HTTPException(status_code=403, detail="Admin access required")),
    ):
        resp = client.get(f"/orgs/{ORG_ID}/invites")
    assert resp.status_code == 403


def test_cancel_invite_ok(client):
    with patch("orgs.router.service.cancel_invite", new=AsyncMock(return_value={"deleted": "i1"})):
        resp = client.delete(f"/orgs/{ORG_ID}/invites/i1")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "i1"}


def test_cancel_invite_denied_for_non_admin_403(client):
    with patch(
        "orgs.router.service.cancel_invite",
        new=AsyncMock(side_effect=HTTPException(status_code=403, detail="Admin access required")),
    ):
        resp = client.delete(f"/orgs/{ORG_ID}/invites/i1")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PUT /orgs/{org_id}/members/{member_id}/role
# ---------------------------------------------------------------------------


def test_update_member_role_ok(client):
    with patch(
        "orgs.router.service.update_member_role", new=AsyncMock(return_value={"id": MEMBER_ID, "role": "admin"})
    ):
        resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/role", json={"role": "admin"})
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"


def test_update_member_role_last_admin_409(client):
    from orgs.service import LastAdminError

    with patch("orgs.router.service.update_member_role", new=AsyncMock(side_effect=LastAdminError("only admin"))):
        resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/role", json={"role": "member"})
    assert resp.status_code == 409


def test_update_member_role_invalid_role_400(client):
    with patch("orgs.router.service.update_member_role", new=AsyncMock(side_effect=ValueError("Invalid role"))):
        resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/role", json={"role": "owner"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /orgs/{org_id}/members/{member_id}/suspend
# DELETE /orgs/{org_id}/members/{member_id}
# POST /orgs/{org_id}/members/{member_id}/reactivate
# ---------------------------------------------------------------------------


def test_suspend_member_ok_fires_analytics(client):
    with (
        patch(
            "orgs.router.service.suspend_member",
            new=AsyncMock(return_value={"id": MEMBER_ID, "status": "suspended"}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/suspend")
    assert resp.status_code == 200
    assert resp.json()["status"] == "suspended"
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "org_license_revoked"
    assert mock_capture.call_args.args[2]["action"] == "suspend"


def test_suspend_member_last_admin_409(client):
    from orgs.service import LastAdminError

    with patch("orgs.router.service.suspend_member", new=AsyncMock(side_effect=LastAdminError("only admin"))):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/suspend")
    assert resp.status_code == 409


def test_suspend_member_reclaim_failed_502(client):
    from orgs.service import ReclaimFailedError

    with patch("orgs.router.service.suspend_member", new=AsyncMock(side_effect=ReclaimFailedError("reclaim failed"))):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/suspend")
    assert resp.status_code == 502


def test_suspend_member_not_found_404(client):
    with patch("orgs.router.service.suspend_member", new=AsyncMock(side_effect=ValueError("Member not found"))):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/suspend")
    assert resp.status_code == 404


def test_remove_member_ok_fires_analytics(client):
    with (
        patch("orgs.router.service.remove_member", new=AsyncMock(return_value={"id": MEMBER_ID, "status": "removed"})),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.delete(f"/orgs/{ORG_ID}/members/{MEMBER_ID}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "removed"
    assert mock_capture.call_args.args[1] == "org_license_revoked"
    assert mock_capture.call_args.args[2]["action"] == "remove"


def test_remove_member_last_admin_409(client):
    from orgs.service import LastAdminError

    with patch("orgs.router.service.remove_member", new=AsyncMock(side_effect=LastAdminError("only admin"))):
        resp = client.delete(f"/orgs/{ORG_ID}/members/{MEMBER_ID}")
    assert resp.status_code == 409


def test_remove_member_reclaim_failed_502(client):
    from orgs.service import ReclaimFailedError

    with patch("orgs.router.service.remove_member", new=AsyncMock(side_effect=ReclaimFailedError("boom"))):
        resp = client.delete(f"/orgs/{ORG_ID}/members/{MEMBER_ID}")
    assert resp.status_code == 502


def test_reactivate_member_ok(client):
    with patch(
        "orgs.router.service.reactivate_member", new=AsyncMock(return_value={"id": MEMBER_ID, "status": "active"})
    ):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reactivate")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


def test_reactivate_member_invalid_state_400(client):
    with patch(
        "orgs.router.service.reactivate_member",
        new=AsyncMock(side_effect=ValueError("Member is not suspended or removed")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reactivate")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# orgs.service.get_org_usage (Task 7) — direct service-level tests against a
# table-mocked db, mirroring test_orgs_service.py's idiom (this module can't
# import that file's private helpers, so a small self-contained fixture set
# lives here). Each table is queried exactly ONCE by design (see the
# function's docstring): one credit_wallets round trip returns BOTH the pool
# wallet and every seat wallet (owner_id never collides between an org and an
# org_members row), and the wallet ids it yields drive one follow-up
# credit_ledger round trip the same way — so a plain name-keyed
# MockQueryBuilder (one fixed response per table) is sufficient here.
# ---------------------------------------------------------------------------


class TestGetOrgUsageService:
    ORG = "20000000-0000-0000-0000-000000000099"
    ADMIN_MEMBER = "40000000-0000-0000-0000-0000000000a1"
    MEMBER_2 = "40000000-0000-0000-0000-0000000000a2"
    REMOVED_WITH_BALANCE = "40000000-0000-0000-0000-0000000000a3"
    REMOVED_AT_ZERO = "40000000-0000-0000-0000-0000000000a4"

    POOL_WALLET = "50000000-0000-0000-0000-0000000000b1"
    SEAT_WALLET_ADMIN = "50000000-0000-0000-0000-0000000000b2"
    SEAT_WALLET_MEMBER_2 = "50000000-0000-0000-0000-0000000000b3"
    SEAT_WALLET_REMOVED_BALANCE = "50000000-0000-0000-0000-0000000000b4"

    U_ADMIN = "00000000-0000-0000-0000-0000000000c1"
    U_MEMBER_2 = "00000000-0000-0000-0000-0000000000c2"
    U_REMOVED_BALANCE = "00000000-0000-0000-0000-0000000000c3"
    U_REMOVED_ZERO = "00000000-0000-0000-0000-0000000000c4"

    def _members(self):
        return [
            {"id": self.ADMIN_MEMBER, "user_id": self.U_ADMIN, "role": "admin", "status": "active"},
            {"id": self.MEMBER_2, "user_id": self.U_MEMBER_2, "role": "member", "status": "active"},
            {
                "id": self.REMOVED_WITH_BALANCE,
                "user_id": self.U_REMOVED_BALANCE,
                "role": "member",
                "status": "removed",
            },
            # No matching credit_wallets row at all for this one — proves the
            # "missing wallet" path also defaults balance to 0 (and is
            # therefore excluded, same as an explicit zero balance).
            {"id": self.REMOVED_AT_ZERO, "user_id": self.U_REMOVED_ZERO, "role": "member", "status": "removed"},
        ]

    def _wallets(self):
        return [
            {
                "id": self.POOL_WALLET,
                "owner_type": "org",
                "owner_id": self.ORG,
                "bundle_balance": 0,
                "reserve_balance": 4000,
            },
            {
                "id": self.SEAT_WALLET_ADMIN,
                "owner_type": "seat",
                "owner_id": self.ADMIN_MEMBER,
                "bundle_balance": 0,
                "reserve_balance": 300,
            },
            {
                "id": self.SEAT_WALLET_MEMBER_2,
                "owner_type": "seat",
                "owner_id": self.MEMBER_2,
                "bundle_balance": 0,
                "reserve_balance": 0,
            },
            {
                "id": self.SEAT_WALLET_REMOVED_BALANCE,
                "owner_type": "seat",
                "owner_id": self.REMOVED_WITH_BALANCE,
                "bundle_balance": 0,
                "reserve_balance": 40,
            },
        ]

    def _ledger(self):
        return [
            {"wallet_id": self.POOL_WALLET, "delta": 5000, "kind": "purchase"},
            {"wallet_id": self.SEAT_WALLET_ADMIN, "delta": -21, "kind": "debit"},
            {"wallet_id": self.SEAT_WALLET_ADMIN, "delta": -3, "kind": "debit"},
            # An allocation INTO the seat is not spend — must not count toward
            # spentAllTime (only kind='debit' rows do).
            {"wallet_id": self.SEAT_WALLET_ADMIN, "delta": 300, "kind": "allocation"},
            {"wallet_id": self.SEAT_WALLET_REMOVED_BALANCE, "delta": -12, "kind": "debit"},
        ]

    def _storage(self):
        return [
            {"user_id": self.U_ADMIN, "total_storage_bytes": 1000},
            {"user_id": self.U_MEMBER_2, "total_storage_bytes": 0},
        ]

    def _db(self, *, members=None, wallets=None, ledger=None, storage=None):
        members = self._members() if members is None else members
        wallets = self._wallets() if wallets is None else wallets
        ledger = self._ledger() if ledger is None else ledger
        storage = self._storage() if storage is None else storage

        def _side(name):
            b = MockQueryBuilder()
            if name == "org_members":
                b.execute.return_value = MagicMock(data=members, count=len(members))
            elif name == "credit_wallets":
                b.execute.return_value = MagicMock(data=wallets, count=len(wallets))
            elif name == "credit_ledger":
                b.execute.return_value = MagicMock(data=ledger, count=len(ledger))
            elif name == "usage_counters":
                b.execute.return_value = MagicMock(data=storage, count=len(storage))
            return b

        db = MagicMock()
        db.table.side_effect = _side
        db.auth.admin.get_user_by_id.side_effect = lambda uid: MagicMock(user=MagicMock(email=f"{uid}@example.com"))
        return db

    async def test_requires_admin_403(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
        db = self._db()
        with pytest.raises(HTTPException) as exc_info:
            await service.get_org_usage(db, self.U_MEMBER_2, self.ORG)
        assert exc_info.value.status_code == 403

    async def test_pool_balance_and_cumulative_purchased_computed_independently(self, monkeypatch):
        """poolBalance comes straight off the wallet row's current balance;
        cumulativePurchased is a separate ledger sum — they must not be
        conflated (the wallet balance reflects spend/allocations too)."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = self._db()
        result = await service.get_org_usage(db, self.U_ADMIN, self.ORG)
        assert result["poolBalance"] == 4000
        assert result["cumulativePurchased"] == 5000

    async def test_seat_balance_and_spent_all_time_math(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = self._db()
        result = await service.get_org_usage(db, self.U_ADMIN, self.ORG)
        seats = {s["orgMemberId"]: s for s in result["seats"]}
        admin_seat = seats[self.ADMIN_MEMBER]
        assert admin_seat["seatBalance"] == 300
        assert admin_seat["spentAllTime"] == 24  # 21 + 3 debits; the +300 allocation is not spend
        assert admin_seat["email"] == f"{self.U_ADMIN}@example.com"
        assert admin_seat["role"] == "admin"
        assert admin_seat["status"] == "active"
        assert admin_seat["userId"] == self.U_ADMIN

        member_2_seat = seats[self.MEMBER_2]
        assert member_2_seat["seatBalance"] == 0
        assert member_2_seat["spentAllTime"] == 0

    async def test_storage_bytes_and_cap_from_env(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setenv("ENTERPRISE_SEAT_STORAGE_BYTES", "999")
        db = self._db()
        result = await service.get_org_usage(db, self.U_ADMIN, self.ORG)
        seats = {s["orgMemberId"]: s for s in result["seats"]}
        assert seats[self.ADMIN_MEMBER]["storageBytes"] == 1000
        assert seats[self.ADMIN_MEMBER]["storageCapBytes"] == 999
        assert seats[self.MEMBER_2]["storageBytes"] == 0
        assert seats[self.MEMBER_2]["storageCapBytes"] == 999

    async def test_removed_member_with_stranded_balance_is_included(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = self._db()
        result = await service.get_org_usage(db, self.U_ADMIN, self.ORG)
        seats = {s["orgMemberId"]: s for s in result["seats"]}
        assert self.REMOVED_WITH_BALANCE in seats
        assert seats[self.REMOVED_WITH_BALANCE]["seatBalance"] == 40
        assert seats[self.REMOVED_WITH_BALANCE]["spentAllTime"] == 12
        assert seats[self.REMOVED_WITH_BALANCE]["status"] == "removed"

    async def test_removed_member_at_zero_balance_is_excluded(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = self._db()
        result = await service.get_org_usage(db, self.U_ADMIN, self.ORG)
        member_ids = {s["orgMemberId"] for s in result["seats"]}
        assert self.REMOVED_AT_ZERO not in member_ids
        # The other three (active x2 + removed-with-balance) ARE present.
        assert len(result["seats"]) == 3

    # -----------------------------------------------------------------------
    # Email resolution (licensing follow-ups Task 4): org_members.email is
    # captured going forward at accept-invite; get_org_usage must read it off
    # the row and fall back to the existing auth-admin lookup ONLY for rows
    # still NULL (pre-migration/creator rows), lazily healing them so the
    # lookup cost is paid at most once per row, ever.
    # -----------------------------------------------------------------------

    async def test_all_rows_with_email_zero_auth_lookups(self, monkeypatch):
        """Every row already carrying an email (the post-migration steady
        state) means the rollup never touches auth.admin — the whole point
        of denormalizing the email onto org_members."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        members = [{**m, "email": f"{m['user_id']}@example.com"} for m in self._members()]
        db = self._db(members=members)

        result = await service.get_org_usage(db, self.U_ADMIN, self.ORG)

        seats = {s["orgMemberId"]: s for s in result["seats"]}
        assert seats[self.ADMIN_MEMBER]["email"] == f"{self.U_ADMIN}@example.com"
        assert seats[self.MEMBER_2]["email"] == f"{self.U_MEMBER_2}@example.com"
        db.auth.admin.get_user_by_id.assert_not_called()

    def _db_with_one_null_email(self, *, on_org_members_update=None):
        """Members fixture with every row carrying an email EXCEPT
        REMOVED_WITH_BALANCE (simulating a legacy/creator row that predates
        the email column). `on_org_members_update` lets a test observe (or
        fail) the write-back UPDATE without disturbing the rest of the
        table-mocking plumbing in `_db`."""
        members = [
            {**self._members()[0], "email": f"{self.U_ADMIN}@example.com"},
            {**self._members()[1], "email": f"{self.U_MEMBER_2}@example.com"},
            {**self._members()[2], "email": None},
            self._members()[3],
        ]

        def _side(name):
            b = MockQueryBuilder()
            if name == "org_members":
                b.execute.return_value = MagicMock(data=members, count=len(members))
                if on_org_members_update is not None:
                    b.update = on_org_members_update
            elif name == "credit_wallets":
                wallets = self._wallets()
                b.execute.return_value = MagicMock(data=wallets, count=len(wallets))
            elif name == "credit_ledger":
                ledger = self._ledger()
                b.execute.return_value = MagicMock(data=ledger, count=len(ledger))
            elif name == "usage_counters":
                storage = self._storage()
                b.execute.return_value = MagicMock(data=storage, count=len(storage))
            return b

        db = MagicMock()
        db.table.side_effect = _side
        db.auth.admin.get_user_by_id.side_effect = lambda uid: MagicMock(user=MagicMock(email=f"{uid}@example.com"))
        return db, members

    async def test_null_email_row_resolves_once_and_heals_second_call_free(self, monkeypatch):
        """KEY TEST: a single NULL-email row (REMOVED_WITH_BALANCE) costs
        exactly one auth-admin lookup AND exactly one write-back UPDATE on
        the first rollup call. The write-back is simulated as landing (the
        row is mutated in place, same as a real UPDATE would persist it), so
        a SECOND rollup call for the same org costs zero lookups — the row
        is healed for good, not re-resolved every time."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        update_calls = []

        def _capture_and_heal(payload, *a, **kw):
            update_calls.append(payload)
            # Simulate the UPDATE landing so the next SELECT sees it healed.
            for row in members:
                if row["id"] == self.REMOVED_WITH_BALANCE:
                    row["email"] = payload.get("email", row.get("email"))
            return MockQueryBuilder()

        db, members = self._db_with_one_null_email(on_org_members_update=_capture_and_heal)

        result_1 = await service.get_org_usage(db, self.U_ADMIN, self.ORG)
        assert db.auth.admin.get_user_by_id.call_count == 1
        db.auth.admin.get_user_by_id.assert_called_once_with(self.U_REMOVED_BALANCE)
        assert update_calls == [{"email": f"{self.U_REMOVED_BALANCE}@example.com"}]
        seats_1 = {s["orgMemberId"]: s for s in result_1["seats"]}
        assert seats_1[self.REMOVED_WITH_BALANCE]["email"] == f"{self.U_REMOVED_BALANCE}@example.com"

        db.auth.admin.get_user_by_id.reset_mock()
        result_2 = await service.get_org_usage(db, self.U_ADMIN, self.ORG)
        assert db.auth.admin.get_user_by_id.call_count == 0
        seats_2 = {s["orgMemberId"]: s for s in result_2["seats"]}
        assert seats_2[self.REMOVED_WITH_BALANCE]["email"] == f"{self.U_REMOVED_BALANCE}@example.com"
        # Only the one write-back from the first call — none on the second.
        assert len(update_calls) == 1

    async def test_write_back_failure_does_not_break_usage_read(self, monkeypatch):
        """The write-back UPDATE is explicitly non-raising (best-effort) —
        a failure there must never surface as a failure of the usage read
        itself; the resolved email still comes back in the response."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

        def _boom(*a, **kw):
            raise RuntimeError("simulated write-back failure")

        db, _members = self._db_with_one_null_email(on_org_members_update=_boom)

        result = await service.get_org_usage(db, self.U_ADMIN, self.ORG)

        seats = {s["orgMemberId"]: s for s in result["seats"]}
        assert seats[self.REMOVED_WITH_BALANCE]["email"] == f"{self.U_REMOVED_BALANCE}@example.com"
