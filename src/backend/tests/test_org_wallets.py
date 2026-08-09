"""Tests for orgs.wallets (the create-on-miss org POOL helper and the paid-in
sum) plus the cap-management service functions and router endpoints.

There are no seat wallets: members spend from the one org pool against a monthly
cap, so what used to be allocate/reclaim transfer tests are now cap-writing tests
— the interesting property being that setting a ceiling moves no money and so has
no failure mode to map. Mirrors tests/test_orgs_service.py's mock idiom for the
service-level tests and tests/test_orgs_router.py's `client` fixture for the
endpoint contracts.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from orgs import service, wallets
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
ORG_ID = "20000000-0000-0000-0000-000000000001"
MEMBER_ID = "40000000-0000-0000-0000-000000000001"
POOL_WALLET = "50000000-0000-0000-0000-000000000002"


# ---------------------------------------------------------------------------
# orgs.wallets — read_or_create_org_wallet
# ---------------------------------------------------------------------------


def test_org_wallet_returns_existing_row_without_insert():
    existing = {"id": POOL_WALLET, "owner_type": "org", "owner_id": ORG_ID, "bundle_balance": 0, "reserve_balance": 500}
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=[existing], count=1)
    db = MagicMock()
    db.table.return_value = b

    assert wallets.read_or_create_org_wallet(db, ORG_ID) == existing
    b.insert.assert_not_called()


def test_org_wallet_insert_payload_has_exactly_owner_type_and_owner_id():
    """KEY TEST: the INSERT payload must be EXACTLY {owner_type, owner_id} — no
    period fields. The pool's first period is written by the dispersal sweep's
    rollover, and seeding one here would arm an expiry nobody asked for. Uses
    INSERT (not upsert) — verified by the duplicate-race tests below."""
    captured = {}
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 1:
                b.execute.return_value = MagicMock(data=[], count=0)  # SELECT miss
            else:
                original_insert = b.insert

                def _insert(payload):
                    captured["payload"] = payload
                    return original_insert(payload)

                b.insert = _insert
                b.execute.return_value = MagicMock(data=[{"id": POOL_WALLET}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = wallets.read_or_create_org_wallet(db, ORG_ID)

    assert result["id"] == POOL_WALLET
    assert captured["payload"] == {"owner_type": "org", "owner_id": ORG_ID}


def test_org_wallet_duplicate_race_falls_back_to_reselect():
    """KEY TEST: the INSERT raising (unique_violation from a concurrent
    create-on-miss winner) must NOT propagate — it's caught and the wallet the
    racer created is re-selected instead."""
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 1:
                b.execute.return_value = MagicMock(data=[], count=0)  # initial SELECT miss
            elif call_count["n"] == 2:
                b.execute.side_effect = Exception("duplicate key value violates unique constraint")
            else:
                b.execute.return_value = MagicMock(data=[{"id": POOL_WALLET}], count=1)  # re-SELECT
        return b

    db = MagicMock()
    db.table.side_effect = _side

    assert wallets.read_or_create_org_wallet(db, ORG_ID) == {"id": POOL_WALLET}
    assert call_count["n"] == 3


def test_org_wallet_duplicate_race_reselect_also_empty_reraises():
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 2:
                b.execute.side_effect = Exception("duplicate key value violates unique constraint")
            else:
                b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    with pytest.raises(Exception, match="duplicate key"):
        wallets.read_or_create_org_wallet(db, ORG_ID)


def test_org_wallet_insert_with_no_data_falls_back_to_reselect():
    """Some client/mocking configurations don't echo the inserted row — the
    helper re-selects rather than raising."""
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 3:
                b.execute.return_value = MagicMock(data=[{"id": POOL_WALLET}], count=1)
            else:
                b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    assert wallets.read_or_create_org_wallet(db, ORG_ID) == {"id": POOL_WALLET}


def test_org_wallet_insert_no_data_and_reselect_empty_raises_runtime_error():
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=[], count=0)
    db = MagicMock()
    db.table.return_value = b

    with pytest.raises(RuntimeError, match="failed to read or create org wallet"):
        wallets.read_or_create_org_wallet(db, ORG_ID)


# ---------------------------------------------------------------------------
# cumulative_paid_in — purchases AND dispersals both count as money in
# ---------------------------------------------------------------------------


def test_cumulative_paid_in_sums_purchase_and_dispersal_kinds():
    """Load-bearing: an org whose only funding is its monthly contract dispersal
    must still clear the activation floor. If this summed 'purchase' alone, a
    contract-only org would sit pending forever with its seats conferring
    nothing. 'monthly_grant' is in the filter because the dispersal sweep is
    implemented via rollover_wallet, which writes THAT kind — on org wallets
    the sweep is its only writer, so it IS the dispersal component (kind
    'dispersal' itself has never been written by anything)."""
    captured = {}
    b = MockQueryBuilder()
    original_in = b.in_

    def _in(col, vals):
        captured["col"] = col
        captured["vals"] = list(vals)
        return original_in(col, vals)

    b.in_ = _in
    b.execute.return_value = MagicMock(
        data=[{"delta": 10000, "kind": "purchase"}, {"delta": 5000, "kind": "monthly_grant"}], count=2
    )
    db = MagicMock()
    db.table.return_value = b

    assert wallets.cumulative_paid_in(db, POOL_WALLET) == 15000
    assert captured["col"] == "kind"
    # 'admin_grant' added 2026-08-08 — comped org grants count toward activation (admin credits & testers spec)
    assert set(captured["vals"]) == {"purchase", "dispersal", "monthly_grant", "admin_grant"}


# ---------------------------------------------------------------------------
# set_member_cap / set_org_dispersal — service level
# ---------------------------------------------------------------------------


async def test_set_member_cap_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.set_member_cap(MagicMock(), U1, ORG_ID, MEMBER_ID, 2000)
    assert exc_info.value.status_code == 403


async def test_set_member_cap_cross_org_member_404_without_writing(monkeypatch):
    """IDOR guard: the caller is an admin of THIS org, but member_id resolves to
    no org_members row scoped to it. Must 404 before writing anything —
    otherwise an admin of a free self-created org could raise a member's cap in
    someone else's org."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
        data=None
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.set_member_cap(db, U1, ORG_ID, MEMBER_ID, 2000)
    assert exc_info.value.status_code == 404
    db.table.return_value.update.assert_not_called()


async def test_set_member_cap_writes_monthly_cap_and_moves_no_money(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": MEMBER_ID, "monthly_cap": 2000}], count=1)
            original_update = b.update

            def _update(payload):
                captured["payload"] = payload
                return original_update(payload)

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.set_member_cap(db, U1, ORG_ID, MEMBER_ID, 2000)

    assert captured["payload"] == {"monthly_cap": 2000}
    assert result["monthly_cap"] == 2000
    db.rpc.assert_not_called()  # a ceiling is not a transfer


async def test_set_member_cap_none_clears_to_org_default(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": MEMBER_ID, "monthly_cap": None}], count=1)
            original_update = b.update

            def _update(payload):
                captured["payload"] = payload
                return original_update(payload)

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side

    await service.set_member_cap(db, U1, ORG_ID, MEMBER_ID, None)

    assert captured["payload"] == {"monthly_cap": None}


async def test_set_member_cap_rejects_negative(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
        data={"id": MEMBER_ID}
    )
    with pytest.raises(ValueError, match="cap must be >= 0"):
        await service.set_member_cap(db, U1, ORG_ID, MEMBER_ID, -1)


async def test_set_org_dispersal_writes_the_contract_volume(monkeypatch):
    """No org-admin authz inside: the dispersal is an OPERATOR dial, gated by the
    admin router. The service takes no acting user_id precisely so it can't be
    wired to an org-admin route by accident."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG_ID}], count=1)
            original_update = b.update

            def _update(payload):
                captured["payload"] = payload
                return original_update(payload)

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side

    await service.set_org_dispersal(db, ORG_ID, 10000)

    assert captured["payload"] == {"monthly_dispersal_credits": 10000}
    # Raising it must NOT top the pool up here — the sweep is the only writer of
    # dispersal credits, which is what keeps its monthly idempotency honest.
    db.rpc.assert_not_called()


async def test_set_org_dispersal_rejects_negative(monkeypatch):
    with pytest.raises(ValueError, match="monthly_dispersal_credits must be >= 0"):
        await service.set_org_dispersal(MagicMock(), ORG_ID, -1)


async def test_set_org_dispersal_does_not_touch_the_member_cap(monkeypatch):
    """default_member_cap is the CUSTOMER's dial (it divides what they paid for)
    and rides PUT /orgs/{id}; the operator endpoint must never write it."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG_ID}], count=1)
            original_update = b.update

            def _update(payload):
                captured["payload"] = payload
                return original_update(payload)

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side

    await service.set_org_dispersal(db, ORG_ID, 500)

    assert "default_member_cap" not in captured["payload"]


# ---------------------------------------------------------------------------
# Router contracts
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _licensing_on_by_default(monkeypatch):
    monkeypatch.setenv("LICENSING_ENABLED", "true")


def test_cap_router_ok_relays_result_and_fires_analytics(client):
    with (
        patch("orgs.router.service.set_member_cap", new=AsyncMock(return_value={"id": MEMBER_ID, "monthly_cap": 2000})),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/cap", json={"cap": 2000})
    assert resp.status_code == 200
    assert resp.json()["monthly_cap"] == 2000
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "org_member_cap_set"


def test_cap_router_accepts_null_to_clear(client):
    with (
        patch("orgs.router.service.set_member_cap", new=AsyncMock(return_value={"id": MEMBER_ID})) as mock_set,
        patch("orgs.router.analytics_capture"),
    ):
        resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/cap", json={"cap": None})
    assert resp.status_code == 200
    assert mock_set.await_args.args[-1] is None


def test_cap_router_rejects_negative_422(client):
    resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/cap", json={"cap": -1})
    assert resp.status_code == 422


def test_cap_router_404_when_licensing_off(client, monkeypatch):
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)
    resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/cap", json={"cap": 100})
    assert resp.status_code == 404


def test_org_router_has_no_dispersal_endpoint(client):
    """REGRESSION GUARD for a hole this replaced: the dispersal used to sit on
    /orgs/{id}/dispersal behind org-admin authz. Since any signed-in user can
    create an org and is auto-made its admin — and dispersed credits count toward
    the activation floor — that let anyone mint themselves unlimited monthly
    credits and self-activate. It must never come back to a customer route."""
    resp = client.put(f"/orgs/{ORG_ID}/dispersal", json={"monthly_dispersal_credits": 1_000_000})
    assert resp.status_code in (404, 405), f"a customer-facing dispersal route exists: {resp.status_code}"


def test_admin_dispersal_endpoint_requires_platform_admin(client, monkeypatch):
    """And on the admin surface it is gated by Msanii's own admin dependency,
    not by org membership.

    ADMIN_EMAILS is pinned to somebody OTHER than the caller (the house idiom —
    see test_admin_router's _set_admin_emails). Without it this test reads the
    developer's .env: with an allowlist present require_admin returns 403, but
    with an EMPTY one it takes its deliberate "no admins configured" branch and
    returns 500, so the same assertion passed locally and failed in CI.

    Nothing else is patched — the dependency must reject the caller BEFORE any
    service call, and a passing service mock would hide the very thing under test.
    """
    monkeypatch.setenv("ADMIN_EMAILS", "someone-else@example.com")
    resp = client.put(f"/admin/orgs/{ORG_ID}/dispersal", json={"monthly_dispersal_credits": 10000})
    assert resp.status_code == 403, f"expected an admin gate, got {resp.status_code}: {resp.text[:200]}"
