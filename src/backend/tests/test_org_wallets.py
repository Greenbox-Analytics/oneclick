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


# ---------------------------------------------------------------------------
# orgs.wallets — read_or_create_user_wallet (Task 10: moved here from
# EntitlementsService._read_or_create_wallet, which is now a one-line
# delegate — see TestEntitlementsWalletDelegate below).
# ---------------------------------------------------------------------------


def test_user_wallet_returns_existing_row_without_upsert():
    existing = {"id": "w1", "owner_type": "user", "owner_id": U1, "bundle_balance": 10, "reserve_balance": 20}
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=[existing], count=1)
    db = MagicMock()
    db.table.return_value = b

    assert wallets.read_or_create_user_wallet(db, U1) == existing


def test_user_wallet_creates_when_missing_seeds_period_end_now():
    """On a miss, the upsert payload is exactly {owner_type, owner_id,
    period_start, period_end} — period_end=now() is the seeding trick that
    makes the caller's next _maybe_rollover_wallet fire immediately and grant
    the tier's monthly credits, so no wallet ever starts un-granted."""
    captured = {}
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 1:
                b.execute.return_value = MagicMock(data=[], count=0)  # SELECT miss
            elif call_count["n"] == 2:
                original_upsert = b.upsert

                def _upsert(payload, **kw):
                    captured["payload"] = payload
                    captured["kw"] = kw
                    return original_upsert(payload, **kw)

                b.upsert = _upsert
                b.execute.return_value = MagicMock(data=[{"id": "w2"}], count=1)
            else:
                b.execute.return_value = MagicMock(
                    data=[
                        {"id": "w2", "owner_type": "user", "owner_id": U1, "bundle_balance": 0, "reserve_balance": 0}
                    ],
                    count=1,
                )
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = wallets.read_or_create_user_wallet(db, U1)

    assert result["id"] == "w2"
    assert captured["payload"]["owner_type"] == "user"
    assert captured["payload"]["owner_id"] == U1
    assert "period_start" in captured["payload"]
    assert "period_end" in captured["payload"]
    assert captured["kw"] == {"on_conflict": "owner_type,owner_id"}


class TestEntitlementsWalletDelegate:
    """EntitlementsService._read_or_create_wallet is now a one-line delegate
    to wallets.read_or_create_user_wallet (Task 10 ponytail cut: one
    wallet-read-or-create per owner type, not siblings). This pins the
    delegate's OWN shape; tests/test_entitlements_service.py and
    tests/test_credits_service.py (esp. TestWalletOwnerScoping) are what
    prove the refactor caused no behavioral regression end-to-end."""

    def test_delegates_to_orgs_wallets_read_or_create_user_wallet(self):
        from subscriptions.service import EntitlementsService

        existing = {"id": "w1", "owner_type": "user", "owner_id": U1, "bundle_balance": 10, "reserve_balance": 20}
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=[existing], count=1)
        db = MagicMock()
        db.table.return_value = b

        direct = wallets.read_or_create_user_wallet(db, U1)
        via_service = EntitlementsService(db)._read_or_create_wallet(U1)

        assert via_service == direct == existing


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


async def test_set_member_cap_accepts_the_unlimited_sentinel(monkeypatch):
    """-1 writes through: it is how an admin says "no limit" for a member whose
    org default would otherwise cap them."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
        data={"id": MEMBER_ID}
    )
    db.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"id": MEMBER_ID, "monthly_cap": -1}]
    )
    out = await service.set_member_cap(db, U1, ORG_ID, MEMBER_ID, service.UNLIMITED_CAP)
    assert out["monthly_cap"] == -1


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


def test_cap_router_rejects_below_the_sentinel_422(client):
    """-1 (no limit) is valid at the model; -2 is not."""
    resp = client.put(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/cap", json={"cap": -2})
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


# ---------------------------------------------------------------------------
# Cap chain: new members start capped (org default 2,000), and -1 means
# "no limit" — the meaning NULL used to carry before every org gained a default.
# ---------------------------------------------------------------------------


class TestEffectiveMemberCap:
    def test_members_own_cap_wins(self):
        assert service.effective_member_cap(500, 2000) == 500

    def test_null_inherits_the_org_default(self):
        """This is why -1 had to exist: NULL can no longer mean "no limit"."""
        assert service.effective_member_cap(None, 2000) == 2000

    def test_sentinel_means_no_ceiling(self):
        assert service.effective_member_cap(service.UNLIMITED_CAP, 2000) is None

    def test_org_wide_sentinel_uncaps_inheriting_members(self):
        assert service.effective_member_cap(None, service.UNLIMITED_CAP) is None

    def test_member_cap_overrides_an_unlimited_org_default(self):
        assert service.effective_member_cap(300, service.UNLIMITED_CAP) == 300

    def test_zero_is_a_real_ceiling_not_unlimited(self):
        """0 blocks spend; it must never be confused with "no limit"."""
        assert service.effective_member_cap(0, 2000) == 0

    def test_unset_chain_is_still_uncapped(self):
        assert service.effective_member_cap(None, None) is None


# ---------------------------------------------------------------------------
# transfer_credits_to_pool (Task 10, spec §4.1) — the owner-requested funding
# inlet: an active admin moves credits from their OWN personal reserve into
# this org's pool via the transfer_credits RPC.
# ---------------------------------------------------------------------------

TRANSFER_ORG = "20000000-0000-0000-0000-000000000002"
PERSONAL_WALLET = "50000000-0000-0000-0000-000000000003"


def _seq_db(table_data):
    """table_data: dict table_name -> list of `data` payloads, consumed in
    call order (last value repeats on overrun). Local variant of
    tests/test_orgs_service.py's `_db_seq` idiom, scoped to this module."""
    counters = {k: 0 for k in table_data}

    def _side(name):
        b = MockQueryBuilder()
        if name in table_data:
            seq = table_data[name]
            i = min(counters[name], len(seq) - 1)
            counters[name] += 1
            b.execute.return_value = MagicMock(data=seq[i], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


def _transfer_org_row(**overrides):
    """_first_org's select list: kind, status, covered_by, covered_at,
    archived_at, dissolved_at."""
    row = {
        "kind": "self_serve",
        "status": "active",
        "covered_by": U1,
        "covered_at": "2026-08-01T00:00:00+00:00",
        "archived_at": None,
        "dissolved_at": None,
    }
    row.update(overrides)
    return row


def _wallet_row(owner_type, owner_id, **overrides):
    row = {
        "id": PERSONAL_WALLET if owner_type == "user" else POOL_WALLET,
        "owner_type": owner_type,
        "owner_id": owner_id,
        "bundle_balance": 0,
        "reserve_balance": 500,
    }
    row.update(overrides)
    return row


async def test_transfer_credits_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.transfer_credits_to_pool(MagicMock(), U1, TRANSFER_ORG, 100)
    assert exc_info.value.status_code == 403


async def test_transfer_credits_enterprise_org_409(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _seq_db({"organizations": [_transfer_org_row(kind="enterprise")]})

    with pytest.raises(HTTPException) as exc_info:
        await service.transfer_credits_to_pool(db, U1, TRANSFER_ORG, 100)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "This organization is managed by Msanii"
    db.rpc.assert_not_called()


async def test_transfer_credits_archived_org_409(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _seq_db({"organizations": [_transfer_org_row(archived_at="2026-08-01T00:00:00+00:00")]})

    with pytest.raises(HTTPException) as exc_info:
        await service.transfer_credits_to_pool(db, U1, TRANSFER_ORG, 100)

    assert exc_info.value.status_code == 409
    db.rpc.assert_not_called()


async def test_transfer_credits_dissolved_org_409(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _seq_db(
        {
            "organizations": [
                _transfer_org_row(archived_at="2026-08-01T00:00:00+00:00", dissolved_at="2026-08-01T00:00:00+00:00")
            ]
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.transfer_credits_to_pool(db, U1, TRANSFER_ORG, 100)

    assert exc_info.value.status_code == 409
    db.rpc.assert_not_called()


async def test_transfer_credits_happy_path_calls_rpc_with_both_wallets_and_metadata(monkeypatch):
    """KEY TEST: the RPC name and every argument — both wallet ids (in the
    right slots), the amount, and the metadata the pool ledger groups
    per-admin spend on."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    personal_wallet = _wallet_row("user", U1, reserve_balance=500)
    pool_wallet = _wallet_row("org", TRANSFER_ORG)
    db = _seq_db(
        {
            "organizations": [_transfer_org_row()],
            # credit_wallets reads (via read_wallet) are plain .select() calls,
            # not .maybe_single() — `data` is a LIST of rows, unlike the
            # `organizations` entry above.
            "credit_wallets": [[personal_wallet], [pool_wallet]],
        }
    )
    captured = {}

    def _rpc(name, params):
        captured["name"] = name
        captured["params"] = params
        m = MagicMock()
        m.execute.return_value = MagicMock(data={"duplicate": False, "balance_after": 400})
        return m

    db.rpc.side_effect = _rpc

    result = await service.transfer_credits_to_pool(db, U1, TRANSFER_ORG, 100)

    assert captured["name"] == "transfer_credits"
    assert captured["params"]["p_from_wallet"] == PERSONAL_WALLET
    assert captured["params"]["p_to_wallet"] == POOL_WALLET
    assert captured["params"]["p_amount"] == 100
    assert captured["params"]["p_request_id"].startswith("xfer:")
    assert captured["params"]["p_metadata"] == {"org_id": TRANSFER_ORG, "admin_user_id": U1}
    assert result == {"duplicate": False, "balance_after": 400}


async def test_transfer_credits_duplicate_rpc_result_is_treated_as_success(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    personal_wallet = _wallet_row("user", U1, reserve_balance=500)
    pool_wallet = _wallet_row("org", TRANSFER_ORG)
    db = _seq_db(
        {
            "organizations": [_transfer_org_row()],
            "credit_wallets": [[personal_wallet], [pool_wallet]],
        }
    )
    db.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": True, "balance_after": 300})

    result = await service.transfer_credits_to_pool(db, U1, TRANSFER_ORG, 100)

    assert result == {"duplicate": True, "balance_after": 300}


async def test_transfer_credits_insufficient_reserve_maps_to_409_with_balance(monkeypatch):
    """The RPC RAISEs (doesn't clamp) on an under-funded reserve — mapped to
    409 with the reserve balance RE-READ (not the pre-call snapshot)."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    personal_wallet = _wallet_row("user", U1, reserve_balance=10)
    pool_wallet = _wallet_row("org", TRANSFER_ORG)
    reread_wallet = _wallet_row("user", U1, reserve_balance=10)
    db = _seq_db(
        {
            "organizations": [_transfer_org_row()],
            "credit_wallets": [[personal_wallet], [pool_wallet], [reread_wallet]],
        }
    )

    def _rpc(name, params):
        m = MagicMock()
        m.execute.side_effect = Exception("insufficient reserve: have 10, need 100")
        return m

    db.rpc.side_effect = _rpc

    with pytest.raises(HTTPException) as exc_info:
        await service.transfer_credits_to_pool(db, U1, TRANSFER_ORG, 100)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["reserveBalance"] == 10
    assert exc_info.value.detail["reason"]


async def test_transfer_credits_other_rpc_exception_propagates(monkeypatch):
    """Anything that isn't the insufficient-reserve message is a real failure
    (bad wallet id, DB down, ...) and must propagate — 500 at the router, not
    a swallowed/misreported 409."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    personal_wallet = _wallet_row("user", U1, reserve_balance=500)
    pool_wallet = _wallet_row("org", TRANSFER_ORG)
    db = _seq_db(
        {
            "organizations": [_transfer_org_row()],
            "credit_wallets": [[personal_wallet], [pool_wallet]],
        }
    )

    def _rpc(name, params):
        m = MagicMock()
        m.execute.side_effect = RuntimeError("connection reset")
        return m

    db.rpc.side_effect = _rpc

    with pytest.raises(RuntimeError, match="connection reset"):
        await service.transfer_credits_to_pool(db, U1, TRANSFER_ORG, 100)


# --- Router contract: POST /orgs/{org_id}/transfer-credits ---


def test_transfer_credits_router_fires_analytics_on_fresh_transfer(client):
    with (
        patch(
            "orgs.router.service.transfer_credits_to_pool",
            new=AsyncMock(return_value={"duplicate": False, "balance_after": 300}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{TRANSFER_ORG}/transfer-credits", json={"amount": 100})

    assert resp.status_code == 200
    assert resp.json()["duplicate"] is False
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "credits_transferred"
    assert mock_capture.call_args.args[2] == {"org_id": TRANSFER_ORG, "amount": 100}


def test_transfer_credits_router_duplicate_is_200_without_analytics(client):
    with (
        patch(
            "orgs.router.service.transfer_credits_to_pool",
            new=AsyncMock(return_value={"duplicate": True, "balance_after": 300}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{TRANSFER_ORG}/transfer-credits", json={"amount": 100})

    assert resp.status_code == 200
    assert resp.json()["duplicate"] is True
    mock_capture.assert_not_called()


@pytest.mark.parametrize("amount", [0, -5])
def test_transfer_credits_router_rejects_non_positive_amount_422(client, amount):
    resp = client.post(f"/orgs/{TRANSFER_ORG}/transfer-credits", json={"amount": amount})
    assert resp.status_code == 422


def test_transfer_credits_router_rejects_over_ceiling_422(client):
    resp = client.post(f"/orgs/{TRANSFER_ORG}/transfer-credits", json={"amount": 1_000_001})
    assert resp.status_code == 422


def test_transfer_credits_router_404_when_licensing_off(client, monkeypatch):
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)
    resp = client.post(f"/orgs/{TRANSFER_ORG}/transfer-credits", json={"amount": 100})
    assert resp.status_code == 404
