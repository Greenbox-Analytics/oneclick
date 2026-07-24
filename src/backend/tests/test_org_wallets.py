"""Tests for orgs.wallets (create-on-miss seat/org wallet helpers) and the
allocate/reclaim service functions + router endpoints (Licensing Phase B,
Task 4). Mirrors tests/test_orgs_service.py's `_db_seq`-style mock idiom for
the service-level tests and tests/test_orgs_router.py's `client`-fixture
idiom for the endpoint-contract tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from orgs import service, wallets
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
ORG_ID = "20000000-0000-0000-0000-000000000001"
MEMBER_ID = "40000000-0000-0000-0000-000000000001"
SEAT_WALLET = "50000000-0000-0000-0000-000000000001"
POOL_WALLET = "50000000-0000-0000-0000-000000000002"


# ---------------------------------------------------------------------------
# orgs.wallets — read_or_create_org_wallet / read_or_create_seat_wallet
# ---------------------------------------------------------------------------


def test_org_wallet_returns_existing_row_without_insert():
    existing = {"id": POOL_WALLET, "owner_type": "org", "owner_id": ORG_ID, "bundle_balance": 0, "reserve_balance": 500}
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=[existing], count=1)
    db = MagicMock()
    db.table.return_value = b

    result = wallets.read_or_create_org_wallet(db, ORG_ID)

    assert result == existing
    b.insert.assert_not_called()


def test_seat_wallet_returns_existing_row_without_insert():
    existing = {
        "id": SEAT_WALLET,
        "owner_type": "seat",
        "owner_id": MEMBER_ID,
        "bundle_balance": 0,
        "reserve_balance": 200,
    }
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=[existing], count=1)
    db = MagicMock()
    db.table.return_value = b

    result = wallets.read_or_create_seat_wallet(db, MEMBER_ID)

    assert result == existing
    b.insert.assert_not_called()


def test_org_wallet_insert_payload_has_exactly_owner_type_and_owner_id():
    """KEY TEST: the INSERT payload must be EXACTLY {owner_type, owner_id} —
    no period_start/period_end/anything else. NULL periods forever is the
    structural rollover exemption (rule 1). Uses INSERT (not upsert) —
    verified separately by the duplicate-race tests below."""
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


def test_seat_wallet_insert_payload_has_exactly_owner_type_and_owner_id():
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
                b.execute.return_value = MagicMock(data=[{"id": SEAT_WALLET}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = wallets.read_or_create_seat_wallet(db, MEMBER_ID)

    assert result["id"] == SEAT_WALLET
    assert captured["payload"] == {"owner_type": "seat", "owner_id": MEMBER_ID}


def test_org_wallet_duplicate_race_falls_back_to_reselect():
    """KEY TEST: the INSERT raising (unique_violation from a concurrent
    create-on-miss winner) must NOT propagate — it's caught and the wallet
    the racer created is re-selected instead."""
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
                b.execute.return_value = MagicMock(data=[{"id": POOL_WALLET}], count=1)  # re-SELECT after race
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = wallets.read_or_create_org_wallet(db, ORG_ID)

    assert result == {"id": POOL_WALLET}
    assert call_count["n"] == 3


def test_seat_wallet_duplicate_race_falls_back_to_reselect():
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 1:
                b.execute.return_value = MagicMock(data=[], count=0)
            elif call_count["n"] == 2:
                b.execute.side_effect = Exception("duplicate key value violates unique constraint")
            else:
                b.execute.return_value = MagicMock(data=[{"id": SEAT_WALLET}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = wallets.read_or_create_seat_wallet(db, MEMBER_ID)

    assert result == {"id": SEAT_WALLET}


def test_org_wallet_duplicate_race_reselect_also_empty_reraises():
    """If the INSERT raised AND the re-select still finds nothing, the
    original exception propagates rather than being swallowed silently."""
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 1:
                b.execute.return_value = MagicMock(data=[], count=0)
            elif call_count["n"] == 2:
                b.execute.side_effect = RuntimeError("duplicate key value violates unique constraint")
            else:
                b.execute.return_value = MagicMock(data=[], count=0)  # still missing
        return b

    db = MagicMock()
    db.table.side_effect = _side

    with pytest.raises(RuntimeError):
        wallets.read_or_create_org_wallet(db, ORG_ID)


def test_org_wallet_insert_with_no_data_falls_back_to_reselect():
    """Some client/mock configurations echo no `data` on a successful INSERT
    even though the row landed — the helper falls back to a re-select before
    giving up (distinct from the duplicate-race/exception path above)."""
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            if call_count["n"] == 1:
                b.execute.return_value = MagicMock(data=[], count=0)
            elif call_count["n"] == 2:
                b.execute.return_value = MagicMock(data=[], count=0)  # INSERT echoes nothing
            else:
                b.execute.return_value = MagicMock(data=[{"id": POOL_WALLET}], count=1)  # re-SELECT succeeds
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = wallets.read_or_create_org_wallet(db, ORG_ID)

    assert result == {"id": POOL_WALLET}


def test_seat_wallet_insert_no_data_and_reselect_empty_raises_runtime_error():
    call_count = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "credit_wallets":
            call_count["n"] += 1
            b.execute.return_value = MagicMock(data=[], count=0)  # every call comes up empty
        return b

    db = MagicMock()
    db.table.side_effect = _side

    with pytest.raises(RuntimeError):
        wallets.read_or_create_seat_wallet(db, MEMBER_ID)


# ---------------------------------------------------------------------------
# orgs.service.allocate_credits
# ---------------------------------------------------------------------------


def _patch_wallets(monkeypatch, seat=None, pool=None):
    seat = seat or {"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 0}
    pool = pool or {"id": POOL_WALLET, "bundle_balance": 0, "reserve_balance": 0}
    monkeypatch.setattr(service.wallets, "read_or_create_seat_wallet", lambda db, member_id: seat)
    monkeypatch.setattr(service.wallets, "read_or_create_org_wallet", lambda db, org_id: pool)
    return seat, pool


async def test_allocate_credits_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.allocate_credits(MagicMock(), U1, ORG_ID, MEMBER_ID, 100, "key-1")
    assert exc_info.value.status_code == 403


async def test_allocate_credits_cross_org_member_404_no_transfer(monkeypatch):
    """IDOR guard: caller is admin of THIS org, but member_id resolves to no
    org_members row scoped to it (belongs to another org / doesn't exist). Must
    404 before resolving any wallet or calling transfer_credits — otherwise an
    admin of a free self-created org could siphon another org's seat credits."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
        data=None
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.allocate_credits(db, U1, ORG_ID, MEMBER_ID, 100, "key-1")
    assert exc_info.value.status_code == 404
    db.rpc.assert_not_called()


async def test_reclaim_credits_cross_org_member_404_no_transfer(monkeypatch):
    """IDOR guard (reclaim direction): same missing-scope check — a foreign
    member_id must 404 before any seat->pool transfer moves credits out."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
        data=None
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.reclaim_credits(db, U1, ORG_ID, MEMBER_ID, None, "key-1")
    assert exc_info.value.status_code == 404
    db.rpc.assert_not_called()


async def test_allocate_credits_rpc_call_shape_uses_base_request_id(monkeypatch):
    """KEY TEST: pool -> seat direction, 'allocation' kind, the BASE
    `alloc:{key}` request_id (transfer_credits appends :from/:to itself —
    never suffix in Python), and the {org_id, member_id, actor} metadata."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch)
    db = MagicMock()
    db.rpc.return_value.execute.return_value = MagicMock(
        data={"duplicate": False, "from_balance": 900, "to_balance": 100}
    )

    result = await service.allocate_credits(db, U1, ORG_ID, MEMBER_ID, 100, "key-1")

    name, params = db.rpc.call_args[0]
    assert name == "transfer_credits"
    assert params["p_from_wallet"] == POOL_WALLET
    assert params["p_to_wallet"] == SEAT_WALLET
    assert params["p_amount"] == 100
    assert params["p_kind"] == "allocation"
    assert params["p_request_id"] == "alloc:key-1"
    assert params["p_metadata"] == {"org_id": ORG_ID, "member_id": MEMBER_ID, "actor": U1}
    assert result == {"duplicate": False, "from_balance": 900, "to_balance": 100}


async def test_allocate_credits_insufficient_pool_balance_maps_to_409_error(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch)
    db = MagicMock()
    db.rpc.return_value.execute.side_effect = RuntimeError("insufficient balance on source wallet (have 10, need 100)")

    with pytest.raises(service.PoolBalanceInsufficientError):
        await service.allocate_credits(db, U1, ORG_ID, MEMBER_ID, 100, "key-1")


async def test_allocate_credits_other_rpc_errors_propagate_unmapped(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch)
    db = MagicMock()
    db.rpc.return_value.execute.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await service.allocate_credits(db, U1, ORG_ID, MEMBER_ID, 100, "key-1")


# ---------------------------------------------------------------------------
# orgs.service.reclaim_credits
# ---------------------------------------------------------------------------


async def test_reclaim_credits_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.reclaim_credits(MagicMock(), U1, ORG_ID, MEMBER_ID, 100, "key-1")
    assert exc_info.value.status_code == 403


async def test_reclaim_credits_explicit_amount_rpc_call_shape(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch, seat={"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 900})
    db = MagicMock()
    db.rpc.return_value.execute.return_value = MagicMock(
        data={"duplicate": False, "from_balance": 800, "to_balance": 100}
    )

    result = await service.reclaim_credits(db, U1, ORG_ID, MEMBER_ID, 100, "key-2")

    name, params = db.rpc.call_args[0]
    assert name == "transfer_credits"
    assert params["p_from_wallet"] == SEAT_WALLET
    assert params["p_to_wallet"] == POOL_WALLET
    assert params["p_amount"] == 100
    assert params["p_kind"] == "reclaim"
    assert params["p_request_id"] == "reclaim:key-2"
    assert params["p_metadata"] == {"org_id": ORG_ID, "member_id": MEMBER_ID, "actor": U1}
    assert result["from_balance"] == 800


async def test_reclaim_all_reads_balance_and_passes_it_as_amount(monkeypatch):
    """KEY TEST: amount=None reads the seat's current balance
    (bundle + reserve) from the SAME create-on-miss lookup used to resolve
    its wallet id, and uses THAT as the RPC's p_amount."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch, seat={"id": SEAT_WALLET, "bundle_balance": 30, "reserve_balance": 270})
    db = MagicMock()
    db.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})

    await service.reclaim_credits(db, U1, ORG_ID, MEMBER_ID, None, "key-3")

    params = db.rpc.call_args[0][1]
    assert params["p_amount"] == 300
    assert params["p_request_id"] == "reclaim:key-3"


async def test_reclaim_all_zero_balance_returns_removed_zero_without_rpc(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch, seat={"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 0})
    db = MagicMock()

    result = await service.reclaim_credits(db, U1, ORG_ID, MEMBER_ID, None, "key-4")

    assert result == {"removed": 0}
    db.rpc.assert_not_called()


async def test_reclaim_all_negative_balance_returns_removed_zero_without_rpc(monkeypatch):
    """KEY TEST (round 5 guard): accepted debit drift can push a seat's
    bundle negative. Reclaim-all must no-op rather than pass a negative
    amount to transfer_credits, which would raise on p_amount <= 0 — an
    unmapped 500 instead of a quiet no-op."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch, seat={"id": SEAT_WALLET, "bundle_balance": -50, "reserve_balance": 20})
    db = MagicMock()

    result = await service.reclaim_credits(db, U1, ORG_ID, MEMBER_ID, None, "key-5")

    assert result == {"removed": 0}
    db.rpc.assert_not_called()


async def test_reclaim_credits_stale_balance_race_maps_to_409_error(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch, seat={"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 100})
    db = MagicMock()
    db.rpc.return_value.execute.side_effect = RuntimeError("insufficient balance on source wallet (have 0, need 100)")

    with pytest.raises(service.SeatBalanceChangedError):
        await service.reclaim_credits(db, U1, ORG_ID, MEMBER_ID, 100, "key-6")


async def test_reclaim_credits_other_rpc_errors_propagate_unmapped(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    _patch_wallets(monkeypatch, seat={"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 100})
    db = MagicMock()
    db.rpc.return_value.execute.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await service.reclaim_credits(db, U1, ORG_ID, MEMBER_ID, 100, "key-6")


# ---------------------------------------------------------------------------
# Router: POST /orgs/{org_id}/members/{member_id}/allocate
#         POST /orgs/{org_id}/members/{member_id}/reclaim
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _licensing_on_by_default(monkeypatch):
    monkeypatch.setenv("LICENSING_ENABLED", "true")


def test_allocate_router_ok_relays_rpc_result_and_fires_analytics(client):
    with (
        patch(
            "orgs.router.service.allocate_credits",
            new=AsyncMock(return_value={"duplicate": False, "from_balance": 900, "to_balance": 100}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(
            f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 100, "idempotency_key": "k1"}
        )
    assert resp.status_code == 200
    assert resp.json() == {"duplicate": False, "from_balance": 900, "to_balance": 100}
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "org_credits_allocated"


def test_allocate_router_duplicate_replay_skips_analytics(client):
    with (
        patch("orgs.router.service.allocate_credits", new=AsyncMock(return_value={"duplicate": True})),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(
            f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 100, "idempotency_key": "k1"}
        )
    assert resp.status_code == 200
    assert resp.json() == {"duplicate": True}
    mock_capture.assert_not_called()


def test_allocate_router_insufficient_pool_409(client):
    from orgs.service import PoolBalanceInsufficientError

    with patch(
        "orgs.router.service.allocate_credits",
        new=AsyncMock(side_effect=PoolBalanceInsufficientError("The pool doesn't have enough credits.")),
    ):
        resp = client.post(
            f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 100, "idempotency_key": "k1"}
        )
    assert resp.status_code == 409
    assert resp.json()["detail"] == "The pool doesn't have enough credits."


def test_allocate_router_rejects_zero_amount_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 0, "idempotency_key": "k1"})
    assert resp.status_code == 422


def test_allocate_router_rejects_amount_over_limit_422(client):
    resp = client.post(
        f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 1_000_001, "idempotency_key": "k1"}
    )
    assert resp.status_code == 422


def test_allocate_router_rejects_empty_idempotency_key_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 100, "idempotency_key": ""})
    assert resp.status_code == 422


def test_allocate_router_requires_idempotency_key_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 100})
    assert resp.status_code == 422


def test_allocate_router_404_when_licensing_off(client, monkeypatch):
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/allocate", json={"amount": 100, "idempotency_key": "k1"})
    assert resp.status_code == 404


def test_reclaim_router_ok_relays_rpc_result_and_fires_analytics(client):
    with (
        patch(
            "orgs.router.service.reclaim_credits",
            new=AsyncMock(return_value={"duplicate": False, "from_balance": 0, "to_balance": 1000}),
        ),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": 100, "idempotency_key": "k1"})
    assert resp.status_code == 200
    assert resp.json() == {"duplicate": False, "from_balance": 0, "to_balance": 1000}
    mock_capture.assert_called_once()
    assert mock_capture.call_args.args[1] == "org_credits_reclaimed"


def test_reclaim_router_duplicate_replay_skips_analytics(client):
    with (
        patch("orgs.router.service.reclaim_credits", new=AsyncMock(return_value={"duplicate": True})),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": 100, "idempotency_key": "k1"})
    assert resp.status_code == 200
    mock_capture.assert_not_called()


def test_reclaim_router_no_op_response_relayed_and_skips_analytics(client):
    """KEY TEST: the {"removed": 0} no-op path (reclaim-all against a
    non-positive balance) must relay verbatim and must NOT fire analytics —
    nothing moved, there is nothing to log."""
    with (
        patch("orgs.router.service.reclaim_credits", new=AsyncMock(return_value={"removed": 0})),
        patch("orgs.router.analytics_capture") as mock_capture,
    ):
        resp = client.post(
            f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": None, "idempotency_key": "k1"}
        )
    assert resp.status_code == 200
    assert resp.json() == {"removed": 0}
    mock_capture.assert_not_called()


def test_reclaim_router_stale_balance_409(client):
    from orgs.service import SeatBalanceChangedError

    with patch(
        "orgs.router.service.reclaim_credits",
        new=AsyncMock(side_effect=SeatBalanceChangedError("Balance changed — refresh and retry.")),
    ):
        resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": 100, "idempotency_key": "k1"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Balance changed — refresh and retry."


def test_reclaim_router_accepts_null_amount_and_forwards_none(client):
    with patch("orgs.router.service.reclaim_credits", new=AsyncMock(return_value={"removed": 0})) as mock_reclaim:
        resp = client.post(
            f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": None, "idempotency_key": "k1"}
        )
    assert resp.status_code == 200
    assert mock_reclaim.call_args.args[-2] is None  # amount forwarded as None (reclaim-all)


def test_reclaim_router_rejects_zero_amount_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": 0, "idempotency_key": "k1"})
    assert resp.status_code == 422


def test_reclaim_router_rejects_negative_amount_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": -5, "idempotency_key": "k1"})
    assert resp.status_code == 422


def test_reclaim_router_rejects_amount_over_limit_422(client):
    resp = client.post(
        f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": 1_000_001, "idempotency_key": "k1"}
    )
    assert resp.status_code == 422


def test_reclaim_router_requires_idempotency_key_422(client):
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": 100})
    assert resp.status_code == 422


def test_reclaim_router_404_when_licensing_off(client, monkeypatch):
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)
    resp = client.post(f"/orgs/{ORG_ID}/members/{MEMBER_ID}/reclaim", json={"amount": 100, "idempotency_key": "k1"})
    assert resp.status_code == 404
