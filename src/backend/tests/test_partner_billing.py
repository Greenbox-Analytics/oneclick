from unittest.mock import MagicMock

from partner_api import service as psvc

KEY_A = "00000000-0000-0000-0000-0000000000bb"
KEY_B = "00000000-0000-0000-0000-0000000000cc"
P1 = "2026-09-30T00:00:00+00:00"
P2 = "2026-10-31T00:00:00+00:00"


def test_request_id_deterministic_namespaced_and_payload_bound():
    a1 = psvc.derive_request_id(KEY_A, "retry-1", "fp-1", P1)
    a2 = psvc.derive_request_id(KEY_A, "retry-1", "fp-1", P1)
    b1 = psvc.derive_request_id(KEY_B, "retry-1", "fp-1", P1)
    new_payload = psvc.derive_request_id(KEY_A, "retry-1", "fp-2", P1)
    assert a1 == a2
    assert a1 != b1
    assert a1 != new_payload  # same header, different payload => pays again
    assert psvc.derive_request_id(KEY_A, None, "fp-1", P1) != psvc.derive_request_id(KEY_A, None, "fp-1", P1)


def test_request_id_is_scoped_to_the_billing_period():
    # The unique index on credit_ledger.request_id never expires, so without a
    # period term one pinned Idempotency-Key would ride the same free duplicate
    # forever. Same key + payload, NEW period => a new id, which pays again.
    assert psvc.derive_request_id(KEY_A, "nightly", "fp-1", P1) != psvc.derive_request_id(KEY_A, "nightly", "fp-1", P2)


def test_request_id_without_a_period_is_still_stable():
    # A pool wallet whose first dispersal has not landed has no period_end.
    a1 = psvc.derive_request_id(KEY_A, "retry-1", "fp-1", None)
    assert a1 == psvc.derive_request_id(KEY_A, "retry-1", "fp-1", None)
    assert a1 != psvc.derive_request_id(KEY_A, "retry-1", "fp-1", P1)


def test_get_price_reads_the_partner_action_row():
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{"credits": 30}]
    assert psvc.get_price(sb, psvc.ONECLICK_ACTION) == 30
    assert sb.table.return_value.select.return_value.eq.call_args[0] == ("action", "partner_oneclick_run")


def test_check_pool_compares_against_price(monkeypatch):
    monkeypatch.setattr(
        "orgs.wallets.read_or_create_org_wallet",
        lambda sb, org_id: {"id": "w1", "bundle_balance": 20, "reserve_balance": 0},
    )
    assert psvc.check_pool(MagicMock(), "org1", 21)["ok"] is False
    monkeypatch.setattr(
        "orgs.wallets.read_or_create_org_wallet",
        lambda sb, org_id: {"id": "w1", "bundle_balance": 20, "reserve_balance": 1, "period_end": P1},
    )
    pool = psvc.check_pool(MagicMock(), "org1", 21)
    assert pool["ok"] is True
    # period_end rides along so the caller can scope the dedupe id to the period.
    assert pool["period_end"] == P1


def test_debit_run_is_direct_rpc_with_charge_and_attribution_metadata():
    sb = MagicMock()
    sb.rpc.return_value.execute.return_value.data = {"duplicate": True}
    out = psvc.debit_run(
        sb,
        wallet_id="w1",
        amount=37,
        request_id="r1",
        key_id=KEY_A,
        metadata={"base": 30, "tail_credits": 7, "metered": True},
    )
    assert out == {"duplicate": True}  # a caller can tell "charged" from "already charged"
    name, payload = sb.rpc.call_args[0]
    assert name == "debit_credits"
    assert payload["p_wallet_id"] == "w1"
    assert payload["p_amount"] == 37
    assert payload["p_action"] == "partner_oneclick_run"
    assert payload["p_kind"] == "debit"
    assert "p_member_id" not in payload
    assert payload["p_metadata"] == {
        "base": 30,
        "tail_credits": 7,
        "metered": True,
        "source": "partner_api",
        "partner_key_id": KEY_A,
    }


def test_attribution_keys_win_over_caller_metadata():
    # The charge metadata can never impersonate a different key/user.
    sb = MagicMock()
    psvc.debit_run(
        sb,
        wallet_id="w1",
        amount=30,
        request_id="r1",
        key_id=KEY_A,
        metadata={"source": "spoof", "partner_key_id": KEY_B},
    )
    meta = sb.rpc.call_args[0][1]["p_metadata"]
    assert meta["source"] == "partner_api" and meta["partner_key_id"] == KEY_A
