"""Tests for the payout write side of the ledger reconciliation:

  - Same-currency credit netting at payout creation (excess PAID coverage
    re-allocates onto owed buckets, attributed to the ORIGINAL payout).
  - Staleness gate: dead-source lines block create_payouts unless force=True.
  - Revert guard: a payout whose coverage was moved as credit can't be reverted.
  - split_payee locks reassigned lines (and chained splits keep the original
    locked_party_key).

All tests run against the in-memory Supabase fake — no real DB or network.
fx.convert needs no patching: every conversion here is same-currency and
short-circuits before touching the DB.
"""

from unittest.mock import patch

import pytest

from oneclick.royalties import service
from tests.fake_supabase import USER, FakeDB

PAYEE = "payee-1"
PAID_PAYOUT = "PO-1"

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _payee(db, pid=PAYEE, name="alice", ccy="USD"):
    row = {
        "id": pid,
        "user_id": USER,
        "display_name": name.title(),
        "normalized_name": name,
        "payout_currency": ccy,
        "registry_user_id": None,
        "email": None,
    }
    db.tables["royalty_payees"].rows.append(row)
    return row


def _line(
    db,
    line_id,
    payee=PAYEE,
    stmt="S1",
    amount=5.0,
    ccy="USD",
    project="proj-1",
    song="Home",
    sources=(("F1", "contract.pdf", "h1"),),
    locked=False,
    locked_party=None,
):
    row = {
        "id": line_id,
        "user_id": USER,
        "payee_id": payee,
        "royalty_statement_id": stmt,
        "calculation_id": None,
        "project_id": project,
        "song_title": song,
        "role": "Writer",
        "royalty_type": "streaming",
        "percentage": 20.0,
        "amount_owed": amount,
        "statement_currency": ccy,
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
        "payee_locked": locked,
        "locked_party_key": locked_party,
        "source_contracts": [{"id": i, "name": n, "hash": h} for i, n, h in sources],
    }
    db.tables["royalty_lines"].rows.append(row)
    return row


def _payout(db, pid=PAID_PAYOUT, status="paid", payee=PAYEE, method="manual"):
    row = {
        "id": pid,
        "user_id": USER,
        "payee_id": payee,
        "status": status,
        "pay_currency": "USD",
        "total_amount": 0.0,
        "payment_method": method,
        "paid_at": None,
        "breakdown_snapshot": {},
        "idempotency_key": None,
    }
    db.tables["royalty_payouts"].rows.append(row)
    return row


def _coverage(db, cov_id, stmt, amount, payout=PAID_PAYOUT, payee=PAYEE, project="proj-1", moved_from=None):
    row = {
        "id": cov_id,
        "payout_id": payout,
        "payee_id": payee,
        "project_id": project,
        "royalty_statement_id": stmt,
        "covered_amount": amount,
        "moved_from": moved_from,
    }
    db.tables["royalty_payout_coverage"].rows.append(row)
    return row


def _file(db, fid="F1", project="proj-1", content_hash="h1"):
    db.tables["project_files"].rows.append(
        {"id": fid, "project_id": project, "file_name": f"{fid}.pdf", "content_hash": content_hash}
    )


def _create(db, force=False, key=None):
    # FakeDB has no "projects" table; name lookup is snapshot cosmetics only.
    with patch.object(service, "_project_name_map", return_value={}):
        return service.create_payouts(db, USER, [PAYEE], key, None, force=force)


def _cov_rows(db, **match):
    return [r for r in db.tables["royalty_payout_coverage"].rows if all(r.get(k) == v for k, v in match.items())]


# ---------------------------------------------------------------------------
# Credit netting
# ---------------------------------------------------------------------------


def test_credit_nets_payout_total():
    db = FakeDB()
    _payee(db)
    _file(db)
    # Overpaid bucket: earned 1.00, paid 1.50 → excess 0.50
    _line(db, "L-over", stmt="S-OVER", amount=1.0)
    _payout(db)
    _coverage(db, "cov-1", "S-OVER", 1.5)
    # Owed bucket: earned 5.00, uncovered
    _line(db, "L-owed", stmt="S-OWED", amount=5.0)

    res = _create(db)

    assert len(res) == 1
    assert res[0]["total_amount"] == pytest.approx(4.5)
    # Source coverage row reduced by the moved slice
    assert _cov_rows(db, id="cov-1")[0]["covered_amount"] == pytest.approx(1.0)
    # New coverage on the owed bucket, attributed to the ORIGINAL payout
    moved = _cov_rows(db, payout_id=PAID_PAYOUT, royalty_statement_id="S-OWED")
    assert len(moved) == 1
    assert moved[0]["covered_amount"] == pytest.approx(0.5)
    assert moved[0]["moved_from"] == {"statement_id": "S-OVER", "project_id": "proj-1", "action": "payout_credit"}
    # New draft payout covers only the remainder
    new_cov = _cov_rows(db, payout_id=res[0]["id"])
    assert len(new_cov) == 1
    assert new_cov[0]["covered_amount"] == pytest.approx(4.5)
    # One history row for the move
    assert [h["action"] for h in db.tables["royalty_ledger_history"].rows] == ["coverage_moved"]


def test_credit_merge_on_existing_target_row():
    db = FakeDB()
    _payee(db)
    _file(db)
    # Overpaid bucket: earned 1.00, paid 1.50 → excess 0.50
    _line(db, "L-over", stmt="S-OVER", amount=1.0)
    _payout(db)
    _coverage(db, "cov-1", "S-OVER", 1.5)
    # Owed bucket already partially covered by the SAME payout: earned 5, paid 2
    _line(db, "L-owed", stmt="S-OWED", amount=5.0)
    _coverage(db, "cov-2", "S-OWED", 2.0)

    res = _create(db)

    assert len(res) == 1
    assert res[0]["total_amount"] == pytest.approx(2.5)  # 5 - 2 paid - 0.5 credit
    # Composite PK: still exactly ONE row for (PO-1, S-OWED, proj-1), amount summed
    target = _cov_rows(db, payout_id=PAID_PAYOUT, royalty_statement_id="S-OWED", project_id="proj-1")
    assert len(target) == 1
    assert target[0]["id"] == "cov-2"
    assert target[0]["covered_amount"] == pytest.approx(2.5)
    assert target[0]["moved_from"] == {"statement_id": "S-OVER", "project_id": "proj-1", "action": "payout_credit"}
    assert _cov_rows(db, id="cov-1")[0]["covered_amount"] == pytest.approx(1.0)


def test_full_credit_no_payout_row():
    db = FakeDB()
    _payee(db)
    _file(db)
    # Excess 0.50 fully covers the 0.40 owed
    _line(db, "L-over", stmt="S-OVER", amount=1.0)
    _payout(db)
    _coverage(db, "cov-1", "S-OVER", 1.5)
    _line(db, "L-owed", stmt="S-OWED", amount=0.4)

    res = _create(db)

    assert res == []
    assert len(db.tables["royalty_payouts"].rows) == 1  # only the pre-existing paid payout
    # Moves still applied
    assert _cov_rows(db, id="cov-1")[0]["covered_amount"] == pytest.approx(1.1)
    moved = _cov_rows(db, payout_id=PAID_PAYOUT, royalty_statement_id="S-OWED")
    assert len(moved) == 1
    assert moved[0]["covered_amount"] == pytest.approx(0.4)
    assert moved[0]["moved_from"]["action"] == "payout_credit"
    # Bucket settles: nothing owed anymore
    assert service.payee_owed_buckets(db, USER, PAYEE) == []


def test_no_cross_currency_netting():
    db = FakeDB()
    _payee(db)
    _file(db)
    # EUR credit: earned 1.00 EUR, paid 2.00 EUR → excess 1.00 EUR
    _line(db, "L-eur", stmt="S-EUR", amount=1.0, ccy="EUR")
    _payout(db)
    _coverage(db, "cov-eur", "S-EUR", 2.0)
    # USD owed bucket
    _line(db, "L-usd", stmt="S-USD", amount=5.0)

    res = _create(db)

    assert len(res) == 1
    assert res[0]["total_amount"] == pytest.approx(5.0)  # full amount, no EUR netting
    assert _cov_rows(db, id="cov-eur")[0]["covered_amount"] == pytest.approx(2.0)  # untouched
    assert all(not r.get("moved_from") for r in db.tables["royalty_payout_coverage"].rows)
    assert db.tables["royalty_ledger_history"].rows == []


# ---------------------------------------------------------------------------
# Staleness gate
# ---------------------------------------------------------------------------


def test_stale_sources_blocks_unless_forced():
    db = FakeDB()
    _payee(db)
    # Only source id is dead and its hash matches no live file in the user's projects
    _line(db, "L1", stmt="S1", amount=5.0, sources=(("F-DEAD", "gone.pdf", "h-dead"),))
    # A live file with a DIFFERENT hash — must not count
    _file(db, fid="F-OTHER", content_hash="h-other")

    with pytest.raises(service.StaleSourcesError) as exc:
        _create(db)
    assert exc.value.lines == [{"song": "Home", "line_id": "L1"}]
    assert db.tables["royalty_payouts"].rows == []

    res = _create(db, force=True)
    assert len(res) == 1
    assert res[0]["total_amount"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Revert guard
# ---------------------------------------------------------------------------


def test_revert_blocked_after_move():
    db = FakeDB()
    _payout(db, "PO-moved")
    _coverage(
        db,
        "cov-m",
        "S1",
        1.0,
        payout="PO-moved",
        moved_from={"statement_id": "S0", "project_id": "proj-1", "action": "payout_credit"},
    )
    _payout(db, "PO-clean")
    _coverage(db, "cov-c", "S2", 1.0, payout="PO-clean")

    with pytest.raises(ValueError, match="revert is unavailable"):
        service.revert_payout_to_draft(db, USER, "PO-moved")

    reverted = service.revert_payout_to_draft(db, USER, "PO-clean")
    assert reverted["status"] == "draft"


# ---------------------------------------------------------------------------
# split_payee locking
# ---------------------------------------------------------------------------


def test_split_payee_locks_lines():
    db = FakeDB()
    _payee(db)
    _line(db, "L1", stmt="S1", amount=1.0)
    _line(db, "L2", stmt="S1", amount=2.0)
    _line(db, "L3", stmt="S1", amount=3.0)  # not selected

    target = service.split_payee(db, USER, PAYEE, ["L1", "L2"], "Bob")

    lines = {r["id"]: r for r in db.tables["royalty_lines"].rows}
    for lid in ("L1", "L2"):
        assert lines[lid]["payee_id"] == target["id"]
        assert lines[lid]["payee_locked"] is True
        assert lines[lid]["locked_party_key"] == "alice"
    assert lines["L3"]["payee_id"] == PAYEE
    assert not lines["L3"]["payee_locked"]
    assert lines["L3"]["locked_party_key"] is None


def test_chained_split_keeps_original_party_key():
    db = FakeDB()
    _payee(db, pid="payee-bob", name="bob")
    # Line landed on Bob via an earlier split from Alice — already locked
    _line(db, "L1", payee="payee-bob", stmt="S1", amount=1.0, locked=True, locked_party="alice")

    target = service.split_payee(db, USER, "payee-bob", ["L1"], "Carol")

    line = db.tables["royalty_lines"].rows[0]
    assert line["payee_id"] == target["id"]
    assert line["payee_locked"] is True
    assert line["locked_party_key"] == "alice"  # NOT overwritten with "bob"


def test_split_to_existing_payee_with_same_identity_rejected():
    """Splitting a line to an EXISTING payee who already holds a line with the
    same (statement, project, song, type) would violate the unique identity
    index — reject with a friendly ValueError, reassign nothing."""
    db = FakeDB()
    _payee(db)
    _payee(db, pid="payee-bob", name="bob")
    src = _line(db, "L1", stmt="S1", amount=1.0)
    existing = _line(db, "L2", payee="payee-bob", stmt="S1", amount=2.0)
    for row in (src, existing):
        row["song_key"] = "home"
        row["royalty_type_key"] = "streaming"

    with pytest.raises(ValueError, match="already has a royalty entry"):
        service.split_payee(db, USER, PAYEE, ["L1"], "Bob")

    lines = {r["id"]: r for r in db.tables["royalty_lines"].rows}
    assert lines["L1"]["payee_id"] == PAYEE and not lines["L1"]["payee_locked"]
    assert lines["L2"]["payee_id"] == "payee-bob"
