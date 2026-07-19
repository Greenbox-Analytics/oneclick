"""Tests for payout functions in oneclick/royalties/service.py.

All tests mock the Supabase client and monkeypatch fx.convert — no real DB or
network calls.

Critical money invariants verified explicitly:
  1. coverage.covered_amount == bucket.owed_b (STATEMENT currency, NOT pay-ccy)
  2. payout.total_amount == Σ fx.convert(owed_b) over all buckets (pay-ccy)
  3. Empty-owed payee → skipped, no insert
  4. Same (idempotency_key, payee_id) → returns existing payout, no second insert
  5. snapshot contains statement_total (from statement_meta) distinct from payee_subtotal_owed
  6. mark_paid / cancel_payout ownership and state-machine constraints
  7. list_payouts / get_payout ownership scoping + derived orphan_state
"""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import service

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

USER_ID = "user-aaa"
OTHER_USER_ID = "user-bbb"
PAYEE_ID = "payee-111"
PAYEE_ID_2 = "payee-222"
PAYOUT_ID = "payout-xyz"

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_payee(payee_id=PAYEE_ID, user_id=USER_ID, payout_ccy="USD", display_name="Alice"):
    return {
        "id": payee_id,
        "user_id": user_id,
        "display_name": display_name,
        "payout_currency": payout_ccy,
        "normalized_name": display_name.lower(),
        "registry_user_id": None,
        "email": None,
    }


def _make_line(
    payee_id=PAYEE_ID,
    stmt_id="stmt-1",
    amount=200.0,
    ccy="USD",
    project_id="proj-1",
    line_id="line-1",
    calc_id="calc-1",
    song="Song A",
    role="Writer",
    royalty_type="mechanical",
    percentage=50.0,
):
    return {
        "id": line_id,
        "user_id": USER_ID,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "calculation_id": calc_id,
        "project_id": project_id,
        "song_title": song,
        "role": role,
        "royalty_type": royalty_type,
        "percentage": percentage,
        "amount_owed": amount,
        "statement_currency": ccy,
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
    }


def _make_payout(
    payout_id=PAYOUT_ID,
    payee_id=PAYEE_ID,
    user_id=USER_ID,
    status="draft",
    total_amount=200.0,
    pay_currency="USD",
    idempotency_key=None,
    payment_method="manual",
):
    return {
        "id": payout_id,
        "user_id": user_id,
        "payee_id": payee_id,
        "status": status,
        "pay_currency": pay_currency,
        "total_amount": total_amount,
        "fx_rate_date": "2026-06-23",
        "created_at": "2026-06-23T00:00:00Z",
        "paid_at": None,
        "note": None,
        "breakdown_snapshot": {"projects": [{"project_id": "proj-1", "name": "Project One", "statements": []}]},
        "idempotency_key": idempotency_key,
        "payment_method": payment_method,
    }


def _make_coverage(payout_id=PAYOUT_ID, payee_id=PAYEE_ID, stmt_id="stmt-1", covered_amount=200.0, project_id="proj-1"):
    return {
        "payout_id": payout_id,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "covered_amount": covered_amount,
        "project_id": project_id,
    }


# ---------------------------------------------------------------------------
# Mock DB factory — stateful insert capture
# ---------------------------------------------------------------------------


class MockDB:
    """Minimal stateful mock of the Supabase client for payout tests.

    Tracks:
      inserted_payouts:   list of dicts passed to royalty_payouts.insert()
      inserted_coverage:  list of dicts passed to royalty_payout_coverage.insert()
      deleted_payouts:    list of payout ids passed to royalty_payouts.delete()
    """

    def __init__(
        self,
        payees=None,
        lines=None,
        payouts=None,
        coverage=None,
        projects=None,
        existing_idempotency=None,  # dict: derived_key -> payout dict
        statement_rows=None,
    ):
        self.payees = payees or []
        self.lines = lines or []
        self.payouts = payouts or []
        self.coverage = coverage or []
        self.projects = projects or [{"id": "proj-1", "name": "Project One"}]
        self.existing_idempotency = existing_idempotency or {}
        self.statement_rows = statement_rows or []

        # Capture side-effects
        self.inserted_payouts = []
        self.inserted_coverage = []
        self.deleted_payout_ids = []
        self.updated_payouts = []

        # Auto-assign payout ids on insert
        self._payout_id_counter = 0

    def table(self, name):
        return _TableProxy(self, name)


class _TableProxy:
    """Fluent mock proxy for db.table(name).select(...).eq(...).execute()."""

    def __init__(self, db: MockDB, name: str):
        self._db = db
        self._name = name
        self._filters: dict = {}
        self._pending_insert = None
        self._pending_update = None
        self._pending_delete = False

    # Chainable methods — return self
    def select(self, *a, **kw):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def neq(self, col, val):
        return self

    def in_(self, col, vals):
        return self

    def order(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def single(self):
        return self

    def maybe_single(self):
        return self

    def insert(self, data):
        self._pending_insert = data
        return self

    def update(self, data):
        self._pending_update = data
        return self

    def delete(self):
        self._pending_delete = True
        return self

    def upsert(self, data, **kw):
        return self

    def execute(self):
        db = self._db
        name = self._name

        # Handle inserts
        if self._pending_insert is not None:
            if name == "royalty_payouts":
                rows = self._pending_insert if isinstance(self._pending_insert, list) else [self._pending_insert]
                created = []
                for row in rows:
                    db._payout_id_counter += 1
                    row_with_id = {"id": f"payout-new-{db._payout_id_counter}", **row}
                    db.inserted_payouts.append(row_with_id)
                    created.append(row_with_id)
                return MagicMock(data=created)
            elif name == "royalty_payout_coverage":
                rows = self._pending_insert if isinstance(self._pending_insert, list) else [self._pending_insert]
                db.inserted_coverage.extend(rows)
                return MagicMock(data=rows)
            return MagicMock(data=[])

        # Handle updates
        if self._pending_update is not None:
            if name == "royalty_payouts":
                payout_id = self._filters.get("id")
                matched = [p for p in db.payouts if p.get("id") == payout_id]
                if matched:
                    updated = {**matched[0], **self._pending_update}
                    db.updated_payouts.append(updated)
                    return MagicMock(data=[updated])
            return MagicMock(data=[])

        # Handle deletes
        if self._pending_delete:
            if name == "royalty_payouts":
                payout_id = self._filters.get("id")
                db.deleted_payout_ids.append(payout_id)
            return MagicMock(data=[])

        # Handle selects — apply filters
        if name == "royalty_payees":
            rows = db.payees
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        elif name == "royalty_lines":
            rows = db.lines
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        elif name == "royalty_payouts":
            # idempotency_key query
            if "idempotency_key" in self._filters:
                ikey = self._filters["idempotency_key"]
                matched = db.existing_idempotency.get(ikey)
                if matched:
                    return MagicMock(data=[matched])
                return MagicMock(data=[])
            rows = db.payouts
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        elif name == "royalty_payout_coverage":
            rows = db.coverage
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        elif name == "projects":
            rows = db.projects
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        elif name == "royalty_statement_rows":
            rows = db.statement_rows
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        elif name in ("fx_rate_snapshots",):
            return MagicMock(data=[])

        return MagicMock(data=[])


# ---------------------------------------------------------------------------
# FX helpers
# ---------------------------------------------------------------------------


def _identity_fx(db, amount, frm, to, on="latest"):
    """1:1 for same-ccy; 2× for EUR→USD; 0.5× for USD→EUR."""
    if frm.upper() == to.upper():
        return amount
    if frm.upper() == "EUR" and to.upper() == "USD":
        return amount * 2.0
    if frm.upper() == "USD" and to.upper() == "EUR":
        return amount * 0.5
    return amount  # fallback identity


# ---------------------------------------------------------------------------
# Tests: payee_owed_buckets
# ---------------------------------------------------------------------------


class TestPayeeOwedBuckets:
    def test_returns_bucket_for_owed_amount(self):
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=200.0, ccy="USD")],
            payouts=[],
            coverage=[],
        )
        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)
        assert len(buckets) == 1
        b = buckets[0]
        assert b["owed_b"] == pytest.approx(200.0)
        assert b["ccy"] == "USD"
        assert b["royalty_statement_id"] == "stmt-1"

    def test_skips_fully_covered_bucket(self):
        payout = _make_payout(status="paid", total_amount=200.0)
        coverage = _make_coverage(covered_amount=200.0)
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=200.0, ccy="USD")],
            payouts=[payout],
            coverage=[coverage],
        )
        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)
        assert buckets == []

    def test_partial_coverage_returns_remainder(self):
        payout = _make_payout(status="paid", total_amount=100.0)
        coverage = _make_coverage(covered_amount=100.0)
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=200.0, ccy="USD")],
            payouts=[payout],
            coverage=[coverage],
        )
        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)
        assert len(buckets) == 1
        assert buckets[0]["owed_b"] == pytest.approx(100.0)

    def test_empty_lines_returns_empty(self):
        db = MockDB(payees=[_make_payee()], lines=[], payouts=[], coverage=[])
        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)
        assert buckets == []

    def test_over_covered_bucket_clamped_to_zero(self):
        payout = _make_payout(status="paid", total_amount=300.0)
        coverage = _make_coverage(covered_amount=300.0)
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=200.0)],
            payouts=[payout],
            coverage=[coverage],
        )
        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)
        assert buckets == []


# ---------------------------------------------------------------------------
# Tests: create_payouts — critical money invariants
# ---------------------------------------------------------------------------


class TestCreatePayouts:
    def _db_with_one_owed_bucket(self, payout_ccy="USD", owed_b=200.0, ccy="USD"):
        return MockDB(
            payees=[_make_payee(payout_ccy=payout_ccy)],
            lines=[_make_line(amount=owed_b, ccy=ccy)],
            payouts=[],
            coverage=[],
            statement_rows=[
                {"sale_date": "2025-01-01", "net_payable": 500.0, "calculation_id": "calc-1"},
            ],
        )

    def test_creates_one_payout_for_one_payee(self):
        db = self._db_with_one_owed_bucket()
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

        assert len(result) == 1
        assert len(db.inserted_payouts) == 1

    def test_payout_status_is_draft(self):
        db = self._db_with_one_owed_bucket()
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)
        assert result[0]["status"] == "draft"

    def test_coverage_covered_amount_is_statement_ccy_not_pay_ccy(self):
        """THE critical money invariant: covered_amount must be owed_b (statement ccy).

        Setup: owed_b = 200 EUR, payout_ccy = USD.
        fx: EUR→USD = ×2, so pay_amt = 400 USD.
        covered_amount MUST be 200 (EUR), NOT 400 (USD).
        """
        db = self._db_with_one_owed_bucket(payout_ccy="USD", owed_b=200.0, ccy="EUR")
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

        assert len(db.inserted_coverage) == 1
        cov = db.inserted_coverage[0]
        # Statement currency is EUR, owed_b = 200 EUR
        assert cov["covered_amount"] == pytest.approx(200.0), (
            f"covered_amount must be 200.0 (EUR, statement ccy), got {cov['covered_amount']} "
            f"— must NOT be 400.0 (USD, pay-ccy)"
        )

    def test_total_amount_is_converted_pay_ccy(self):
        """total_amount must be the sum of fx.convert(owed_b) for each bucket."""
        # owed_b = 200 EUR, payout = USD, fx = ×2 → total = 400 USD
        db = self._db_with_one_owed_bucket(payout_ccy="USD", owed_b=200.0, ccy="EUR")
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)
        assert result[0]["total_amount"] == pytest.approx(400.0)

    def test_coverage_and_total_differ_when_cross_currency(self):
        """covered_amount != total_amount when ccy != payout_ccy."""
        db = self._db_with_one_owed_bucket(payout_ccy="USD", owed_b=200.0, ccy="EUR")
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

        cov = db.inserted_coverage[0]
        assert cov["covered_amount"] != pytest.approx(result[0]["total_amount"])

    def test_empty_owed_payee_is_skipped(self):
        """A payee with no owed buckets must not generate any payout or coverage."""
        payout = _make_payout(status="paid", total_amount=200.0)
        coverage = _make_coverage(covered_amount=200.0)
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=200.0)],
            payouts=[payout],
            coverage=[coverage],
        )
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

        assert result == []
        assert db.inserted_payouts == []
        assert db.inserted_coverage == []

    def test_idempotency_returns_existing_no_second_insert(self):
        """Same (idempotency_key, payee_id) → return existing, no new insert."""
        derived_key = "my-key:payee-111"
        existing_payout = _make_payout(idempotency_key=derived_key)
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=200.0)],
            payouts=[],
            coverage=[],
            existing_idempotency={derived_key: existing_payout},
        )
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], "my-key", None)

        assert len(result) == 1
        assert result[0]["idempotency_key"] == derived_key
        assert db.inserted_payouts == [], "No new payout must be inserted on idempotent replay"
        assert db.inserted_coverage == [], "No new coverage must be inserted on idempotent replay"

    def test_idempotency_key_is_derived_per_payee(self):
        """derived_key = f'{client_key}:{payee_id}' — each payee gets its own key."""
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=100.0)],
            payouts=[],
            coverage=[],
        )
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            service.create_payouts(db, USER_ID, [PAYEE_ID], "client-key", None)

        assert len(db.inserted_payouts) == 1
        assert db.inserted_payouts[0]["idempotency_key"] == f"client-key:{PAYEE_ID}"

    def test_no_idempotency_key_stores_none(self):
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=100.0)],
            payouts=[],
            coverage=[],
        )
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

        assert "idempotency_key" not in db.inserted_payouts[0]

    def test_snapshot_statement_total_differs_from_payee_subtotal(self):
        """statement_total (net from statement_meta) must differ from payee_subtotal_owed.

        statement_meta returns the FULL statement net (e.g. 500 USD).
        payee_subtotal_owed is just the payee's portion (e.g. 200 USD).
        They must be different — this verifies the snapshot captures both correctly.
        """
        db = MockDB(
            payees=[_make_payee()],
            lines=[_make_line(amount=200.0, calc_id="calc-1")],
            payouts=[],
            coverage=[],
            # statement_meta will sum net_payable → 500
            statement_rows=[
                {"sale_date": "2025-01-01", "net_payable": 300.0, "calculation_id": "calc-1"},
                {"sale_date": "2025-02-01", "net_payable": 200.0, "calculation_id": "calc-1"},
            ],
        )
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

        payout = result[0]
        snapshot = payout["breakdown_snapshot"]
        stmt = snapshot["projects"][0]["statements"][0]

        assert stmt["statement_total"] == pytest.approx(500.0), (
            f"statement_total should be 500 (sum of net_payable), got {stmt['statement_total']}"
        )
        assert stmt["payee_subtotal_owed"] == pytest.approx(200.0), (
            f"payee_subtotal_owed should be 200 (the bucket owed_b), got {stmt['payee_subtotal_owed']}"
        )
        assert stmt["statement_total"] != pytest.approx(stmt["payee_subtotal_owed"]), (
            "statement_total and payee_subtotal_owed must differ (full statement vs payee slice)"
        )

    def test_snapshot_payee_subtotal_pay_ccy_is_converted(self):
        """payee_subtotal_pay_ccy must be the converted value, not owed_b."""
        db = MockDB(
            payees=[_make_payee(payout_ccy="USD")],
            lines=[_make_line(amount=200.0, ccy="EUR", calc_id="calc-1")],
            payouts=[],
            coverage=[],
        )
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

        stmt = result[0]["breakdown_snapshot"]["projects"][0]["statements"][0]
        # EUR→USD ×2, so pay_ccy = 400
        assert stmt["payee_subtotal_pay_ccy"] == pytest.approx(400.0)
        assert stmt["payee_subtotal_owed"] == pytest.approx(200.0)  # statement ccy unchanged

    def test_ownership_error_for_unknown_payee(self):
        """Payee not owned by caller → PermissionError."""
        db = MockDB(payees=[], lines=[], payouts=[], coverage=[])
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx), pytest.raises(PermissionError):
            service.create_payouts(db, USER_ID, [PAYEE_ID], None, None)

    def test_multiple_payees_each_get_own_payout(self):
        """Two payees with owed amounts each get their own payout row."""
        db = MockDB(
            payees=[
                _make_payee(payee_id=PAYEE_ID, payout_ccy="USD"),
                _make_payee(payee_id=PAYEE_ID_2, payout_ccy="USD", display_name="Bob"),
            ],
            lines=[
                _make_line(payee_id=PAYEE_ID, stmt_id="stmt-1", line_id="l1", amount=100.0),
                _make_line(payee_id=PAYEE_ID_2, stmt_id="stmt-2", line_id="l2", amount=150.0),
            ],
            payouts=[],
            coverage=[],
        )
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.create_payouts(db, USER_ID, [PAYEE_ID, PAYEE_ID_2], None, None)

        assert len(result) == 2
        assert len(db.inserted_payouts) == 2
        assert len(db.inserted_coverage) == 2


# ---------------------------------------------------------------------------
# Tests: mark_paid
# ---------------------------------------------------------------------------


class TestMarkPaid:
    def test_sets_status_to_paid(self):
        payout = _make_payout(status="draft")
        db = MockDB(payouts=[payout])
        result = service.mark_paid(db, USER_ID, PAYOUT_ID)
        assert result["status"] == "paid"

    def test_sets_paid_at(self):
        payout = _make_payout(status="draft")
        db = MockDB(payouts=[payout])
        result = service.mark_paid(db, USER_ID, PAYOUT_ID)
        assert result.get("paid_at") is not None

    def test_wrong_user_raises_permission_error(self):
        payout = _make_payout(user_id=USER_ID)
        db = MockDB(payouts=[payout])
        with pytest.raises(PermissionError):
            service.mark_paid(db, OTHER_USER_ID, PAYOUT_ID)

    def test_nonexistent_payout_raises_permission_error(self):
        db = MockDB(payouts=[])
        with pytest.raises(PermissionError):
            service.mark_paid(db, USER_ID, "nonexistent-id")


# ---------------------------------------------------------------------------
# Tests: cancel_payout
# ---------------------------------------------------------------------------


class TestCancelPayout:
    def test_deletes_draft_payout(self):
        payout = _make_payout(status="draft")
        db = MockDB(payouts=[payout])
        service.cancel_payout(db, USER_ID, PAYOUT_ID)
        assert PAYOUT_ID in db.deleted_payout_ids

    def test_non_draft_raises_value_error(self):
        payout = _make_payout(status="paid")
        db = MockDB(payouts=[payout])
        with pytest.raises(ValueError, match="only drafts can be canceled"):
            service.cancel_payout(db, USER_ID, PAYOUT_ID)

    def test_wrong_user_raises_permission_error(self):
        payout = _make_payout(user_id=USER_ID)
        db = MockDB(payouts=[payout])
        with pytest.raises(PermissionError):
            service.cancel_payout(db, OTHER_USER_ID, PAYOUT_ID)

    def test_nonexistent_payout_raises_permission_error(self):
        db = MockDB(payouts=[])
        with pytest.raises(PermissionError):
            service.cancel_payout(db, USER_ID, "nonexistent-id")


# ---------------------------------------------------------------------------
# Tests: revert_payout_to_draft — undo an accidental mark-paid
# ---------------------------------------------------------------------------


class TestRevertPayout:
    def test_reverts_manual_paid_to_draft(self):
        payout = _make_payout(status="paid", payment_method="manual")
        db = MockDB(payouts=[payout])
        result = service.revert_payout_to_draft(db, USER_ID, PAYOUT_ID)
        assert result["status"] == "draft"

    def test_clears_paid_at(self):
        payout = _make_payout(status="paid", payment_method="manual")
        payout["paid_at"] = "2026-06-23T00:00:00Z"
        db = MockDB(payouts=[payout])
        result = service.revert_payout_to_draft(db, USER_ID, PAYOUT_ID)
        assert result.get("paid_at") is None

    def test_non_paid_raises_value_error(self):
        payout = _make_payout(status="draft")
        db = MockDB(payouts=[payout])
        with pytest.raises(ValueError, match="Only completed payouts"):
            service.revert_payout_to_draft(db, USER_ID, PAYOUT_ID)

    def test_paypal_payout_cannot_be_reverted(self):
        payout = _make_payout(status="paid", payment_method="paypal")
        db = MockDB(payouts=[payout])
        with pytest.raises(ValueError, match="paid through PayPal"):
            service.revert_payout_to_draft(db, USER_ID, PAYOUT_ID)

    def test_wrong_user_raises_permission_error(self):
        payout = _make_payout(status="paid", user_id=USER_ID)
        db = MockDB(payouts=[payout])
        with pytest.raises(PermissionError):
            service.revert_payout_to_draft(db, OTHER_USER_ID, PAYOUT_ID)

    def test_nonexistent_payout_raises_permission_error(self):
        db = MockDB(payouts=[])
        with pytest.raises(PermissionError):
            service.revert_payout_to_draft(db, USER_ID, "nonexistent-id")


# ---------------------------------------------------------------------------
# Tests: list_payouts / get_payout — ownership scoping
# ---------------------------------------------------------------------------


class TestListPayouts:
    def test_returns_only_caller_payouts(self):
        """list_payouts must only return the caller's payouts."""
        p1 = _make_payout(payout_id="p1", user_id=USER_ID)
        db = MockDB(payouts=[p1], coverage=[])
        result = service.list_payouts(db, USER_ID)
        assert len(result) == 1
        assert result[0]["id"] == "p1"

    def test_attaches_orphan_state(self):
        """list_payouts must include an orphan_state key on each payout."""
        p = _make_payout()
        db = MockDB(payouts=[p], coverage=[])
        result = service.list_payouts(db, USER_ID)
        assert "orphan_state" in result[0]


class TestGetPayout:
    def test_returns_payout_for_correct_owner(self):
        p = _make_payout()
        db = MockDB(payouts=[p], coverage=[])
        result = service.get_payout(db, USER_ID, PAYOUT_ID)
        assert result["id"] == PAYOUT_ID

    def test_raises_permission_error_for_wrong_user(self):
        """Payout not scoped to OTHER_USER_ID → PermissionError (→ 404)."""
        p = _make_payout(user_id=USER_ID)
        # The mock filters by user_id=OTHER_USER_ID, so it returns nothing
        db = MockDB(payouts=[p], coverage=[])
        with pytest.raises(PermissionError):
            service.get_payout(db, OTHER_USER_ID, PAYOUT_ID)

    def test_raises_permission_error_for_nonexistent(self):
        db = MockDB(payouts=[], coverage=[])
        with pytest.raises(PermissionError):
            service.get_payout(db, USER_ID, "no-such-id")

    def test_attaches_orphan_state(self):
        p = _make_payout()
        db = MockDB(payouts=[p], coverage=[])
        result = service.get_payout(db, USER_ID, PAYOUT_ID)
        assert "orphan_state" in result


# ---------------------------------------------------------------------------
# Tests: orphan_state derivation
# ---------------------------------------------------------------------------


class TestOrphanState:
    def test_no_coverage_returns_orphaned(self):
        """When coverage is empty for a payout, orphan_state must be 'orphaned'."""
        p = _make_payout()
        db = MockDB(payouts=[p], coverage=[])
        result = service.get_payout(db, USER_ID, PAYOUT_ID)
        assert result["orphan_state"] == "orphaned"

    def test_full_coverage_returns_none(self):
        """When all snapshot project_ids are covered, orphan_state = 'none'."""
        p = _make_payout()
        cov = _make_coverage()  # covers proj-1 which is in snapshot
        db = MockDB(payouts=[p], coverage=[cov])
        result = service.get_payout(db, USER_ID, PAYOUT_ID)
        assert result["orphan_state"] == "none"

    def test_partial_coverage_returns_partial(self):
        """Snapshot lists two projects, coverage only for one → 'partial'."""
        p = _make_payout()
        # Override snapshot to reference two projects
        p["breakdown_snapshot"]["projects"].append({"project_id": "proj-2", "name": "P2", "statements": []})
        cov = _make_coverage(project_id="proj-1")  # only covers proj-1
        db = MockDB(payouts=[p], coverage=[cov])
        result = service.get_payout(db, USER_ID, PAYOUT_ID)
        assert result["orphan_state"] == "partial"


# ---------------------------------------------------------------------------
# Tests: router HTTP layer (smoke tests via TestClient)
# ---------------------------------------------------------------------------


def _make_router_client(service_mock_target=None, service_mock_side_effect=None, service_mock_return=None):
    """Build a minimal FastAPI app with the royalties router, auth/gating bypassed.

    Uses dependency_overrides (the FastAPI-idiomatic approach) rather than
    patching module symbols, which don't work for Depends()-resolved callables.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth import get_current_user_id
    from oneclick.royalties.router import router

    app = FastAPI()
    app.include_router(router)

    # Override auth dependency
    async def _mock_user_id():
        return USER_ID

    app.dependency_overrides[get_current_user_id] = _mock_user_id

    # Patch gated_feature so it's a no-op
    gating_patcher = patch("oneclick.royalties.router.gated_feature", return_value=None)
    # Patch _get_supabase so it returns a trivial mock
    supabase_patcher = patch("oneclick.royalties.router._get_supabase", return_value=MagicMock())

    service_patcher = None
    if service_mock_target:
        if service_mock_side_effect is not None:
            service_patcher = patch(service_mock_target, side_effect=service_mock_side_effect)
        else:
            service_patcher = patch(
                service_mock_target, return_value=service_mock_return if service_mock_return is not None else []
            )

    gating_patcher.start()
    supabase_patcher.start()
    if service_patcher:
        service_patcher.start()

    client = TestClient(app, raise_server_exceptions=False)

    yield client

    gating_patcher.stop()
    supabase_patcher.stop()
    if service_patcher:
        service_patcher.stop()


class TestPayoutsRouterSmoke:
    """Verify HTTP status codes — use FastAPI dependency_overrides for auth/gating."""

    def test_post_payouts_200(self):
        for client in _make_router_client("oneclick.royalties.service.create_payouts", service_mock_return=[]):
            resp = client.post("/payouts", json={"payee_ids": [PAYEE_ID]})
            assert resp.status_code == 200

    def test_post_payouts_403_on_permission_error(self):
        for client in _make_router_client(
            "oneclick.royalties.service.create_payouts", service_mock_side_effect=PermissionError("denied")
        ):
            resp = client.post("/payouts", json={"payee_ids": [PAYEE_ID]})
            assert resp.status_code == 403

    def test_post_cancel_409_on_value_error(self):
        for client in _make_router_client(
            "oneclick.royalties.service.cancel_payout",
            service_mock_side_effect=ValueError("only drafts can be canceled"),
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/cancel")
            assert resp.status_code == 409

    def test_post_cancel_403_on_permission_error(self):
        for client in _make_router_client(
            "oneclick.royalties.service.cancel_payout", service_mock_side_effect=PermissionError("not yours")
        ):
            resp = client.post(f"/payouts/{PAYOUT_ID}/cancel")
            assert resp.status_code == 403

    def test_get_payout_404_on_permission_error(self):
        for client in _make_router_client(
            "oneclick.royalties.service.get_payout", service_mock_side_effect=PermissionError("not found")
        ):
            resp = client.get(f"/payouts/{PAYOUT_ID}")
            assert resp.status_code == 404

    def test_get_payouts_200(self):
        for client in _make_router_client("oneclick.royalties.service.list_payouts", service_mock_return=[]):
            resp = client.get("/payouts")
            assert resp.status_code == 200
