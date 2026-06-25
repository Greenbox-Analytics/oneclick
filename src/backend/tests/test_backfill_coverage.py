"""Tests for scripts/backfill_project_aware_coverage.py.

All tests mock the Supabase client — no real DB or network calls.

NOTE: these tests cover only the "clean" case where no prior partial coverage
existed for a statement.  If a statement has already been partially covered by
multiple payouts the proportional split is still an approximation — the true
per-project attribution is unknowable — and only the sum invariant is verified
here.
"""

from unittest.mock import MagicMock

import pytest

from scripts.backfill_project_aware_coverage import (
    _proportional_split,
    apply_backfill,
    find_affected_payouts,
    main,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PAYOUT_ID = "payout-po1"
PAYEE_ID = "payee-111"
STMT_ID = "stmt-s1"
PROJECT_A = "proj-aaa"
PROJECT_B = "proj-bbb"

# ---------------------------------------------------------------------------
# Minimal stateful mock DB for the backfill script
# ---------------------------------------------------------------------------


class MockDB:
    """Minimal mock of the Supabase service-role client for backfill tests.

    Tracks:
      deleted_coverage_filters  — list of filter dicts passed to delete chains
      inserted_coverage         — list of rows passed to coverage.insert()
    """

    def __init__(self, payouts=None, coverage=None, lines=None):
        self._payouts = payouts or []
        self._coverage = coverage or []
        self._lines = lines or []

        # Capture side-effects
        self.deleted_coverage_filters = []
        self.inserted_coverage = []

    def table(self, name):
        return _TableProxy(self, name)


class _TableProxy:
    """Fluent chainable mock proxy — mirrors the style in test_royalties_payouts.py."""

    def __init__(self, db: MockDB, name: str):
        self._db = db
        self._name = name
        self._filters: dict = {}
        self._pending_insert = None
        self._pending_delete = False

    # --- chainable methods ---------------------------------------------------

    def select(self, *a, **kw):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def neq(self, col, val):
        return self

    def order(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def insert(self, data):
        self._pending_insert = data
        return self

    def delete(self):
        self._pending_delete = True
        return self

    # --- execute -------------------------------------------------------------

    def execute(self):
        db = self._db
        name = self._name

        # INSERT
        if self._pending_insert is not None:
            if name == "royalty_payout_coverage":
                rows = self._pending_insert if isinstance(self._pending_insert, list) else [self._pending_insert]
                db.inserted_coverage.extend(rows)
                return MagicMock(data=rows)
            return MagicMock(data=[])

        # DELETE
        if self._pending_delete:
            if name == "royalty_payout_coverage":
                db.deleted_coverage_filters.append(dict(self._filters))
            return MagicMock(data=[])

        # SELECT — apply equality filters
        if name == "royalty_payouts":
            rows = db._payouts
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        if name == "royalty_payout_coverage":
            rows = db._coverage
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        if name == "royalty_lines":
            rows = db._lines
            for col, val in self._filters.items():
                rows = [r for r in rows if r.get(col) == val]
            return MagicMock(data=rows)

        return MagicMock(data=[])


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_payout(payout_id=PAYOUT_ID, payee_id=PAYEE_ID):
    return {"id": payout_id, "payee_id": payee_id}


def _make_coverage(payout_id=PAYOUT_ID, payee_id=PAYEE_ID, stmt_id=STMT_ID, project_id=PROJECT_A, covered_amount=140.0):
    return {
        "payout_id": payout_id,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "project_id": project_id,
        "covered_amount": covered_amount,
    }


def _make_line(payee_id=PAYEE_ID, stmt_id=STMT_ID, project_id=PROJECT_A, amount=100.0):
    return {
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "project_id": project_id,
        "amount_owed": amount,
    }


# ---------------------------------------------------------------------------
# Unit tests: _proportional_split
# ---------------------------------------------------------------------------


class TestProportionalSplit:
    def test_splits_proportionally(self):
        earned = {PROJECT_A: 100.0, PROJECT_B: 40.0}
        result = _proportional_split(140.0, earned)
        assert result[PROJECT_A] == pytest.approx(100.0)
        assert result[PROJECT_B] == pytest.approx(40.0)

    def test_sum_preserved_exactly(self):
        earned = {PROJECT_A: 100.0, PROJECT_B: 40.0}
        result = _proportional_split(140.0, earned)
        assert sum(result.values()) == pytest.approx(140.0)

    def test_zero_total_earned_equal_split(self):
        earned = {PROJECT_A: 0.0, PROJECT_B: 0.0}
        result = _proportional_split(100.0, earned)
        assert result[PROJECT_A] == pytest.approx(50.0)
        assert result[PROJECT_B] == pytest.approx(50.0)

    def test_empty_earned_returns_empty(self):
        result = _proportional_split(100.0, {})
        assert result == {}

    def test_single_project_gets_all(self):
        result = _proportional_split(77.5, {PROJECT_A: 200.0})
        assert result[PROJECT_A] == pytest.approx(77.5)


# ---------------------------------------------------------------------------
# Integration test: find_affected_payouts + apply_backfill
#
# Seed: payout po1 with a single coverage row (po1, s1, project=A, covered=140)
#       but royalty_lines for (payee, s1) = project A earned 100 + project B earned 40.
# ---------------------------------------------------------------------------


class TestBackfillMultiProject:
    def _make_db(self):
        return MockDB(
            payouts=[_make_payout()],
            coverage=[_make_coverage(covered_amount=140.0, project_id=PROJECT_A)],
            lines=[
                _make_line(project_id=PROJECT_A, amount=100.0),
                _make_line(project_id=PROJECT_B, amount=40.0),
            ],
        )

    def test_finds_one_affected_entry(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        assert len(affected) == 1

    def test_affected_entry_has_correct_payout_and_statement(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        entry = affected[0]
        assert entry["payout_id"] == PAYOUT_ID
        assert entry["statement_id"] == STMT_ID

    def test_proposed_rows_count(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        assert len(affected[0]["proposed_rows"]) == 2

    def test_apply_produces_two_coverage_rows(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)
        assert len(db.inserted_coverage) == 2

    def test_apply_project_a_covered_amount(self):
        """Project A earned 100 out of 140 total → should get 100."""
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)

        proj_a_rows = [r for r in db.inserted_coverage if r["project_id"] == PROJECT_A]
        assert len(proj_a_rows) == 1
        assert proj_a_rows[0]["covered_amount"] == pytest.approx(100.0)

    def test_apply_project_b_covered_amount(self):
        """Project B earned 40 out of 140 total → should get 40."""
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)

        proj_b_rows = [r for r in db.inserted_coverage if r["project_id"] == PROJECT_B]
        assert len(proj_b_rows) == 1
        assert proj_b_rows[0]["covered_amount"] == pytest.approx(40.0)

    def test_sum_of_new_rows_equals_original_covered_total(self):
        """Critical: the per-(payout, statement) covered total must be preserved."""
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)

        total = sum(r["covered_amount"] for r in db.inserted_coverage)
        assert total == pytest.approx(140.0)

    def test_old_coverage_row_is_deleted(self):
        """The stale single-row must be deleted before the new rows are inserted."""
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)

        assert len(db.deleted_coverage_filters) == 1
        deleted_filter = db.deleted_coverage_filters[0]
        assert deleted_filter.get("payout_id") == PAYOUT_ID
        assert deleted_filter.get("royalty_statement_id") == STMT_ID

    def test_new_rows_carry_correct_payee_id(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)

        for row in db.inserted_coverage:
            assert row["payee_id"] == PAYEE_ID

    def test_new_rows_carry_correct_payout_id(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)

        for row in db.inserted_coverage:
            assert row["payout_id"] == PAYOUT_ID


# ---------------------------------------------------------------------------
# No-op test: single-project statement → nothing to backfill
# ---------------------------------------------------------------------------


class TestNoOpSingleProject:
    def _make_db(self):
        """Statement s1 has lines only in project A — no multi-project split needed."""
        return MockDB(
            payouts=[_make_payout()],
            coverage=[_make_coverage(covered_amount=100.0, project_id=PROJECT_A)],
            lines=[
                # Only one project
                _make_line(project_id=PROJECT_A, amount=100.0),
            ],
        )

    def test_no_affected_payouts(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        assert affected == []

    def test_main_dry_run_prints_nothing_to_backfill(self, capsys):
        """main() with --dry-run on a single-project setup prints the zero-results message."""
        db = self._make_db()

        # Patch _get_supabase to return our mock db
        import unittest.mock as mock

        with mock.patch("scripts.backfill_project_aware_coverage._get_supabase", return_value=db):
            rc = main(["--dry-run"])

        out = capsys.readouterr().out
        assert rc == 0
        assert "0 mixed-project payouts" in out

    def test_no_db_writes_on_single_project(self):
        db = self._make_db()
        affected = find_affected_payouts(db)
        apply_backfill(db, affected)

        assert db.inserted_coverage == []
        assert db.deleted_coverage_filters == []


# ---------------------------------------------------------------------------
# Dry-run safety tests
# ---------------------------------------------------------------------------


class TestDryRun:
    def _make_mixed_db(self):
        return MockDB(
            payouts=[_make_payout()],
            coverage=[_make_coverage(covered_amount=140.0, project_id=PROJECT_A)],
            lines=[
                _make_line(project_id=PROJECT_A, amount=100.0),
                _make_line(project_id=PROJECT_B, amount=40.0),
            ],
        )

    def test_dry_run_no_db_writes(self):
        """--dry-run must not insert or delete any rows."""
        db = self._make_mixed_db()

        import unittest.mock as mock

        with mock.patch("scripts.backfill_project_aware_coverage._get_supabase", return_value=db):
            rc = main(["--dry-run"])

        assert rc == 0
        assert db.inserted_coverage == []
        assert db.deleted_coverage_filters == []

    def test_no_yes_flag_also_safe(self):
        """Running without --yes (and without --dry-run) must also be read-only."""
        db = self._make_mixed_db()

        import unittest.mock as mock

        with mock.patch("scripts.backfill_project_aware_coverage._get_supabase", return_value=db):
            rc = main([])  # no flags at all

        assert rc == 0
        assert db.inserted_coverage == []
        assert db.deleted_coverage_filters == []

    def test_yes_flag_applies_changes(self):
        """--yes must actually apply the backfill."""
        db = self._make_mixed_db()

        import unittest.mock as mock

        with mock.patch("scripts.backfill_project_aware_coverage._get_supabase", return_value=db):
            rc = main(["--yes"])

        assert rc == 0
        assert len(db.inserted_coverage) == 2
        assert len(db.deleted_coverage_filters) == 1
