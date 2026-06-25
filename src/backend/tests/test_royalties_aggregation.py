"""Tests for oneclick/royalties/service.py aggregation logic.

All tests mock the Supabase client and monkeypatch fx.convert so no
real DB or network calls are made.

Critical invariant verified:
  owed is computed per-bucket in statement currency (clamped ≥ 0), then
  converted to the reporting base and summed.  Over-coverage in one
  bucket (S2) MUST NOT reduce the owed amount of another bucket (S1).
"""

from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import service

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

USER_ID = "user-111"
PAYEE_ID = "payee-aaa"


# ---------------------------------------------------------------------------
# Helpers to build mock DB responses
# ---------------------------------------------------------------------------


def _make_payee(payee_id=PAYEE_ID, payout_ccy="USD", normalized_name="alice"):
    return {
        "id": payee_id,
        "user_id": USER_ID,
        "display_name": "Alice",
        "payout_currency": payout_ccy,
        "registry_user_id": None,
        "email": None,
        "normalized_name": normalized_name,
    }


def _make_line(
    payee_id=PAYEE_ID,
    stmt_id="stmt-1",
    amount=100.0,
    ccy="USD",
    project_id="proj-1",
    line_id="line-1",
    calc_id="calc-1",
):
    return {
        "id": line_id,
        "user_id": USER_ID,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "calculation_id": calc_id,
        "project_id": project_id,
        "song_title": "Song A",
        "role": "Writer",
        "royalty_type": "mechanical",
        "percentage": 50.0,
        "amount_owed": amount,
        "statement_currency": ccy,
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
    }


def _make_payout(payout_id, payee_id=PAYEE_ID, status="paid", ccy="USD", amount=0.0):
    return {
        "id": payout_id,
        "user_id": USER_ID,
        "payee_id": payee_id,
        "status": status,
        "pay_currency": ccy,
        "total_amount": amount,
        "fx_rate_date": "latest",
        "created_at": "2025-04-01T00:00:00Z",
        "paid_at": "2025-04-05T00:00:00Z" if status == "paid" else None,
        "note": None,
        "breakdown_snapshot": {},
    }


def _make_coverage(payout_id, payee_id=PAYEE_ID, stmt_id="stmt-1", covered_amount=0.0):
    return {
        "payout_id": payout_id,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "covered_amount": covered_amount,
        "project_id": "proj-1",
    }


# ---------------------------------------------------------------------------
# Mock DB factory
# ---------------------------------------------------------------------------


def _build_db(payees, lines, payouts, coverage_by_payee=None):
    """Build a minimal Supabase mock that routes table() calls to preset data."""
    coverage_by_payee = coverage_by_payee or {}

    db = MagicMock()

    def table_side_effect(name):
        b = MagicMock()
        # Make every chainable filter return self
        for method in ("select", "eq", "neq", "in_", "order", "limit", "single", "maybe_single"):
            getattr(b, method).return_value = b

        if name == "royalty_payees":
            b.execute.return_value = MagicMock(data=payees)
        elif name == "royalty_lines":
            b.execute.return_value = MagicMock(data=lines)
        elif name == "royalty_payouts":
            b.execute.return_value = MagicMock(data=payouts)
        elif name == "royalty_payout_coverage":
            # Returns coverage filtered by payee_id — we simulate by returning all
            # (the service filters internally by payee_id via .eq("payee_id", ...))
            # We return all coverage regardless; the service trusts the DB filter.
            all_cov = []
            for cov_list in coverage_by_payee.values():
                all_cov.extend(cov_list)
            b.execute.return_value = MagicMock(data=all_cov)
        elif name == "registry_collaborators":
            b.execute.return_value = MagicMock(data=[])
        elif name == "projects":
            b.execute.return_value = MagicMock(data=[{"id": "proj-1", "name": "Project One"}])
        elif name in ("subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"):
            b.execute.return_value = MagicMock(data=[], count=0)
        else:
            b.execute.return_value = MagicMock(data=[])
        return b

    db.table.side_effect = table_side_effect
    return db


def _identity_fx(db, amount, frm, to, **kwargs):
    """Identity FX: same ccy = unchanged, GBP→USD ×1.27, USD→GBP /1.27."""
    if frm.upper() == to.upper():
        return amount
    if frm.upper() == "GBP" and to.upper() == "USD":
        return amount * 1.27
    if frm.upper() == "USD" and to.upper() == "GBP":
        return amount / 1.27
    return amount  # fallback identity for other pairs


# ---------------------------------------------------------------------------
# Test: mixed-currency, over-covered bucket — the critical money-safety test
# ---------------------------------------------------------------------------


class TestMixedCurrencyOverCoveredBucket:
    """S1 (USD, no coverage) + S2 (GBP, over-covered).

    S2 is over-paid: earned=100 GBP, covered=120 GBP → owed(S2)=max(0,−20)=0.
    S1 earns 100 USD, no coverage → owed(S1)=100 USD.

    Total owed in USD base = 100 (from S1) + 0 (from S2) = 100.
    The buggy implementation would do 100+100 GBP earned − 120 GBP coverage → owed=80,
    then convert → wrong answer.
    """

    def setup_method(self):
        self.payee = _make_payee(payout_ccy="USD")
        # S1: 100 USD, no coverage
        self.line_s1 = _make_line(stmt_id="stmt-s1", amount=100.0, ccy="USD", line_id="l1")
        # S2: 100 GBP, coverage=120 from a PAID payout
        self.line_s2 = _make_line(stmt_id="stmt-s2", amount=100.0, ccy="GBP", line_id="l2")
        self.payout_p1 = _make_payout("payout-1", status="paid", ccy="GBP", amount=120.0)
        self.coverage_s2 = _make_coverage("payout-1", stmt_id="stmt-s2", covered_amount=120.0)

        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line_s1, self.line_s2],
            payouts=[self.payout_p1],
            coverage_by_payee={PAYEE_ID: [self.coverage_s2]},
        )

    def test_s2_over_coverage_does_not_reduce_s1(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        assert len(summaries) == 1
        s = summaries[0]

        # S2: owed_b = max(0, 100 GBP - 120 GBP) = 0; converted to USD → 0
        # S1: owed_b = max(0, 100 USD) = 100; converted to USD → 100
        # Total owed = 100
        assert s["owed"] == pytest.approx(100.0), (
            f"Expected owed=100 (S1 only), got {s['owed']} — S2's over-coverage must not bleed into S1"
        )

    def test_s2_over_coverage_buggy_payee_level_result_is_different(self):
        """Document what the buggy result would look like, to confirm we avoid it."""
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        s = summaries[0]
        # Buggy: earned_all_in_usd = 100 + 100*1.27 = 227; covered_in_usd = 120*1.27 = 152.4
        # buggy_owed = 227 - 152.4 = 74.6  (WRONG — mixes currencies before clamp)
        # Correct: 100 (USD)
        assert s["owed"] != pytest.approx(74.6), "Got buggy cross-currency result"
        assert s["owed"] == pytest.approx(100.0)

    def test_earned_includes_both_buckets(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        s = summaries[0]
        # earned = 100 USD + 100 GBP * 1.27 = 100 + 127 = 227
        assert s["earned"] == pytest.approx(100.0 + 100.0 * 1.27)

    def test_status_is_owed(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        assert summaries[0]["status"] == "owed"

    def test_native_totals_use_payout_currency(self):
        """Native totals are in the payee's payout_currency (USD in this case)."""
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        s = summaries[0]
        # payout_currency = USD = base, so native == reporting
        assert s["owed_native"] == pytest.approx(s["owed"])
        assert s["earned_native"] == pytest.approx(s["earned"])


# ---------------------------------------------------------------------------
# Test: fully covered bucket → owed = 0
# ---------------------------------------------------------------------------


class TestFullyCoveredBucket:
    def setup_method(self):
        self.payee = _make_payee(payout_ccy="USD")
        self.line = _make_line(stmt_id="stmt-1", amount=100.0, ccy="USD")
        self.payout = _make_payout("payout-1", status="paid", ccy="USD", amount=100.0)
        self.coverage = _make_coverage("payout-1", stmt_id="stmt-1", covered_amount=100.0)
        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line],
            payouts=[self.payout],
            coverage_by_payee={PAYEE_ID: [self.coverage]},
        )

    def test_owed_is_zero(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        assert summaries[0]["owed"] == pytest.approx(0.0)
        assert summaries[0]["status"] == "settled"

    def test_earned_equals_paid(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        s = summaries[0]
        assert s["earned"] == pytest.approx(100.0)
        assert s["paid"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test: incremental earnings — earned 150, paid 100 → owed 50
# ---------------------------------------------------------------------------


class TestIncrementalEarnings:
    def setup_method(self):
        self.payee = _make_payee(payout_ccy="USD")
        self.line = _make_line(stmt_id="stmt-1", amount=150.0, ccy="USD")
        self.payout = _make_payout("payout-1", status="paid", ccy="USD", amount=100.0)
        self.coverage = _make_coverage("payout-1", stmt_id="stmt-1", covered_amount=100.0)
        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line],
            payouts=[self.payout],
            coverage_by_payee={PAYEE_ID: [self.coverage]},
        )

    def test_owed_is_fifty(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        s = summaries[0]
        assert s["owed"] == pytest.approx(50.0)
        assert s["status"] == "owed"
        assert s["earned"] == pytest.approx(150.0)
        assert s["paid"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test: native totals use payout_currency, not base
# ---------------------------------------------------------------------------


class TestNativeTotalsUsePayoutCurrency:
    """Payee has payout_currency=GBP, base=USD. Native totals must be in GBP."""

    def setup_method(self):
        self.payee = _make_payee(payout_ccy="GBP")
        self.line = _make_line(stmt_id="stmt-1", amount=100.0, ccy="USD")
        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line],
            payouts=[],
            coverage_by_payee={PAYEE_ID: []},
        )

    def test_native_owed_is_in_gbp(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        s = summaries[0]
        # earned reporting = 100 USD → 100 USD (identity)
        assert s["earned"] == pytest.approx(100.0)
        # earned_native = 100 USD → GBP = 100/1.27
        assert s["earned_native"] == pytest.approx(100.0 / 1.27)
        assert s["owed_native"] == pytest.approx(100.0 / 1.27)


# ---------------------------------------------------------------------------
# Test: collision detection
# ---------------------------------------------------------------------------


class TestCollisionDetection:
    """collision=True when two registry rows with distinct emails share the normalized name."""

    def _build_db_with_collabs(self, collabs):
        db = MagicMock()
        payee = _make_payee(normalized_name="alice jones")

        def table_side_effect(name):
            b = MagicMock()
            for method in ("select", "eq", "neq", "in_", "order", "limit", "single", "maybe_single"):
                getattr(b, method).return_value = b
            if name == "royalty_payees":
                b.execute.return_value = MagicMock(data=[payee])
            elif name == "royalty_lines" or name == "royalty_payouts" or name == "royalty_payout_coverage":
                b.execute.return_value = MagicMock(data=[])
            elif name == "registry_collaborators":
                b.execute.return_value = MagicMock(data=collabs)
            elif name == "projects":
                b.execute.return_value = MagicMock(data=[])
            else:
                b.execute.return_value = MagicMock(data=[])
            return b

        db.table.side_effect = table_side_effect
        return db

    def test_collision_true_distinct_emails(self):
        """Two rows with different emails → collision=True."""
        collabs = [
            {"name": "Alice Jones", "email": "alice1@example.com", "collaborator_user_id": None},
            {"name": "Alice Jones", "email": "alice2@example.com", "collaborator_user_id": None},
        ]
        db = self._build_db_with_collabs(collabs)
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(db, USER_ID, base="USD")

        assert summaries[0]["collision"] is True

    def test_collision_false_same_user_id(self):
        """Same collaborator_user_id across two works → one person → no collision."""
        collabs = [
            {"name": "Alice Jones", "email": "alice@example.com", "collaborator_user_id": "uid-alice"},
            {"name": "Alice Jones", "email": "alice@example.com", "collaborator_user_id": "uid-alice"},
        ]
        db = self._build_db_with_collabs(collabs)
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(db, USER_ID, base="USD")

        assert summaries[0]["collision"] is False

    def test_collision_false_single_entry(self):
        """Single registry row → only one person → no collision."""
        collabs = [
            {"name": "Alice Jones", "email": "alice@example.com", "collaborator_user_id": None},
        ]
        db = self._build_db_with_collabs(collabs)
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(db, USER_ID, base="USD")

        assert summaries[0]["collision"] is False

    def test_collision_true_different_user_ids(self):
        """Two different collaborator_user_ids → two distinct people → collision=True."""
        collabs = [
            {"name": "Alice Jones", "email": "any@example.com", "collaborator_user_id": "uid-1"},
            {"name": "Alice Jones", "email": "any@example.com", "collaborator_user_id": "uid-2"},
        ]
        db = self._build_db_with_collabs(collabs)
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(db, USER_ID, base="USD")

        assert summaries[0]["collision"] is True


# ---------------------------------------------------------------------------
# Test: payee_detail ownership check
# ---------------------------------------------------------------------------


class TestPayeeDetailOwnership:
    def test_raises_permission_error_for_wrong_user(self):
        db = MagicMock()

        def table_side_effect(name):
            b = MagicMock()
            for method in ("select", "eq", "neq", "in_", "order", "limit", "single", "maybe_single"):
                getattr(b, method).return_value = b
            # Return empty → payee not found for this user
            b.execute.return_value = MagicMock(data=[])
            return b

        db.table.side_effect = table_side_effect
        with pytest.raises(PermissionError):
            service.payee_detail(db, "wrong-user", PAYEE_ID, base="USD")


# ---------------------------------------------------------------------------
# Test: empty payees → empty results
# ---------------------------------------------------------------------------


class TestEmptyPayees:
    def test_payee_summary_empty(self):
        db = MagicMock()

        def table_side_effect(name):
            b = MagicMock()
            for method in ("select", "eq", "neq", "in_", "order", "limit", "single", "maybe_single"):
                getattr(b, method).return_value = b
            b.execute.return_value = MagicMock(data=[])
            return b

        db.table.side_effect = table_side_effect
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.payee_summary(db, USER_ID, base="USD")

        assert result == []

    def test_periods_ledger_empty(self):
        db = MagicMock()

        def table_side_effect(name):
            b = MagicMock()
            for method in ("select", "eq", "neq", "in_", "order", "limit", "single", "maybe_single"):
                getattr(b, method).return_value = b
            b.execute.return_value = MagicMock(data=[])
            return b

        db.table.side_effect = table_side_effect
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            result = service.periods_ledger(db, USER_ID, base="USD")

        assert result["base"] == "USD"
        assert result["rows"] == []


# ---------------------------------------------------------------------------
# Test: draft payout (scheduled status)
# ---------------------------------------------------------------------------


class TestDraftPayoutStatus:
    def setup_method(self):
        self.payee = _make_payee()
        self.line = _make_line(stmt_id="stmt-1", amount=100.0, ccy="USD")
        self.payout = _make_payout("payout-1", status="draft", ccy="USD", amount=100.0)
        self.coverage = _make_coverage("payout-1", stmt_id="stmt-1", covered_amount=100.0)
        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line],
            payouts=[self.payout],
            coverage_by_payee={PAYEE_ID: [self.coverage]},
        )

    def test_status_is_scheduled(self):
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(self.db, USER_ID, base="USD")

        s = summaries[0]
        # earned=100, drafted=100, owed=max(0, 100-100)=0 → scheduled
        assert s["owed"] == pytest.approx(0.0)
        assert s["drafted"] == pytest.approx(100.0)
        assert s["status"] == "scheduled"


# ---------------------------------------------------------------------------
# Test: project_count
# ---------------------------------------------------------------------------


class TestProjectCount:
    def test_project_count_distinct(self):
        payee = _make_payee()
        lines = [
            _make_line(stmt_id="stmt-1", project_id="proj-A", line_id="l1"),
            _make_line(stmt_id="stmt-2", project_id="proj-B", line_id="l2"),
            _make_line(stmt_id="stmt-3", project_id="proj-A", line_id="l3"),  # duplicate
        ]
        db = _build_db(payees=[payee], lines=lines, payouts=[], coverage_by_payee={PAYEE_ID: []})
        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(db, USER_ID, base="USD")

        assert summaries[0]["project_count"] == 2


# ---------------------------------------------------------------------------
# Test: multi-project statement splits per project (P0.1 regression test)
# ---------------------------------------------------------------------------


class TestMultiProjectStatementRebucket:
    """A single royalty statement with lines in two distinct projects must produce
    two independent (statement, project) buckets — not one combined bucket.

    This is the P0.1 regression guard: before the fix the whole statement was
    attributed to the first line's project and coverage was keyed by stmt_id alone.
    """

    def test_multiproject_statement_splits_per_project(self):
        """One payee, one statement s1, lines in proj-A (earned 100) and proj-B (earned 40),
        no coverage → payee_owed_buckets returns two buckets: one for each project."""
        payee = _make_payee()
        line_a = _make_line(stmt_id="s1", amount=100.0, ccy="USD", project_id="proj-A", line_id="la")
        line_b = _make_line(stmt_id="s1", amount=40.0, ccy="USD", project_id="proj-B", line_id="lb")

        db = _build_db(
            payees=[payee],
            lines=[line_a, line_b],
            payouts=[],
            coverage_by_payee={PAYEE_ID: []},
        )

        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)

        assert len(buckets) == 2, (
            f"Expected 2 buckets (one per project), got {len(buckets)} — "
            "multi-project statement must not be lumped into a single bucket"
        )

        owed_by_project = {b["project_id"]: b["owed_b"] for b in buckets}
        assert owed_by_project.get("proj-A") == pytest.approx(100.0), (
            f"proj-A bucket should owe 100, got {owed_by_project.get('proj-A')}"
        )
        assert owed_by_project.get("proj-B") == pytest.approx(40.0), (
            f"proj-B bucket should owe 40, got {owed_by_project.get('proj-B')}"
        )

        # Both buckets reference the same statement
        assert all(b["royalty_statement_id"] == "s1" for b in buckets)

    def test_multiproject_buckets_sum_to_payee_total(self):
        """The sum of multi-project bucket owed_b values must equal the payee's
        total owed in the payee_summary (after FX, both are USD so identity holds)."""
        payee = _make_payee(payout_ccy="USD")
        line_a = _make_line(stmt_id="s1", amount=100.0, ccy="USD", project_id="proj-A", line_id="la")
        line_b = _make_line(stmt_id="s1", amount=40.0, ccy="USD", project_id="proj-B", line_id="lb")

        db = _build_db(
            payees=[payee],
            lines=[line_a, line_b],
            payouts=[],
            coverage_by_payee={PAYEE_ID: []},
        )

        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            summaries = service.payee_summary(db, USER_ID, base="USD")

        assert summaries[0]["owed"] == pytest.approx(140.0), "Total owed (100+40) must equal sum of per-project buckets"

    def test_single_project_statement_back_compat(self):
        """Back-compat: a single-project statement still yields exactly one bucket
        with the same owed as before the re-bucketing change."""
        payee = _make_payee()
        line = _make_line(stmt_id="stmt-1", amount=200.0, ccy="USD", project_id="proj-1", line_id="l1")

        db = _build_db(
            payees=[payee],
            lines=[line],
            payouts=[],
            coverage_by_payee={PAYEE_ID: []},
        )

        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)

        assert len(buckets) == 1, f"Single-project statement must yield exactly one bucket, got {len(buckets)}"
        assert buckets[0]["owed_b"] == pytest.approx(200.0)
        assert buckets[0]["project_id"] == "proj-1"
        assert buckets[0]["royalty_statement_id"] == "stmt-1"

    def test_multiproject_statement_coverage_per_project(self):
        """Coverage for proj-A should not reduce owed for proj-B when they share a statement."""
        payee = _make_payee()
        line_a = _make_line(stmt_id="s1", amount=100.0, ccy="USD", project_id="proj-A", line_id="la")
        line_b = _make_line(stmt_id="s1", amount=40.0, ccy="USD", project_id="proj-B", line_id="lb")

        payout = _make_payout("payout-1", status="paid")
        # Coverage only for proj-A bucket within statement s1
        cov_a = {
            "payout_id": "payout-1",
            "payee_id": PAYEE_ID,
            "royalty_statement_id": "s1",
            "covered_amount": 100.0,
            "project_id": "proj-A",
        }

        db = _build_db(
            payees=[payee],
            lines=[line_a, line_b],
            payouts=[payout],
            coverage_by_payee={PAYEE_ID: [cov_a]},
        )

        buckets = service.payee_owed_buckets(db, USER_ID, PAYEE_ID)

        # proj-A is fully covered → skipped; proj-B still owes 40
        assert len(buckets) == 1, f"Expected 1 bucket (proj-A fully covered), got {len(buckets)}"
        assert buckets[0]["project_id"] == "proj-B"
        assert buckets[0]["owed_b"] == pytest.approx(40.0), "Coverage for proj-A must not reduce owed for proj-B"


# ---------------------------------------------------------------------------
# Test: periods_ledger multi-project state independence (P0.1 follow-up)
# ---------------------------------------------------------------------------


class TestPeriodsLedgerMultiProject:
    """A multi-project statement must produce independent PeriodCells: a fully-covered
    project must not flip an uncovered project's cell to 'settled'.

    Before the re-bucket, periods_ledger grouped by stmt_id alone and keyed coverage
    by stmt_id alone, so proj-A's coverage could mask proj-B's owed within the cell.
    """

    def test_covered_project_does_not_flip_uncovered_project_state(self):
        payee = _make_payee(payout_ccy="USD")
        # One statement s1, two projects: proj-A (covered) + proj-B (uncovered)
        line_a = _make_line(stmt_id="s1", amount=100.0, ccy="USD", project_id="proj-A", line_id="la")
        line_b = _make_line(stmt_id="s1", amount=40.0, ccy="USD", project_id="proj-B", line_id="lb")

        payout = _make_payout("payout-1", status="paid")
        # Coverage fully settles proj-A within s1 only
        cov_a = {
            "payout_id": "payout-1",
            "payee_id": PAYEE_ID,
            "royalty_statement_id": "s1",
            "covered_amount": 100.0,
            "project_id": "proj-A",
        }

        db = _build_db(
            payees=[payee],
            lines=[line_a, line_b],
            payouts=[payout],
            coverage_by_payee={PAYEE_ID: [cov_a]},
        )

        with patch("oneclick.royalties.service.fx.convert", side_effect=_identity_fx):
            ledger = service.periods_ledger(db, USER_ID, base="USD")

        assert len(ledger["rows"]) == 1
        cells = ledger["rows"][0]["cells"]

        # Two cells: one per (statement, project)
        assert len(cells) == 2, f"Expected 2 cells (one per project), got {len(cells)}"

        states_by_earned = {round(c["earned"], 2): c["state"] for c in cells}
        # proj-A: earned 100, fully covered → settled
        assert states_by_earned[100.0] == "settled"
        # proj-B: earned 40, no coverage → owed (must NOT be flipped to settled)
        assert states_by_earned[40.0] == "owed", "proj-A's coverage must not flip proj-B's uncovered cell to settled"

        # Row total still foots to the full earned (100 + 40)
        assert ledger["rows"][0]["total"] == pytest.approx(140.0)


# ---------------------------------------------------------------------------
# Test: unconvertible bucket is excluded and counted
# ---------------------------------------------------------------------------


class TestUnconvertibleBucketExcluded:
    """When fx.convert returns None for a bucket (on_missing='none'), that bucket
    must be excluded from the base total and unconvertible_count must be 1."""

    def test_ngn_bucket_excluded_and_counted(self):
        """One USD bucket (convertible) + one NGN bucket (unconvertible).

        The NGN bucket must be excluded from earned/owed totals, and
        unconvertible_count must equal 1.
        """
        payee = _make_payee(payout_ccy="USD")
        line_usd = _make_line(stmt_id="stmt-usd", amount=100.0, ccy="USD", project_id="proj-1", line_id="l-usd")
        line_ngn = _make_line(stmt_id="stmt-ngn", amount=50000.0, ccy="NGN", project_id="proj-1", line_id="l-ngn")

        db = _build_db(
            payees=[payee],
            lines=[line_usd, line_ngn],
            payouts=[],
            coverage_by_payee={PAYEE_ID: []},
        )

        def _fx_with_ngn_none(db, amount, frm, to, **kwargs):
            on_missing = kwargs.get("on_missing", "amount")
            if frm.upper() == "NGN" and on_missing == "none":
                return None
            # identity for everything else
            return _identity_fx(db, amount, frm, to)

        with patch("oneclick.royalties.service.fx.convert", side_effect=_fx_with_ngn_none):
            summaries = service.payee_summary(db, USER_ID, base="USD")

        assert len(summaries) == 1
        s = summaries[0]

        # Only the USD bucket (100.0) should be included in the base total
        assert s["earned"] == pytest.approx(100.0), f"NGN bucket must be excluded from earned; got {s['earned']}"
        assert s["owed"] == pytest.approx(100.0), f"NGN bucket must be excluded from owed; got {s['owed']}"

        # Exactly one bucket was unconvertible
        assert s["unconvertible_count"] == 1, f"Expected unconvertible_count=1, got {s['unconvertible_count']}"
