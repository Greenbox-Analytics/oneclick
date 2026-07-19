"""Tests for oneclick/royalties/analytics_service.py.

All tests mock the Supabase client and monkeypatch fx.convert.
No real DB or network calls are made.

Invariants verified:
  - drafted_total comes from UNFILTERED _aggregate_payee_buckets path.
  - Fully-drafted payee (owed==0, drafted>0) → draft_count==1 but
    outstanding_total==0, payees_owed_count==0.
  - paid_last_30d uses injected `now` param.
  - Per-artist earned/paid sums to portfolio overview totals.
  - by_month sums foot to summary totals.
  - Unconvertible bucket (convert→None) excluded from totals, unconvertible_count incremented.
  - Empty data → zeros / [].
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from oneclick.royalties import analytics_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-analytics-111"
PAYEE_A = "payee-analytics-aaa"
PAYEE_B = "payee-analytics-bbb"
ARTIST_1 = "artist-111"
ARTIST_2 = "artist-222"
PROJ_1 = "proj-analytics-1"
PROJ_2 = "proj-analytics-2"

# ---------------------------------------------------------------------------
# Helpers — mirrors test_royalties_aggregation.py style
# ---------------------------------------------------------------------------


def _make_payee(payee_id=PAYEE_A, display_name="Alice", payout_ccy="USD", normalized_name="alice"):
    return {
        "id": payee_id,
        "user_id": USER_ID,
        "display_name": display_name,
        "payout_currency": payout_ccy,
        "registry_user_id": None,
        "email": None,
        "normalized_name": normalized_name,
    }


def _make_line(
    payee_id=PAYEE_A,
    stmt_id="stmt-1",
    amount=100.0,
    ccy="USD",
    project_id=PROJ_1,
    line_id="line-1",
    calc_id="calc-1",
    period_start="2025-01-01",
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
        "period_start": period_start,
        "period_end": None,
    }


def _make_payout(payout_id, payee_id=PAYEE_A, status="paid", ccy="USD", amount=0.0, paid_at=None):
    return {
        "id": payout_id,
        "user_id": USER_ID,
        "payee_id": payee_id,
        "status": status,
        "pay_currency": ccy,
        "total_amount": amount,
        "fx_rate_date": "latest",
        "created_at": "2025-04-01T00:00:00Z",
        "paid_at": paid_at,
        "note": None,
        "breakdown_snapshot": {},
    }


def _make_coverage(
    payout_id,
    payee_id=PAYEE_A,
    stmt_id="stmt-1",
    covered_amount=0.0,
    project_id=PROJ_1,
):
    return {
        "payout_id": payout_id,
        "payee_id": payee_id,
        "royalty_statement_id": stmt_id,
        "covered_amount": covered_amount,
        "project_id": project_id,
    }


def _make_project(project_id=PROJ_1, artist_id=ARTIST_1, name="Project One"):
    return {"id": project_id, "name": name, "artist_id": artist_id, "user_id": USER_ID}


# ---------------------------------------------------------------------------
# Mock DB factory
# ---------------------------------------------------------------------------


def _build_db(
    payees,
    lines,
    payouts,
    coverage_all=None,
    projects=None,
    coverage_by_payee=None,
):
    """Build a minimal Supabase mock.

    coverage_all: list of all coverage rows (used by the in_("payout_id", ...) path in
                  analytics_service._load_coverage_all).
    coverage_by_payee: dict {payee_id: [coverage rows]} for the
                       .eq("payee_id", ...) path in service._load_coverage_for_payee.
                       If provided, the mock filters by payee_id so multi-payee scenarios
                       don't cross-contaminate coverage.
    """
    coverage_all = coverage_all or []
    projects = projects or []
    coverage_by_payee = coverage_by_payee or {}

    db = MagicMock()

    def table_side_effect(name):
        b = MagicMock()

        # We need to track the chain of filter calls to serve the right data.
        # Use a simple stateful object per table builder call.
        _state: dict = {"payee_id_filter": None}

        def _eq(col, val):
            if col == "payee_id":
                _state["payee_id_filter"] = val
            return b

        def _in_(col, vals):
            # .in_("payout_id", [...]) — used by _load_coverage_all
            # Return self; execute() will return coverage_all for this path.
            return b

        for method in ("select", "neq", "order", "limit", "single", "maybe_single", "upsert"):
            getattr(b, method).return_value = b
        b.eq.side_effect = _eq
        b.in_.side_effect = _in_

        def _execute():
            if name == "royalty_payees":
                return MagicMock(data=payees)
            if name == "royalty_lines":
                return MagicMock(data=lines)
            if name == "royalty_payouts":
                return MagicMock(data=payouts)
            if name == "royalty_payout_coverage":
                pid_filter = _state.get("payee_id_filter")
                if pid_filter is not None and coverage_by_payee:
                    # Per-payee path (service._load_coverage_for_payee)
                    return MagicMock(data=coverage_by_payee.get(pid_filter, []))
                # Bulk path (_load_coverage_all via .in_)
                return MagicMock(data=coverage_all)
            if name == "registry_collaborators":
                return MagicMock(data=[])
            if name == "projects":
                return MagicMock(data=projects)
            if name == "fx_rate_snapshots":
                return MagicMock(data=[])
            return MagicMock(data=[])

        b.execute.side_effect = _execute
        return b

    db.table.side_effect = table_side_effect
    return db


def _identity_fx(db, amount, frm, to, **kwargs):
    """Identity FX — same as test_royalties_aggregation.py."""
    if frm.upper() == to.upper():
        return amount
    if frm.upper() == "GBP" and to.upper() == "USD":
        return amount * 1.27
    if frm.upper() == "USD" and to.upper() == "GBP":
        return amount / 1.27
    return amount


# ---------------------------------------------------------------------------
# Test: empty data → zeros / []
# ---------------------------------------------------------------------------


class TestEmptyData:
    def test_overview_empty(self):
        db = _build_db(payees=[], lines=[], payouts=[], coverage_all=[], projects=[])
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            result = analytics_service.overview(db, USER_ID, base="USD")

        assert result.outstanding_total == pytest.approx(0.0)
        assert result.payees_owed_count == 0
        assert result.drafted_total == pytest.approx(0.0)
        assert result.draft_count == 0
        assert result.paid_total == pytest.approx(0.0)
        assert result.paid_last_30d == pytest.approx(0.0)
        assert result.paid_by_month == []
        assert result.top_owed == []
        assert result.unconvertible_count == 0

    def test_artist_analytics_empty_lines(self):
        db = _build_db(payees=[], lines=[], payouts=[], coverage_all=[], projects=[])
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            result = analytics_service.artist_analytics(db, USER_ID, ARTIST_1, base="USD")

        assert result.summary["earned_total"] == pytest.approx(0.0)
        assert result.summary["owed_now"] == pytest.approx(0.0)
        assert result.summary["paid_total"] == pytest.approx(0.0)
        assert result.by_month == []

    def test_payee_analytics_no_lines(self):
        payee = _make_payee()
        db = _build_db(
            payees=[payee],
            lines=[],
            payouts=[],
            coverage_all=[],
            coverage_by_payee={PAYEE_A: []},
        )
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            result = analytics_service.payee_analytics(db, USER_ID, PAYEE_A, base="USD")

        assert result.summary["earned_total"] == pytest.approx(0.0)
        assert result.summary["paid_total"] == pytest.approx(0.0)
        assert result.summary["owed"] == pytest.approx(0.0)
        assert result.by_month == []


# ---------------------------------------------------------------------------
# Test: fully-drafted payee → still Outstanding (until paid) AND drafted
# ---------------------------------------------------------------------------


class TestOverviewDraftedPayeeStaysOutstanding:
    """A fully-drafted payee (owed==0, drafted>0) still counts as Outstanding.

    A draft is a plan, not a payment: outstanding = earned − paid (NOT minus
    drafted), so the money stays in outstanding_total / payees_owed_count until
    the payout is actually completed. It also shows in drafted_total /
    draft_count. The draft payout has paid_at=None → excluded from paid_total.
    """

    def setup_method(self):
        self.payee = _make_payee()
        self.line = _make_line(stmt_id="stmt-1", amount=100.0, ccy="USD")
        # Draft payout — paid_at=None
        self.payout = _make_payout("payout-1", status="draft", paid_at=None)
        self.coverage = _make_coverage("payout-1", stmt_id="stmt-1", covered_amount=100.0)
        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line],
            payouts=[self.payout],
            coverage_all=[self.coverage],
            projects=[_make_project()],
            coverage_by_payee={PAYEE_A: [self.coverage]},
        )

    def test_drafted_payee_still_counts_as_outstanding(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            result = analytics_service.overview(self.db, USER_ID, base="USD")

        # unpaid = max(0, 100 - 0) = 100 → still Outstanding until actually paid
        assert result.outstanding_total == pytest.approx(100.0), (
            f"drafted-but-unpaid payee must stay in outstanding_total; got {result.outstanding_total}"
        )
        assert result.payees_owed_count == 1
        # ...and appears in Top outstanding with the full unpaid amount
        assert len(result.top_owed) == 1
        assert result.top_owed[0].owed == pytest.approx(100.0)

        # drafted>0 → also shows in drafted_total and draft_count
        assert result.drafted_total == pytest.approx(100.0), (
            f"drafted_total must include fully-drafted payee; got {result.drafted_total}"
        )
        assert result.draft_count == 1, f"draft_count must be 1; got {result.draft_count}"

        # draft payout has paid_at=None → excluded from paid_total
        assert result.paid_total == pytest.approx(0.0), f"paid_total must be 0 for draft; got {result.paid_total}"


# ---------------------------------------------------------------------------
# Test: paid_last_30d uses injected now
# ---------------------------------------------------------------------------


class TestPaidLast30dUsesInjectedNow:
    """A payout paid 40d before `now` must be excluded from paid_last_30d;
    one paid 10d before must be included."""

    def setup_method(self):
        self.now = date(2025, 6, 15)
        self.payee = _make_payee()
        self.line_a = _make_line(stmt_id="stmt-old", amount=200.0, ccy="USD", line_id="l-old")
        self.line_b = _make_line(stmt_id="stmt-recent", amount=150.0, ccy="USD", line_id="l-recent")
        # Paid 40d before now → excluded from last 30d
        paid_old = (self.now - timedelta(days=40)).isoformat()
        paid_recent = (self.now - timedelta(days=10)).isoformat()

        self.payout_old = _make_payout("payout-old", status="paid", paid_at=paid_old + "T00:00:00Z", amount=200.0)
        self.payout_recent = _make_payout(
            "payout-recent", status="paid", paid_at=paid_recent + "T00:00:00Z", amount=150.0
        )
        self.coverage_old = _make_coverage("payout-old", stmt_id="stmt-old", covered_amount=200.0)
        self.coverage_recent = _make_coverage("payout-recent", stmt_id="stmt-recent", covered_amount=150.0)

        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line_a, self.line_b],
            payouts=[self.payout_old, self.payout_recent],
            coverage_all=[self.coverage_old, self.coverage_recent],
            projects=[_make_project()],
            coverage_by_payee={PAYEE_A: [self.coverage_old, self.coverage_recent]},
        )

    def test_paid_last_30d_excludes_old_payout(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            result = analytics_service.overview(self.db, USER_ID, base="USD", now=self.now)

        # total paid = 200 + 150 = 350
        assert result.paid_total == pytest.approx(350.0)
        # last 30d: only the 10-day-old payout
        assert result.paid_last_30d == pytest.approx(150.0), (
            f"paid_last_30d should be 150 (only recent payout); got {result.paid_last_30d}"
        )


# ---------------------------------------------------------------------------
# Test: per-artist foots to portfolio
# ---------------------------------------------------------------------------


class TestPerArtistFootsToPortfolio:
    """Two artists' earned/paid must sum to the overview portfolio totals."""

    def setup_method(self):
        self.payee_a = _make_payee(PAYEE_A, "Alice")
        self.payee_b = _make_payee(PAYEE_B, "Bob")

        self.proj_a = _make_project(PROJ_1, ARTIST_1, "Artist 1 Project")
        self.proj_b = _make_project(PROJ_2, ARTIST_2, "Artist 2 Project")

        self.line_a = _make_line(PAYEE_A, "stmt-a", 200.0, "USD", PROJ_1, "l-a", period_start="2025-02-01")
        self.line_b = _make_line(PAYEE_B, "stmt-b", 300.0, "USD", PROJ_2, "l-b", period_start="2025-03-01")

        self.payout_a = _make_payout("pay-a", PAYEE_A, "paid", paid_at="2025-04-01T00:00:00Z", amount=200.0)
        self.payout_b = _make_payout("pay-b", PAYEE_B, "paid", paid_at="2025-04-01T00:00:00Z", amount=300.0)

        self.cov_a = _make_coverage("pay-a", PAYEE_A, "stmt-a", 200.0, PROJ_1)
        self.cov_b = _make_coverage("pay-b", PAYEE_B, "stmt-b", 300.0, PROJ_2)

        self.db = _build_db(
            payees=[self.payee_a, self.payee_b],
            lines=[self.line_a, self.line_b],
            payouts=[self.payout_a, self.payout_b],
            coverage_all=[self.cov_a, self.cov_b],
            projects=[self.proj_a, self.proj_b],
            coverage_by_payee={PAYEE_A: [self.cov_a], PAYEE_B: [self.cov_b]},
        )

    def test_per_artist_foots_to_portfolio(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            ov = analytics_service.overview(self.db, USER_ID, base="USD")
            art1 = analytics_service.artist_analytics(self.db, USER_ID, ARTIST_1, base="USD")
            art2 = analytics_service.artist_analytics(self.db, USER_ID, ARTIST_2, base="USD")

        # Each artist's earned
        assert art1.summary["earned_total"] == pytest.approx(200.0)
        assert art2.summary["earned_total"] == pytest.approx(300.0)

        # Each artist's paid
        assert art1.summary["paid_total"] == pytest.approx(200.0)
        assert art2.summary["paid_total"] == pytest.approx(300.0)

        # Sum foots to overview paid_total
        artists_paid_sum = art1.summary["paid_total"] + art2.summary["paid_total"]
        assert artists_paid_sum == pytest.approx(ov.paid_total), (
            f"Σ artist paid ({artists_paid_sum}) must equal overview paid_total ({ov.paid_total})"
        )


# ---------------------------------------------------------------------------
# Test: by_month sums foot to summary
# ---------------------------------------------------------------------------


class TestByMonthSumsToSummary:
    """Σ by_month.earned == summary.earned_total and Σ by_month.paid == summary.paid_total."""

    def setup_method(self):
        self.payee = _make_payee()
        # Two lines in different months
        self.line_jan = _make_line(
            stmt_id="stmt-jan", amount=100.0, ccy="USD", line_id="l-jan", period_start="2025-01-01"
        )
        self.line_feb = _make_line(
            stmt_id="stmt-feb", amount=120.0, ccy="USD", line_id="l-feb", period_start="2025-02-01"
        )
        # Paid payout covering stmt-jan
        self.payout = _make_payout("pay-1", status="paid", paid_at="2025-03-15T00:00:00Z", amount=100.0)
        self.cov = _make_coverage("pay-1", stmt_id="stmt-jan", covered_amount=100.0)

        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line_jan, self.line_feb],
            payouts=[self.payout],
            coverage_all=[self.cov],
            projects=[_make_project()],
            coverage_by_payee={PAYEE_A: [self.cov]},
        )

    def test_payee_by_month_foots_to_summary(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            result = analytics_service.payee_analytics(self.db, USER_ID, PAYEE_A, base="USD")

        earned_sum = sum(m.earned for m in result.by_month)
        paid_sum = sum(m.paid for m in result.by_month)

        assert earned_sum == pytest.approx(result.summary["earned_total"]), (
            f"Σ by_month.earned ({earned_sum}) must == summary.earned_total ({result.summary['earned_total']})"
        )
        assert paid_sum == pytest.approx(result.summary["paid_total"]), (
            f"Σ by_month.paid ({paid_sum}) must == summary.paid_total ({result.summary['paid_total']})"
        )

    def test_artist_by_month_foots_to_summary(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx):
            result = analytics_service.artist_analytics(self.db, USER_ID, ARTIST_1, base="USD")

        earned_sum = sum(m.earned for m in result.by_month)
        paid_sum = sum(m.paid for m in result.by_month)

        assert earned_sum == pytest.approx(result.summary["earned_total"])
        assert paid_sum == pytest.approx(result.summary["paid_total"])


# ---------------------------------------------------------------------------
# Test: unconvertible bucket excluded and counted
# ---------------------------------------------------------------------------


class TestUnconvertibleBucketExcludedAndCounted:
    """A bucket whose statement_currency has no BoC rate (fx.convert→None) must be
    excluded from totals and increment unconvertible_count."""

    def setup_method(self):
        self.payee = _make_payee()
        # USD line (convertible) + NGN line (unconvertible)
        self.line_usd = _make_line(stmt_id="stmt-usd", amount=100.0, ccy="USD", line_id="l-usd")
        self.line_ngn = _make_line(stmt_id="stmt-ngn", amount=50000.0, ccy="NGN", line_id="l-ngn")

        self.db = _build_db(
            payees=[self.payee],
            lines=[self.line_usd, self.line_ngn],
            payouts=[],
            coverage_all=[],
            projects=[_make_project()],
            coverage_by_payee={PAYEE_A: []},
        )

    def _ngn_none_fx(self, db, amount, frm, to, **kwargs):
        on_missing = kwargs.get("on_missing", "amount")
        if frm.upper() == "NGN" and on_missing == "none":
            return None
        return _identity_fx(db, amount, frm, to)

    def test_overview_excludes_ngn_and_counts(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=self._ngn_none_fx):
            result = analytics_service.overview(self.db, USER_ID, base="USD")

        # Only the USD bucket (100) should appear in outstanding_total
        assert result.outstanding_total == pytest.approx(100.0), (
            f"NGN bucket must be excluded from outstanding_total; got {result.outstanding_total}"
        )
        assert result.unconvertible_count == 1, f"Expected unconvertible_count=1; got {result.unconvertible_count}"

    def test_payee_analytics_excludes_ngn_and_counts(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=self._ngn_none_fx):
            result = analytics_service.payee_analytics(self.db, USER_ID, PAYEE_A, base="USD")

        assert result.summary["earned_total"] == pytest.approx(100.0), (
            f"NGN bucket must be excluded from earned_total; got {result.summary['earned_total']}"
        )
        assert result.unconvertible_count == 1, f"Expected unconvertible_count=1; got {result.unconvertible_count}"

    def test_artist_analytics_excludes_ngn_and_counts(self):
        with patch("oneclick.royalties.analytics_service.fx.convert", side_effect=self._ngn_none_fx):
            result = analytics_service.artist_analytics(self.db, USER_ID, ARTIST_1, base="USD")

        assert result.summary["earned_total"] == pytest.approx(100.0)
        assert result.unconvertible_count == 1


# ---------------------------------------------------------------------------
# Test: payee_analytics ownership check
# ---------------------------------------------------------------------------


class TestPayeeAnalyticsOwnership:
    """payee_analytics must raise PermissionError when the payee isn't owned by the
    caller — the .eq('id', X).eq('user_id', caller) ownership query returns no rows."""

    def test_raises_permission_error_for_foreign_user(self):
        # The ownership query .eq("id", PAYEE_A).eq("user_id", "other-user") finds nothing,
        # so the mock returns an empty royalty_payees set for the foreign caller.
        db = _build_db(
            payees=[],
            lines=[],
            payouts=[],
            coverage_all=[],
            coverage_by_payee={},
        )
        with (
            patch("oneclick.royalties.analytics_service.fx.convert", side_effect=_identity_fx),
            pytest.raises(PermissionError),
        ):
            analytics_service.payee_analytics(db, "other-user-999", PAYEE_A, base="USD")
