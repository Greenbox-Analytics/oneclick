"""Analytics aggregation service for OneClick royalties.

Provides portfolio-level overview, per-artist, and per-payee analytics.

Money-safety invariants (identical to service.py):
  - All conversions use on_missing="none" → None → exclude bucket + increment
    unconvertible_count. Never fold at 1:1.
  - Per-bucket arithmetic in statement currency, clamp, then convert.
  - Draft payouts (paid_at NULL or status != 'paid') excluded from all paid figures.
  - The drafted total comes from _aggregate_payee_buckets (UNFILTERED path) so that
    fully-drafted payees (owed==0, drafted>0) still contribute to drafted_total /
    draft_count but NOT to outstanding_total / payees_owed_count.
"""

from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

from oneclick.royalties import fx
from oneclick.royalties.models import (
    ArtistAnalyticsOut,
    MonthPoint,
    OverviewOut,
    PayeeAnalyticsOut,
    TopOwed,
)
from oneclick.royalties.service import (
    _aggregate_payee_buckets,
    _load_coverage_for_payee,
    _load_lines,
    _load_payees,
    _load_payouts,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ym(dt_str: str) -> str:
    """Extract 'YYYY-MM' from an ISO date/datetime string. Returns '' on failure."""
    if not dt_str:
        return ""
    try:
        return str(dt_str)[:7]
    except Exception:
        return ""


def _load_coverage_all(db, user_id: str) -> list[dict]:
    """Load all royalty_payout_coverage rows visible to user_id.

    Coverage rows are not directly user-scoped, so we join through payouts.
    We load all payouts for the user and then fetch coverage by payout_id in bulk.
    For the analytics use-case we load everything at once and filter in Python.
    """
    # Load all payout ids for this user, then get coverage
    payouts_res = db.table("royalty_payouts").select("id").eq("user_id", user_id).execute()
    payout_ids = [r["id"] for r in (payouts_res.data or [])]
    if not payout_ids:
        return []
    cov_res = db.table("royalty_payout_coverage").select("*").in_("payout_id", payout_ids).execute()
    return cov_res.data or []


def _load_projects_with_artist(db, project_ids) -> list[dict]:
    """Return {id, name, artist_id} for the given project ids.

    ``projects`` has no ``user_id`` column (ownership is via ``artist_id``); callers
    pass ids sourced from the user's own royalty_lines, keeping this user-scoped.
    """
    ids = [pid for pid in set(project_ids) if pid]
    if not ids:
        return []
    res = db.table("projects").select("id, name, artist_id").in_("id", ids).execute()
    return res.data or []


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------


def overview(db, user_id: str, base: str, now: date | None = None) -> "OverviewOut":
    """Compute portfolio-level analytics.

    - outstanding_total / payees_owed_count: net owed (owed>0) across payees.
    - drafted_total / draft_count: from UNFILTERED _aggregate_payee_buckets drafted
      so fully-drafted payees (owed==0, drafted>0) still count.
    - paid_total: Σ coverage.covered_amount over paid payouts (paid_at IS NOT NULL),
      each converted statement-ccy→base.
    - paid_last_30d: same, where paid_at >= now - 30d.
    - paid_by_month: [{month, amount}] bucketed by paid_at month.
    - top_owed: top 8 payees by net owed desc.
    - unconvertible_count: total buckets/amounts skipped.
    """
    if now is None:
        now = datetime.now(UTC).date()

    payees = _load_payees(db, user_id)
    if not payees:
        return OverviewOut(
            base=base,
            outstanding_total=0.0,
            payees_owed_count=0,
            drafted_total=0.0,
            draft_count=0,
            paid_total=0.0,
            paid_last_30d=0.0,
            paid_by_month=[],
            top_owed=[],
            unconvertible_count=0,
        )

    all_lines = _load_lines(db, user_id)
    all_payouts = _load_payouts(db, user_id)

    outstanding_total = 0.0
    payees_owed_count = 0
    drafted_total = 0.0
    draft_count = 0
    total_unconvertible = 0

    top_owed_list: list[dict] = []  # [{payee_id, display_name, owed}]

    for payee in payees:
        payee_id = payee["id"]
        coverage = _load_coverage_for_payee(db, payee_id)
        # UNFILTERED path — includes buckets even when owed==0 (drafted payees)
        totals = _aggregate_payee_buckets(payee, all_lines, all_payouts, coverage, db, base)

        total_unconvertible += totals["unconvertible_count"]

        owed = totals["owed"]
        drafted = totals["drafted"]

        if owed > 0:
            outstanding_total += owed
            payees_owed_count += 1
            top_owed_list.append({"payee_id": payee_id, "display_name": payee.get("display_name", ""), "owed": owed})

        if drafted > 0:
            drafted_total += drafted
            draft_count += 1

    # Sort top_owed descending and take top 8
    top_owed_list.sort(key=lambda x: x["owed"], reverse=True)
    top_owed_models = [
        TopOwed(payee_id=t["payee_id"], display_name=t["display_name"], owed=t["owed"]) for t in top_owed_list[:8]
    ]

    # Paid figures: load all coverage for this user's payouts in one shot, then
    # iterate coverage rows directly — no per-payee loop so no risk of double-counting.
    # Each coverage row is unique (payout_id, payee_id, royalty_statement_id, project_id).
    payout_by_id: dict[str, dict] = {p["id"]: p for p in all_payouts}

    all_coverage = _load_coverage_all(db, user_id)

    paid_total = 0.0
    paid_last_30d = 0.0
    paid_by_month_acc: dict[str, float] = defaultdict(float)

    cutoff_30d = now - timedelta(days=30)

    for cov in all_coverage:
        payout_id = cov.get("payout_id")
        payout = payout_by_id.get(payout_id, {})
        if payout.get("status") != "paid":
            continue
        paid_at_str = payout.get("paid_at")
        if not paid_at_str:
            continue

        # Derive statement currency from matching lines
        cov_payee_id = cov.get("payee_id", "")
        cov_stmt_id = cov.get("royalty_statement_id")
        cov_proj_id = cov.get("project_id", "")
        ccy = _get_stmt_ccy(all_lines, cov_payee_id, cov_stmt_id, cov_proj_id, base)

        # Parse paid_at month FIRST — a truthy-but-unparseable paid_at is skipped from
        # both paid_total and paid_by_month so they always foot (not unconvertible).
        try:
            paid_at_date = date.fromisoformat(str(paid_at_str)[:10])
        except (ValueError, TypeError):
            continue

        amt = float(cov.get("covered_amount") or 0)
        converted = fx.convert(db, amt, ccy, base, on_missing="none")
        if converted is None:
            total_unconvertible += 1
            continue

        paid_total += converted
        if paid_at_date >= cutoff_30d:
            paid_last_30d += converted
        month_key = f"{paid_at_date.year:04d}-{paid_at_date.month:02d}"
        paid_by_month_acc[month_key] += converted

    paid_by_month_sorted = sorted(paid_by_month_acc.items())
    paid_by_month_models = [MonthPoint(month=m, amount=a) for m, a in paid_by_month_sorted]

    return OverviewOut(
        base=base,
        outstanding_total=outstanding_total,
        payees_owed_count=payees_owed_count,
        drafted_total=drafted_total,
        draft_count=draft_count,
        paid_total=paid_total,
        paid_last_30d=paid_last_30d,
        paid_by_month=paid_by_month_models,
        top_owed=top_owed_models,
        unconvertible_count=total_unconvertible,
    )


def _get_stmt_ccy(all_lines: list[dict], payee_id: str, stmt_id: str, proj_id: str, fallback: str) -> str:
    """Look up the statement currency for a (payee, stmt, proj) bucket from lines."""
    for line in all_lines:
        if (
            line.get("payee_id") == payee_id
            and line.get("royalty_statement_id") == stmt_id
            and line.get("project_id", "") == proj_id
        ):
            return (line.get("statement_currency") or fallback).upper()
    return fallback.upper()


# ---------------------------------------------------------------------------
# artist_analytics
# ---------------------------------------------------------------------------


def artist_analytics(db, user_id: str, artist_id: str, base: str) -> "ArtistAnalyticsOut":
    """Compute analytics for a single artist.

    - summary: earned_total (from lines whose project.artist_id == artist_id),
               owed_now (net owed for that artist's lines),
               paid_total (from coverage of paid payouts, same artist filter).
    - by_month: [{month, earned, paid}] — earned by period_start month, paid by paid_at month.
    - unconvertible_count.
    """
    all_lines = _load_lines(db, user_id)
    all_payouts = _load_payouts(db, user_id)
    projects = _load_projects_with_artist(db, (l.get("project_id") for l in all_lines))

    # Build project_id → artist_id map
    proj_artist: dict[str, str] = {p["id"]: p.get("artist_id", "") for p in projects}

    # Filter lines to this artist's projects
    artist_lines = [l for l in all_lines if proj_artist.get(l.get("project_id", "")) == artist_id]

    # Payees that have lines in this artist's projects
    payee_ids_in_artist = {l["payee_id"] for l in artist_lines if l.get("payee_id")}

    # For owed_now: aggregate per payee, restrict to this artist's buckets
    # We replicate the bucket logic but filtered to artist projects
    payout_by_id: dict[str, dict] = {p["id"]: p for p in all_payouts}

    # Build coverage index for artist-relevant (payee, stmt, proj) buckets
    # We need all coverage for payees in this artist
    all_coverage_for_artist: list[dict] = []
    coverage_by_payee: dict[str, list[dict]] = {}
    for pid in payee_ids_in_artist:
        cov = _load_coverage_for_payee(db, pid)
        coverage_by_payee[pid] = cov
        all_coverage_for_artist.extend(cov)

    # Earned by period_start month
    earned_by_month: dict[str, float] = defaultdict(float)
    earned_total = 0.0
    unconvertible_count = 0

    # Group artist lines into (payee, stmt, proj) buckets for owed computation
    bucket_key: dict[tuple, list[dict]] = defaultdict(list)
    for line in artist_lines:
        pid = line.get("payee_id", "")
        sid = line.get("royalty_statement_id", "")
        proj = line.get("project_id", "")
        bucket_key[(pid, sid, proj)].append(line)

    owed_now = 0.0

    for (payee_id, stmt_id, proj_id), bucket_lines in bucket_key.items():
        ccy = (bucket_lines[0].get("statement_currency") or base).upper()
        earned_b = sum(float(bl.get("amount_owed") or 0) for bl in bucket_lines)

        # Coverage for this bucket
        cov_for_payee = coverage_by_payee.get(payee_id, [])
        paid_b = 0.0
        drafted_b = 0.0
        for cov in cov_for_payee:
            if cov.get("royalty_statement_id") == stmt_id and cov.get("project_id", "") == proj_id:
                p = payout_by_id.get(cov.get("payout_id", ""), {})
                amt = float(cov.get("covered_amount") or 0)
                if p.get("status") == "paid":
                    paid_b += amt
                elif p.get("status") == "draft":
                    drafted_b += amt

        owed_b = max(0.0, earned_b - paid_b - drafted_b)

        # Convert earned → base
        earned_conv = fx.convert(db, earned_b, ccy, base, on_missing="none")
        if earned_conv is None:
            unconvertible_count += 1
            continue

        earned_total += earned_conv

        # Earned by month (period_start of first line in bucket)
        period_start = bucket_lines[0].get("period_start") or ""
        month = _ym(period_start)
        if month:
            earned_by_month[month] += earned_conv

        # Owed
        owed_conv = fx.convert(db, owed_b, ccy, base, on_missing="none")
        if owed_conv is not None:
            owed_now += owed_conv

    # Paid from coverage (paid payouts, artist-scoped by project_id)
    paid_total = 0.0
    paid_by_month: dict[str, float] = defaultdict(float)

    for cov in all_coverage_for_artist:
        # Only count coverage for this artist's projects
        cov_proj = cov.get("project_id", "")
        if proj_artist.get(cov_proj) != artist_id:
            continue
        payout_id = cov.get("payout_id")
        payout = payout_by_id.get(payout_id, {})
        if payout.get("status") != "paid":
            continue
        paid_at_str = payout.get("paid_at")
        if not paid_at_str:
            continue

        # Parse paid_at month FIRST — an unparseable paid_at is skipped from both
        # paid_total and paid_by_month so they foot (not counted as unconvertible).
        try:
            paid_at_date = date.fromisoformat(str(paid_at_str)[:10])
        except (ValueError, TypeError):
            continue

        payee_id_cov = cov.get("payee_id", "")
        stmt_id_cov = cov.get("royalty_statement_id")
        ccy = _get_stmt_ccy(all_lines, payee_id_cov, stmt_id_cov, cov_proj, base)
        amt = float(cov.get("covered_amount") or 0)
        converted = fx.convert(db, amt, ccy, base, on_missing="none")
        if converted is None:
            unconvertible_count += 1
            continue

        paid_total += converted
        month = f"{paid_at_date.year:04d}-{paid_at_date.month:02d}"
        paid_by_month[month] += converted

    # Merge months from earned and paid
    all_months = sorted(set(earned_by_month.keys()) | set(paid_by_month.keys()))
    by_month_models = [
        MonthPoint(month=m, earned=earned_by_month.get(m, 0.0), paid=paid_by_month.get(m, 0.0)) for m in all_months
    ]

    return ArtistAnalyticsOut(
        artist_id=artist_id,
        base=base,
        summary={
            "earned_total": earned_total,
            "owed_now": owed_now,
            "paid_total": paid_total,
        },
        by_month=by_month_models,
        unconvertible_count=unconvertible_count,
    )


# ---------------------------------------------------------------------------
# payee_analytics
# ---------------------------------------------------------------------------


def payee_analytics(db, user_id: str, payee_id: str, base: str) -> "PayeeAnalyticsOut":
    """Compute analytics for a single payee.

    - summary: earned_total (from that payee's lines), paid_total (from that payee's
               paid coverage), owed (net owed).
    - by_month: [{month, earned, paid}].
    - unconvertible_count.
    """
    # Ownership check
    res = db.table("royalty_payees").select("*").eq("id", payee_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payee not found or not owned by caller")
    payee = rows[0]

    all_lines = _load_lines(db, user_id)
    all_payouts = _load_payouts(db, user_id)
    coverage = _load_coverage_for_payee(db, payee_id)

    payout_by_id: dict[str, dict] = {p["id"]: p for p in all_payouts}

    # Build (stmt, proj) coverage index
    cov_paid: dict[tuple, float] = defaultdict(float)
    cov_drafted: dict[tuple, float] = defaultdict(float)
    for cov in coverage:
        stmt_id = cov.get("royalty_statement_id")
        proj_id = cov.get("project_id", "")
        p = payout_by_id.get(cov.get("payout_id", ""), {})
        amt = float(cov.get("covered_amount") or 0)
        if p.get("status") == "paid":
            cov_paid[(stmt_id, proj_id)] += amt
        elif p.get("status") == "draft":
            cov_drafted[(stmt_id, proj_id)] += amt

    # Group payee lines into (stmt, proj) buckets
    payee_lines = [l for l in all_lines if l.get("payee_id") == payee_id]

    bucket_map: dict[tuple, list[dict]] = defaultdict(list)
    for line in payee_lines:
        sid = line.get("royalty_statement_id", "")
        proj = line.get("project_id", "")
        bucket_map[(sid, proj)].append(line)

    earned_total = 0.0
    owed_total = 0.0
    earned_by_month: dict[str, float] = defaultdict(float)
    unconvertible_count = 0

    for (stmt_id, proj_id), bucket_lines in bucket_map.items():
        ccy = (bucket_lines[0].get("statement_currency") or base).upper()
        earned_b = sum(float(bl.get("amount_owed") or 0) for bl in bucket_lines)
        paid_b = cov_paid.get((stmt_id, proj_id), 0.0)
        drafted_b = cov_drafted.get((stmt_id, proj_id), 0.0)
        owed_b = max(0.0, earned_b - paid_b - drafted_b)

        earned_conv = fx.convert(db, earned_b, ccy, base, on_missing="none")
        if earned_conv is None:
            unconvertible_count += 1
            continue

        earned_total += earned_conv

        period_start = bucket_lines[0].get("period_start") or ""
        month = _ym(period_start)
        if month:
            earned_by_month[month] += earned_conv

        owed_conv = fx.convert(db, owed_b, ccy, base, on_missing="none")
        if owed_conv is not None:
            owed_total += owed_conv

    # Paid from coverage
    paid_total = 0.0
    paid_by_month: dict[str, float] = defaultdict(float)

    for cov in coverage:
        p = payout_by_id.get(cov.get("payout_id", ""), {})
        if p.get("status") != "paid":
            continue
        paid_at_str = p.get("paid_at")
        if not paid_at_str:
            continue

        # Parse paid_at month FIRST — an unparseable paid_at is skipped from both
        # paid_total and paid_by_month so they foot (not counted as unconvertible).
        try:
            paid_at_date = date.fromisoformat(str(paid_at_str)[:10])
        except (ValueError, TypeError):
            continue

        stmt_id_cov = cov.get("royalty_statement_id")
        proj_id_cov = cov.get("project_id", "")
        ccy = _get_stmt_ccy(all_lines, payee_id, stmt_id_cov, proj_id_cov, base)
        amt = float(cov.get("covered_amount") or 0)
        converted = fx.convert(db, amt, ccy, base, on_missing="none")
        if converted is None:
            unconvertible_count += 1
            continue

        paid_total += converted
        month = f"{paid_at_date.year:04d}-{paid_at_date.month:02d}"
        paid_by_month[month] += converted

    # Merge months
    all_months = sorted(set(earned_by_month.keys()) | set(paid_by_month.keys()))
    by_month_models = [
        MonthPoint(month=m, earned=earned_by_month.get(m, 0.0), paid=paid_by_month.get(m, 0.0)) for m in all_months
    ]

    return PayeeAnalyticsOut(
        payee_id=payee_id,
        display_name=payee.get("display_name", ""),
        base=base,
        summary={
            "earned_total": earned_total,
            "paid_total": paid_total,
            "owed": owed_total,
        },
        by_month=by_month_models,
        unconvertible_count=unconvertible_count,
    )
