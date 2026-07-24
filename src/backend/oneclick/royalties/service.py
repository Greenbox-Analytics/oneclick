"""Aggregation service for OneClick royalties: payee summaries, detail, and period ledger.

Money-safety invariant:
  owed is computed **per (payee, royalty_statement_id, project_id) bucket in statement currency**,
  clamped to ≥ 0, and THEN converted to the reporting base and summed.
  Cross-bucket, cross-currency subtraction is never performed.

Payout invariant:
  royalty_payout_coverage.covered_amount is stored in STATEMENT currency (owed_b).
  Only royalty_payouts.total_amount and snapshot *_pay_ccy fields are in payout currency.
  NEVER store the converted payout-currency figure in covered_amount.
"""

import re
from collections import defaultdict
from datetime import UTC, date, datetime

from oneclick.royalties import fx, history, paypal_client
from oneclick.royalties.ingest import statement_meta, upsert_payee
from oneclick.royalties.models import (
    PayeeDetail,
    PayeeLine,
    PayeeProject,
    PayeeStatement,
    PayeeSummary,
    PeriodCell,
    PeriodLedger,
    PeriodLedgerRow,
)

# ---------------------------------------------------------------------------
# Name normalisation (mirrors ingest.normalize_name)
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_payees(db, user_id: str) -> list[dict]:
    res = db.table("royalty_payees").select("*").eq("user_id", user_id).execute()
    return res.data or []


def _load_lines(db, user_id: str) -> list[dict]:
    res = db.table("royalty_lines").select("*").eq("user_id", user_id).execute()
    return res.data or []


def _load_payouts(db, user_id: str) -> list[dict]:
    res = db.table("royalty_payouts").select("*").eq("user_id", user_id).execute()
    return res.data or []


def _load_coverage_for_payee(db, payee_id: str) -> list[dict]:
    """Load royalty_payout_coverage rows for a single payee (joined by payout_id)."""
    res = db.table("royalty_payout_coverage").select("*").eq("payee_id", payee_id).execute()
    return res.data or []


def _project_name_map(db, project_ids) -> dict[str, str]:
    """Return {project_id: name} for the given ids.

    ``projects`` has no ``user_id`` column (ownership is via ``artist_id``), so we
    look names up by the ids themselves. Callers pass ids sourced from the user's
    own royalty_lines, which keeps the result correctly user-scoped.
    """
    ids = [pid for pid in set(project_ids) if pid]
    if not ids:
        return {}
    res = db.table("projects").select("id, name").in_("id", ids).execute()
    return {r["id"]: r.get("name", "") for r in (res.data or [])}


def _detect_collision(db, user_id: str, normalized_name: str) -> bool:
    """Return True if the payee's normalized_name maps to >1 distinct person
    in registry_collaborators (scoped to invited_by == user_id).

    Distinctness key: collaborator_user_id when present, else lower(email).
    One person across many works is NOT a collision.
    """
    res = (
        db.table("registry_collaborators")
        .select("name, email, collaborator_user_id")
        .eq("invited_by", user_id)
        .execute()
    )
    rows = res.data or []

    # Filter to rows whose normalized name matches
    matching = [r for r in rows if _normalize_name(r.get("name", "")) == normalized_name]

    identities: set[str] = set()
    for r in matching:
        cuid = r.get("collaborator_user_id")
        if cuid:
            identities.add(f"uid:{cuid}")
        else:
            email = (r.get("email") or "").lower().strip()
            if email:
                identities.add(f"email:{email}")

    return len(identities) > 1


# ---------------------------------------------------------------------------
# Bucket-level aggregation (the money-safe core)
# ---------------------------------------------------------------------------


def _bucket_state(earned_b: float, paid_b: float, drafted_b: float) -> str:
    """Single source of truth for a bucket's derived state — used by
    payee_summary, periods_ledger, and payee_detail."""
    if earned_b - paid_b - drafted_b > 0.01:
        return "owed"
    if paid_b > earned_b + 0.01:
        return "overpaid"
    if drafted_b > 0:
        return "scheduled"
    return "settled"


def _aggregate_payee_buckets(payee: dict, lines: list[dict], payouts: list[dict], coverage: list[dict], db, base: str):
    """Return aggregated totals (base-currency and payout-currency) for one payee.

    Per-bucket steps:
      1. earned_b  = Σ amount_owed  (statement ccy)
      2. paid_b    = Σ covered_amount where payout.status == 'paid'  (statement ccy)
      3. drafted_b = Σ covered_amount where payout.status == 'draft' (statement ccy)
      4. owed_b    = max(0, earned_b - paid_b - drafted_b)           (statement ccy, CLAMPED)
      5. Convert each of earned_b/paid_b/drafted_b/owed_b → base → accumulate
      6. Convert each of earned_b/paid_b/drafted_b/owed_b → payout_currency → accumulate

    Overpayment credit (credit_by_ccy) is derived separately, from PAID coverage
    only (never drafted — a draft can be canceled, cascading its coverage away,
    which would make drafted-backed credit phantom). It is tracked per statement
    currency and never FX-netted across buckets, same as owed/unpaid.
    """
    payee_id = payee["id"]
    payout_ccy = payee.get("payout_currency") or base

    # Index payouts by id and by status
    payout_by_id: dict[str, dict] = {p["id"]: p for p in payouts}

    # Build (payee, statement, project) → {paid, drafted} coverage sums
    # coverage rows are for THIS payee already (loaded by payee_id)
    # Key: (royalty_statement_id, project_id) — matches the bucket granularity.
    cov_paid: dict[tuple, float] = defaultdict(float)
    cov_drafted: dict[tuple, float] = defaultdict(float)
    for cov in coverage:
        stmt_id = cov.get("royalty_statement_id")
        proj_id = cov.get("project_id", "")
        payout_id = cov.get("payout_id")
        payout = payout_by_id.get(payout_id, {})
        status = payout.get("status")
        amt = float(cov.get("covered_amount") or 0)
        if status == "paid":
            cov_paid[(stmt_id, proj_id)] += amt
        elif status == "draft":
            cov_drafted[(stmt_id, proj_id)] += amt

    # Group lines by (royalty_statement_id, project_id) bucket
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for line in lines:
        if line.get("payee_id") == payee_id:
            stmt_id = line.get("royalty_statement_id", "")
            proj_id = line.get("project_id", "")
            buckets[(stmt_id, proj_id)].append(line)

    # Reporting-currency accumulators
    earned_r = paid_r = drafted_r = owed_r = unpaid_r = 0.0
    # Native (payout_currency) accumulators
    earned_n = paid_n = drafted_n = owed_n = unpaid_n = 0.0
    # Overpayment credit, per statement currency (never FX-netted across buckets)
    credit_by_ccy: dict[str, float] = {}

    # Collect project ids
    project_ids: set[str] = set()

    # Count buckets whose statement currency is not convertible to the reporting base
    unconvertible_count = 0

    for (stmt_id, proj_id), bucket_lines in buckets.items():
        ccy = (bucket_lines[0].get("statement_currency") or base).upper()
        earned_b = sum(float(bl.get("amount_owed") or 0) for bl in bucket_lines)
        paid_b = cov_paid.get((stmt_id, proj_id), 0.0)
        drafted_b = cov_drafted.get((stmt_id, proj_id), 0.0)
        owed_b = max(0.0, earned_b - paid_b - drafted_b)
        # Outstanding = still owed until actually paid. Unlike owed_b it does NOT
        # subtract drafted amounts — a draft is a plan, not a payment, so the
        # money stays outstanding until the payout is completed.
        unpaid_b = max(0.0, earned_b - paid_b)

        # Overpayment credit — derived from PAID coverage only (a draft can be
        # canceled, cascading its coverage away; credit backed by it would be
        # phantom). Tracked per statement currency: coverage amounts are frozen
        # in statement ccy and are never FX-netted (service invariant, line 8).
        overpaid_b = max(0.0, paid_b - earned_b)
        # Epsilon, not zero: float residue from coverage slicing must not
        # surface as a $0.00 "Overpaid" chip.
        if overpaid_b > 0.01:
            credit_by_ccy[ccy] = credit_by_ccy.get(ccy, 0.0) + overpaid_b

        # Convert each component → reporting base (on_missing="none" → skip unconvertible buckets)
        earned_conv = fx.convert(db, earned_b, ccy, base, on_missing="none")
        if earned_conv is None:
            unconvertible_count += 1
            continue
        paid_conv = fx.convert(db, paid_b, ccy, base, on_missing="none")
        drafted_conv = fx.convert(db, drafted_b, ccy, base, on_missing="none")
        owed_conv = fx.convert(db, owed_b, ccy, base, on_missing="none")
        unpaid_conv = fx.convert(db, unpaid_b, ccy, base, on_missing="none")

        earned_r += earned_conv
        paid_r += paid_conv if paid_conv is not None else 0.0
        drafted_r += drafted_conv if drafted_conv is not None else 0.0
        owed_r += owed_conv if owed_conv is not None else 0.0
        unpaid_r += unpaid_conv if unpaid_conv is not None else 0.0

        # Convert each component → payout_currency (native) — use default "amount" fallback
        earned_n += fx.convert(db, earned_b, ccy, payout_ccy)
        paid_n += fx.convert(db, paid_b, ccy, payout_ccy)
        drafted_n += fx.convert(db, drafted_b, ccy, payout_ccy)
        owed_n += fx.convert(db, owed_b, ccy, payout_ccy)
        unpaid_n += fx.convert(db, unpaid_b, ccy, payout_ccy)

        if proj_id:
            project_ids.add(proj_id)

    return {
        "earned": earned_r,
        "paid": paid_r,
        "drafted": drafted_r,
        "owed": owed_r,
        "unpaid": unpaid_r,
        "earned_native": earned_n,
        "paid_native": paid_n,
        "drafted_native": drafted_n,
        "owed_native": owed_n,
        "unpaid_native": unpaid_n,
        "project_ids": project_ids,
        "unconvertible_count": unconvertible_count,
        "credit_by_ccy": credit_by_ccy,
    }


# ---------------------------------------------------------------------------
# Public aggregation functions
# ---------------------------------------------------------------------------


def payee_summary(db, user_id: str, base: str = "USD") -> list[dict]:
    """Return a list of PayeeSummary-shaped dicts for all payees owned by user_id."""
    payees = _load_payees(db, user_id)
    if not payees:
        return []

    all_lines = _load_lines(db, user_id)
    all_payouts = _load_payouts(db, user_id)

    results = []
    for payee in payees:
        payee_id = payee["id"]
        coverage = _load_coverage_for_payee(db, payee_id)
        totals = _aggregate_payee_buckets(payee, all_lines, all_payouts, coverage, db, base)

        owed = totals["owed"]
        drafted = totals["drafted"]
        credit_by_ccy = totals["credit_by_ccy"]
        if owed > 0:
            status = "owed"
        elif credit_by_ccy:
            status = "overpaid"
        elif drafted > 0:
            status = "scheduled"
        else:
            status = "settled"

        collision = _detect_collision(db, user_id, payee.get("normalized_name", ""))

        results.append(
            PayeeSummary(
                id=payee_id,
                display_name=payee.get("display_name", ""),
                payout_currency=payee.get("payout_currency") or base,
                registry_user_id=payee.get("registry_user_id"),
                email=payee.get("email"),
                collision=collision,
                project_count=len(totals["project_ids"]),
                status=status,
                earned=totals["earned"],
                paid=totals["paid"],
                drafted=totals["drafted"],
                owed=owed,
                unpaid=totals["unpaid"],
                earned_native=totals["earned_native"],
                paid_native=totals["paid_native"],
                drafted_native=totals["drafted_native"],
                owed_native=totals["owed_native"],
                unpaid_native=totals["unpaid_native"],
                unconvertible_count=totals["unconvertible_count"],
                credit_by_ccy=credit_by_ccy,
            ).model_dump()
        )

    return results


def payee_detail(db, user_id: str, payee_id: str, base: str = "USD") -> dict:
    """Return a PayeeDetail-shaped dict for a single payee.

    Raises PermissionError if the payee doesn't belong to user_id (caller should → 404).
    """
    # Ownership check
    res = db.table("royalty_payees").select("*").eq("id", payee_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("not found")
    payee = rows[0]

    all_lines = _load_lines(db, user_id)
    all_payouts = _load_payouts(db, user_id)
    coverage = _load_coverage_for_payee(db, payee_id)

    # Summary
    totals = _aggregate_payee_buckets(payee, all_lines, all_payouts, coverage, db, base)
    owed = totals["owed"]
    drafted = totals["drafted"]
    credit_by_ccy = totals["credit_by_ccy"]
    if owed > 0:
        status = "owed"
    elif credit_by_ccy:
        status = "overpaid"
    elif drafted > 0:
        status = "scheduled"
    else:
        status = "settled"
    collision = _detect_collision(db, user_id, payee.get("normalized_name", ""))

    summary = PayeeSummary(
        id=payee_id,
        display_name=payee.get("display_name", ""),
        payout_currency=payee.get("payout_currency") or base,
        registry_user_id=payee.get("registry_user_id"),
        email=payee.get("email"),
        collision=collision,
        project_count=len(totals["project_ids"]),
        status=status,
        earned=totals["earned"],
        paid=totals["paid"],
        drafted=totals["drafted"],
        owed=owed,
        unpaid=totals["unpaid"],
        earned_native=totals["earned_native"],
        paid_native=totals["paid_native"],
        drafted_native=totals["drafted_native"],
        owed_native=totals["owed_native"],
        unpaid_native=totals["unpaid_native"],
        unconvertible_count=totals["unconvertible_count"],
        credit_by_ccy=credit_by_ccy,
    )

    # Build projects → statements → lines tree
    # Group payee lines by project_id → statement_id
    payee_lines = [l for l in all_lines if l.get("payee_id") == payee_id]

    # Index payouts for coverage lookup
    # Key: (royalty_statement_id, project_id) — matches bucket granularity so that
    # a statement spanning multiple projects does not blur coverage across them.
    payout_by_id: dict[str, dict] = {p["id"]: p for p in all_payouts}
    cov_paid_stmt: dict[tuple, float] = defaultdict(float)
    cov_drafted_stmt: dict[tuple, float] = defaultdict(float)
    for cov in coverage:
        stmt_id = cov.get("royalty_statement_id")
        proj_id = cov.get("project_id", "")
        payout_id = cov.get("payout_id")
        payout = payout_by_id.get(payout_id, {})
        st = payout.get("status")
        amt = float(cov.get("covered_amount") or 0)
        if st == "paid":
            cov_paid_stmt[(stmt_id, proj_id)] += amt
        elif st == "draft":
            cov_drafted_stmt[(stmt_id, proj_id)] += amt

    # Project names (looked up by the ids present in this payee's lines)
    project_name_map: dict[str, str] = _project_name_map(db, (l.get("project_id") for l in payee_lines))

    # Organise by project → statement
    proj_stmt_lines: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in payee_lines:
        pid = line.get("project_id", "")
        sid = line.get("royalty_statement_id", "")
        proj_stmt_lines[pid][sid].append(line)

    projects_out: list[PayeeProject] = []
    for pid, stmt_map in proj_stmt_lines.items():
        statements_out: list[PayeeStatement] = []
        for sid, slines in stmt_map.items():
            ccy = (slines[0].get("statement_currency") or base).upper()
            p_start = slines[0].get("period_start")
            p_end = slines[0].get("period_end")
            calc_id = slines[0].get("calculation_id")

            earned_b = sum(float(sl.get("amount_owed") or 0) for sl in slines)
            paid_b = cov_paid_stmt.get((sid, pid), 0.0)
            drafted_b = cov_drafted_stmt.get((sid, pid), 0.0)
            owed_b = max(0.0, earned_b - paid_b - drafted_b)
            unpaid_b = max(0.0, earned_b - paid_b)
            st_state = _bucket_state(earned_b, paid_b, drafted_b)

            # statement_total from statement_meta (may be None)
            st_meta = None
            if calc_id:
                try:
                    st_meta = statement_meta(db, calc_id)
                except Exception:
                    st_meta = None
            stmt_total = st_meta.get("statement_total") if st_meta else None

            lines_out = [
                PayeeLine(
                    line_id=sl.get("id", ""),
                    song_title=sl.get("song_title", ""),
                    role=sl.get("role"),
                    royalty_type=sl.get("royalty_type"),
                    percentage=sl.get("percentage"),
                    amount_owed=float(sl.get("amount_owed") or 0),
                    statement_currency=sl.get("statement_currency", ""),
                )
                for sl in slines
            ]

            statements_out.append(
                PayeeStatement(
                    royalty_statement_id=sid,
                    period_start=p_start,
                    period_end=p_end,
                    statement_currency=ccy,
                    statement_total=stmt_total,
                    earned=earned_b,
                    paid=paid_b,
                    drafted=drafted_b,
                    owed=owed_b,
                    unpaid=unpaid_b,
                    state=st_state,
                    lines=lines_out,
                )
            )

        projects_out.append(
            PayeeProject(
                project_id=pid,
                name=project_name_map.get(pid, ""),
                statements=statements_out,
            )
        )

    # Payouts for this payee
    payee_payouts = [p for p in all_payouts if p.get("payee_id") == payee_id]

    return PayeeDetail(
        summary=summary,
        projects=projects_out,
        payouts=payee_payouts,
    ).model_dump()


def periods_ledger(db, user_id: str, base: str = "USD") -> dict:
    """Return a PeriodLedger-shaped dict: payee × statement-period matrix.

    Each cell contains earned (in base), state, and period dates.
    Row total = Σ earned (base) across all the payee's cells.
    """
    payees = _load_payees(db, user_id)
    all_lines = _load_lines(db, user_id)
    all_payouts = _load_payouts(db, user_id)

    rows_out: list[PeriodLedgerRow] = []

    for payee in payees:
        payee_id = payee["id"]
        coverage = _load_coverage_for_payee(db, payee_id)

        payee_lines = [l for l in all_lines if l.get("payee_id") == payee_id]
        if not payee_lines:
            continue

        # Index payouts
        # Key: (royalty_statement_id, project_id) — matches the bucket granularity so that
        # a statement spanning multiple projects does not blur coverage (and thus state)
        # across them.
        payout_by_id: dict[str, dict] = {p["id"]: p for p in all_payouts}
        cov_paid: dict[tuple, float] = defaultdict(float)
        cov_drafted: dict[tuple, float] = defaultdict(float)
        for cov in coverage:
            stmt_id = cov.get("royalty_statement_id")
            proj_id = cov.get("project_id", "")
            payout_id = cov.get("payout_id")
            payout = payout_by_id.get(payout_id, {})
            st = payout.get("status")
            amt = float(cov.get("covered_amount") or 0)
            if st == "paid":
                cov_paid[(stmt_id, proj_id)] += amt
            elif st == "draft":
                cov_drafted[(stmt_id, proj_id)] += amt

        # Group by (statement, project) bucket
        stmt_lines: dict[tuple, list[dict]] = defaultdict(list)
        for line in payee_lines:
            stmt_id = line.get("royalty_statement_id", "")
            proj_id = line.get("project_id", "")
            stmt_lines[(stmt_id, proj_id)].append(line)

        cells: list[PeriodCell] = []
        row_total = 0.0
        unconvertible_count = 0

        for (stmt_id, proj_id), slines in stmt_lines.items():
            ccy = (slines[0].get("statement_currency") or base).upper()
            earned_b = sum(float(sl.get("amount_owed") or 0) for sl in slines)
            paid_b = cov_paid.get((stmt_id, proj_id), 0.0)
            drafted_b = cov_drafted.get((stmt_id, proj_id), 0.0)
            state = _bucket_state(earned_b, paid_b, drafted_b)

            earned_base = fx.convert(db, earned_b, ccy, base, on_missing="none")
            if earned_base is None:
                unconvertible_count += 1
                earned_base = 0.0
            else:
                row_total += earned_base

            cells.append(
                PeriodCell(
                    royalty_statement_id=stmt_id,
                    period_start=slines[0].get("period_start"),
                    period_end=slines[0].get("period_end"),
                    earned=earned_base,
                    state=state,
                )
            )

        rows_out.append(
            PeriodLedgerRow(
                payee_id=payee_id,
                display_name=payee.get("display_name", ""),
                cells=cells,
                total=row_total,
                unconvertible_count=unconvertible_count,
            )
        )

    return PeriodLedger(base=base, rows=rows_out).model_dump()


# ---------------------------------------------------------------------------
# Shared owed-bucket helper (used by payee_summary and payout creation)
# ---------------------------------------------------------------------------


def payee_owed_buckets(db, user_id: str, payee_id: str) -> list[dict]:
    """Return the owed buckets for a single payee, filtered to owed_b > 0."""
    return _payee_buckets(db, user_id, payee_id)[0]


def _payee_buckets(db, user_id: str, payee_id: str) -> tuple[list[dict], list[dict]]:
    """Return (owed, overpaid) bucket lists for a single payee, from one pass.

    owed entries (owed_b > 0):
      royalty_statement_id, project_id, calculation_id,
      ccy (statement currency), owed_b (statement-ccy amount, > 0),
      lines (the bucket's royalty_lines rows).

    overpaid entries (paid_b - earned_b > 0.01) — the credit side:
      royalty_statement_id, project_id, ccy, excess (statement ccy),
      coverage_rows (the bucket's PAID payout coverage rows — whose money the
      excess is). PAID coverage only, mirroring _aggregate_payee_buckets'
      credit_by_ccy: drafted-backed credit would be phantom (a draft can be
      canceled, cascading its coverage away). Buckets with no remaining lines
      contribute no credit, same as the read path.

    Coverage is loaded from royalty_payout_coverage for this payee, indexed
    by (statement, project) to compute the per-bucket clamp identical to _aggregate_payee_buckets.
    """
    # Load payee lines for this payee only
    lines_res = db.table("royalty_lines").select("*").eq("user_id", user_id).eq("payee_id", payee_id).execute()
    all_lines = lines_res.data or []
    if not all_lines:
        return [], []

    # Load all payouts for this user (needed to resolve coverage status)
    all_payouts = _load_payouts(db, user_id)
    payout_by_id: dict[str, dict] = {p["id"]: p for p in all_payouts}

    # Load coverage for this payee
    coverage = _load_coverage_for_payee(db, payee_id)

    # Build per-(statement, project) coverage sums (paid + drafted = "spoken for")
    # Key: (royalty_statement_id, project_id) — matches the bucket granularity so that
    # a statement spanning multiple projects does not blur coverage across them.
    cov_paid: dict[tuple, float] = defaultdict(float)
    cov_drafted: dict[tuple, float] = defaultdict(float)
    cov_paid_rows: dict[tuple, list[dict]] = defaultdict(list)
    for cov in coverage:
        stmt_id = cov.get("royalty_statement_id")
        proj_id = cov.get("project_id", "")
        payout_id = cov.get("payout_id")
        payout = payout_by_id.get(payout_id, {})
        status = payout.get("status")
        amt = float(cov.get("covered_amount") or 0)
        if status == "paid":
            cov_paid[(stmt_id, proj_id)] += amt
            cov_paid_rows[(stmt_id, proj_id)].append(cov)
        elif status == "draft":
            cov_drafted[(stmt_id, proj_id)] += amt

    # Group lines by (statement, project) bucket
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for line in all_lines:
        stmt_id = line.get("royalty_statement_id", "")
        proj_id = line.get("project_id", "")
        buckets[(stmt_id, proj_id)].append(line)

    owed: list[dict] = []
    overpaid: list[dict] = []
    for (stmt_id, proj_id), bucket_lines in buckets.items():
        ccy = (bucket_lines[0].get("statement_currency") or "USD").upper()
        earned_b = sum(float(bl.get("amount_owed") or 0) for bl in bucket_lines)
        paid_b = cov_paid.get((stmt_id, proj_id), 0.0)
        drafted_b = cov_drafted.get((stmt_id, proj_id), 0.0)
        owed_b = max(0.0, earned_b - paid_b - drafted_b)

        if paid_b - earned_b > 0.01:
            overpaid.append(
                {
                    "royalty_statement_id": stmt_id,
                    "project_id": proj_id,
                    "ccy": ccy,
                    "excess": paid_b - earned_b,
                    "coverage_rows": cov_paid_rows.get((stmt_id, proj_id), []),
                }
            )

        if owed_b <= 0:
            continue

        owed.append(
            {
                "royalty_statement_id": stmt_id,
                "project_id": proj_id,
                "calculation_id": bucket_lines[0].get("calculation_id"),
                "ccy": ccy,
                "owed_b": owed_b,
                "lines": bucket_lines,
            }
        )

    return owed, overpaid


# ---------------------------------------------------------------------------
# Payout creation and management
# ---------------------------------------------------------------------------


class StaleSourcesError(Exception):
    def __init__(self, lines):
        super().__init__("stale sources")
        self.lines = lines


def _stale_lines(db, buckets) -> list[dict]:
    """Lines with NO live source: every source id dead AND no live project file
    shares any stored source hash (dead ids alongside a live source are normal
    history and never warn)."""
    all_ids, all_hashes = set(), set()
    for b in buckets:
        for line in b["lines"]:
            for e in line.get("source_contracts") or []:
                if e.get("id"):
                    all_ids.add(e["id"])
                if e.get("hash"):
                    all_hashes.add(e["hash"])
    if not all_ids:
        return []
    live = db.table("project_files").select("id, content_hash").in_("id", list(all_ids)).execute().data or []
    live_ids = {r["id"] for r in live}
    live_hashes = set()
    if all_hashes:
        # Scope hash-liveness to THIS user's projects — the service-role client
        # bypasses RLS, and another user's identical file must not count as live.
        project_ids = list({b["project_id"] for b in buckets if b.get("project_id")})
        hash_rows = (
            db.table("project_files")
            .select("content_hash")
            .in_("content_hash", list(all_hashes))
            .in_("project_id", project_ids)
            .execute()
        ).data or []
        live_hashes = {r["content_hash"] for r in hash_rows}
    stale = []
    for b in buckets:
        for line in b["lines"]:
            entries = line.get("source_contracts") or []
            if entries and all(e.get("id") not in live_ids and e.get("hash") not in live_hashes for e in entries):
                stale.append({"song": line.get("song_title"), "line_id": line.get("id")})
    return stale


def create_payouts(
    db,
    user_id: str,
    payee_ids: list[str],
    idempotency_key: str | None,
    note: str | None,
    force: bool = False,
) -> list[dict]:
    """Create draft payouts for one or more payees.

    For each payee_id:
    - Assert the payee belongs to the caller.
    - Compute owed buckets; skip if empty (no $0 invoices).
    - Idempotency: if derived_key already exists, return the existing payout.
    - Staleness gate: raise StaleSourcesError if any owed line has no live
      source contract (unless force=True).
    - Net same-currency overpayment credit against owed buckets by re-allocating
      PAID coverage; skip the payee entirely if credit covers everything.
    - Insert royalty_payouts + royalty_payout_coverage rows.

    CRITICAL invariant: coverage.covered_amount is stored in STATEMENT currency (owed_b),
    NOT in the converted payout currency. Only total_amount and snapshot *_pay_ccy fields
    are in payout currency.
    """
    today = date.today().isoformat()
    results = []

    # Load project names once (for snapshot), keyed by the ids in the user's lines.
    line_res = db.table("royalty_lines").select("project_id").eq("user_id", user_id).execute()
    project_name_map: dict[str, str] = _project_name_map(db, (r.get("project_id") for r in (line_res.data or [])))

    for payee_id in payee_ids:
        # Ownership check
        payee_res = db.table("royalty_payees").select("*").eq("id", payee_id).eq("user_id", user_id).execute()
        payee_rows = payee_res.data or []
        if not payee_rows:
            raise PermissionError(f"Payee {payee_id} not found or not owned by caller")
        payee = payee_rows[0]

        # Compute owed buckets + overpayment credits in one pass
        buckets, credits = _payee_buckets(db, user_id, payee_id)
        if not buckets:
            continue  # skip — no $0 invoices

        payout_ccy = (payee.get("payout_currency") or "USD").upper()

        # Idempotency: derive a per-payee key and check for existing payout.
        # Checked BEFORE the credit moves so a retried request returns the
        # existing payout without re-applying side effects.
        derived_key = f"{idempotency_key}:{payee_id}" if idempotency_key else None
        if derived_key:
            existing_res = (
                db.table("royalty_payouts")
                .select("*")
                .eq("user_id", user_id)
                .eq("idempotency_key", derived_key)
                .execute()
            )
            existing = existing_res.data or []
            if existing:
                results.append(existing[0])
                continue  # idempotent — do not create a second

        stale = _stale_lines(db, buckets)
        if stale and not force:
            raise StaleSourcesError(stale)

        # Apply same-currency credits: excess PAID coverage re-allocates from
        # overpaid buckets onto owed buckets, attributed to the ORIGINAL payout
        # (that's whose money it was). Never FX-converted.
        for bucket in buckets:
            for credit in credits:
                if credit["ccy"] != bucket["ccy"] or credit["excess"] <= 0.01 or bucket["owed_b"] <= 0.01:
                    continue
                take = min(credit["excess"], bucket["owed_b"])
                for cov in credit["coverage_rows"]:
                    if take <= 0.01:
                        break
                    avail = float(cov["covered_amount"])
                    if avail <= 0:
                        continue  # already fully re-allocated onto an earlier bucket
                    slice_amt = min(take, avail)
                    history.record(db, user_id, "coverage_moved", dict(cov), "payout_credit")
                    db.table("royalty_payout_coverage").update({"covered_amount": avail - slice_amt}).eq(
                        "id", cov["id"]
                    ).execute()
                    # Keep the local view fresh: a later bucket slicing the same
                    # row must see the reduced amount, not the loaded snapshot.
                    cov["covered_amount"] = avail - slice_amt
                    moved_from = {
                        "statement_id": cov["royalty_statement_id"],
                        "project_id": cov["project_id"],
                        "action": "payout_credit",
                    }
                    # Composite PK forbids two rows per (payout, statement,
                    # project): UPDATE-add when the original payout already
                    # covers the target bucket, INSERT otherwise.
                    existing_target = (
                        db.table("royalty_payout_coverage")
                        .select("*")
                        .eq("payout_id", cov["payout_id"])
                        .eq("royalty_statement_id", bucket["royalty_statement_id"])
                        .eq("project_id", bucket["project_id"])
                        .execute()
                    ).data or []
                    if existing_target:
                        # moved_from set here too: the own-coverage check in the
                        # revert guard is then sufficient on every path.
                        db.table("royalty_payout_coverage").update(
                            {
                                "covered_amount": float(existing_target[0]["covered_amount"]) + slice_amt,
                                "moved_from": moved_from,
                            }
                        ).eq("id", existing_target[0]["id"]).execute()
                    else:
                        db.table("royalty_payout_coverage").insert(
                            {
                                "payout_id": cov["payout_id"],
                                "payee_id": payee_id,
                                "project_id": bucket["project_id"],
                                "royalty_statement_id": bucket["royalty_statement_id"],
                                "covered_amount": slice_amt,
                                "moved_from": moved_from,
                            }
                        ).execute()
                    take -= slice_amt
                    credit["excess"] -= slice_amt
                    bucket["owed_b"] -= slice_amt
        buckets = [b for b in buckets if b["owed_b"] > 0.01]
        if not buckets:
            continue  # fully covered by credit (or nothing owed) — no payout row

        # Convert each bucket's owed_b → payout_ccy and accumulate total
        total = 0.0
        rates_used: dict[str, float] = {}
        bucket_pay_amts: list[float] = []  # parallel to buckets list

        for bucket in buckets:
            owed_b = bucket["owed_b"]
            ccy = bucket["ccy"]
            pay_amt = fx.convert(db, owed_b, ccy, payout_ccy)
            bucket_pay_amts.append(pay_amt)
            total += pay_amt
            # Track rate used for snapshot (only for cross-currency conversions)
            if ccy.upper() != payout_ccy.upper() and owed_b != 0:
                rate = pay_amt / owed_b
                rates_used[f"{ccy}->{payout_ccy}"] = rate

        # Build breakdown_snapshot
        # Group buckets by project_id
        proj_buckets: dict[str, list] = defaultdict(list)
        for i, bucket in enumerate(buckets):
            proj_buckets[bucket["project_id"]].append((bucket, bucket_pay_amts[i]))

        snapshot_projects = []
        for proj_id, proj_bucket_list in proj_buckets.items():
            snapshot_stmts = []
            for bucket, pay_amt in proj_bucket_list:
                ccy = bucket["ccy"]
                calc_id = bucket["calculation_id"]

                # statement_total from statement_meta
                stmt_total = None
                if calc_id:
                    try:
                        meta = statement_meta(db, calc_id)
                        stmt_total = meta.get("statement_total")
                    except Exception:
                        pass

                # Build lines with both statement-ccy and pay-ccy amounts
                snapshot_lines = []
                bucket_owed_b = bucket["owed_b"]
                for line in bucket["lines"]:
                    line_amt = float(line.get("amount_owed") or 0)
                    # Convert line amount to pay_ccy proportionally
                    line_pay_ccy = fx.convert(db, line_amt, ccy, payout_ccy)
                    snapshot_lines.append(
                        {
                            "song": line.get("song_title", ""),
                            "role": line.get("role"),
                            "royalty_type": line.get("royalty_type"),
                            "percentage": line.get("percentage"),
                            "amount_owed": line_amt,  # statement ccy
                            "amount_pay_ccy": line_pay_ccy,  # payout ccy
                        }
                    )

                snapshot_stmts.append(
                    {
                        "royalty_statement_id": bucket["royalty_statement_id"],
                        "period_start": bucket["lines"][0].get("period_start"),
                        "period_end": bucket["lines"][0].get("period_end"),
                        "statement_currency": ccy,
                        "statement_total": stmt_total,
                        "payee_subtotal_owed": bucket_owed_b,  # statement ccy
                        "payee_subtotal_pay_ccy": pay_amt,  # payout ccy
                        "lines": snapshot_lines,
                    }
                )

            snapshot_projects.append(
                {
                    "project_id": proj_id,
                    "name": project_name_map.get(proj_id, ""),
                    "statements": snapshot_stmts,
                }
            )

        breakdown_snapshot = {
            "payee": {
                "id": payee_id,
                "display_name": payee.get("display_name", ""),
                "payout_currency": payout_ccy,
            },
            "fx": {
                "rate_date": today,
                "rates_used": rates_used,
            },
            "projects": snapshot_projects,
            "total_pay_ccy": total,
        }

        # Insert the payout
        payout_row = {
            "user_id": user_id,
            "payee_id": payee_id,
            "status": "draft",
            "pay_currency": payout_ccy,
            "fx_rate_date": today,
            "total_amount": total,
            "breakdown_snapshot": breakdown_snapshot,
            "note": note,
        }
        if derived_key:
            payout_row["idempotency_key"] = derived_key

        insert_res = db.table("royalty_payouts").insert(payout_row).execute()
        inserted_payout = (insert_res.data or [{}])[0]
        payout_id = inserted_payout.get("id")

        # Insert coverage rows — covered_amount is STATEMENT CURRENCY (owed_b)
        coverage_rows = [
            {
                "payout_id": payout_id,
                "payee_id": payee_id,
                "project_id": bucket["project_id"],
                "royalty_statement_id": bucket["royalty_statement_id"],
                "covered_amount": bucket["owed_b"],  # STATEMENT CCY — never payout ccy
            }
            for bucket in buckets
        ]
        db.table("royalty_payout_coverage").insert(coverage_rows).execute()

        results.append(inserted_payout)

    return results


def mark_paid(db, user_id: str, payout_id: str) -> dict:
    """Mark a payout as paid. Raises PermissionError if not the caller's payout."""
    res = db.table("royalty_payouts").select("*").eq("id", payout_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payout not found")

    update_res = (
        db.table("royalty_payouts")
        .update({"status": "paid", "paid_at": "now()"})
        .eq("id", payout_id)
        .eq("user_id", user_id)
        .execute()
    )
    updated = (update_res.data or [{}])[0]
    return updated


def cancel_payout(db, user_id: str, payout_id: str) -> None:
    """Cancel (delete) a draft payout. Coverage cascades on delete.

    Raises PermissionError for ownership violations.
    Raises ValueError if payout is not in 'draft' status.
    """
    res = db.table("royalty_payouts").select("*").eq("id", payout_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payout not found")
    payout = rows[0]
    if payout.get("status") != "draft":
        raise ValueError("only drafts can be canceled")

    db.table("royalty_payouts").delete().eq("id", payout_id).eq("user_id", user_id).execute()


def revert_payout_to_draft(db, user_id: str, payout_id: str) -> dict:
    """Revert a manually-completed payout back to draft (undo an accidental
    mark-paid). The coverage rows stay attached and move from paid→drafted on
    their own, because the paid/drafted split is derived from the payout status.

    Only *manual* payouts can be reverted: a payout completed through PayPal moved
    real money on PayPal's side, and flipping our status would not undo that — so
    those are blocked here (issue a PayPal refund instead).

    Raises PermissionError for ownership violations.
    Raises ValueError if the payout is not a manually-completed payout.
    """
    res = db.table("royalty_payouts").select("*").eq("id", payout_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payout not found")
    payout = rows[0]
    if payout.get("status") != "paid":
        raise ValueError("Only completed payouts can be reverted to draft")
    if payout.get("payment_method") == "paypal":
        raise ValueError("This payout was paid through PayPal and can't be reverted here")

    # Blocked once this payout's money was re-allocated as credit — reverting
    # would double-free the moved amounts (and a later cancel would cascade
    # them away, violating the payment-record invariant). Every move target
    # (insert, merge, and revision re-point) sets moved_from, and moves only
    # ever touch the SAME payout's rows — so the own-coverage check suffices.
    cov = db.table("royalty_payout_coverage").select("id, moved_from").eq("payout_id", payout_id).execute()
    if any(r.get("moved_from") for r in (cov.data or [])):
        raise ValueError("Credit from this payout has been applied to other periods — revert is unavailable")

    update_res = (
        db.table("royalty_payouts")
        .update({"status": "draft", "paid_at": None})
        .eq("id", payout_id)
        .eq("user_id", user_id)
        .execute()
    )
    return (update_res.data or [{}])[0]


def _derive_orphan_state(db, payout_id: str, snapshot: dict) -> str:
    """Derive orphan_state for a payout by querying remaining coverage.

    Returns:
      "orphaned"  — no coverage rows remain for this payout
      "partial"   — some coverage remains, but snapshot project_ids not all present
      "none"      — all coverage intact
    """
    cov_res = (
        db.table("royalty_payout_coverage")
        .select("royalty_statement_id, project_id")
        .eq("payout_id", payout_id)
        .execute()
    )
    cov_rows = cov_res.data or []

    if not cov_rows:
        return "orphaned"

    # Project ids present in remaining coverage
    cov_project_ids = {r.get("project_id") for r in cov_rows}

    # Project ids referenced in the snapshot
    snapshot_project_ids = {p.get("project_id") for p in snapshot.get("projects", [])}

    if snapshot_project_ids and not snapshot_project_ids.issubset(cov_project_ids):
        return "partial"

    return "none"


def _attach_orphan_state(db, payout: dict) -> dict:
    """Return a copy of *payout* with orphan_state derived from live coverage."""
    snapshot = payout.get("breakdown_snapshot") or {}
    payout_id = payout.get("id")
    orphan = _derive_orphan_state(db, payout_id, snapshot)
    return {**payout, "orphan_state": orphan}


# ---------------------------------------------------------------------------
# Payee mutation functions (patch and split)
# ---------------------------------------------------------------------------


def patch_payee(db, user_id: str, payee_id: str, data: dict) -> dict:
    """Patch mutable fields on a royalty_payees row.

    Allowed keys: payout_currency, registry_user_id, email.
    Providing `email` also sets email_source = 'manual'.
    Raises PermissionError if the payee doesn't belong to user_id.
    """
    # Ownership check
    res = db.table("royalty_payees").select("*").eq("id", payee_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payee not found or not owned by caller")

    update: dict = {}
    for key in ("payout_currency", "registry_user_id", "email"):
        if key in data:
            update[key] = data[key]

    if "email" in data:
        update["email_source"] = "manual"

    update["updated_at"] = datetime.now(UTC).isoformat()

    res = db.table("royalty_payees").update(update).eq("id", payee_id).execute()
    return (res.data or [{}])[0]


def split_payee(db, user_id: str, payee_id: str, line_ids: list[str], new_display_name: str) -> dict:
    """Reassign the selected royalty_lines to a new (or existing) payee.

    Paid-bucket guard: if any selected line's (payee_id, royalty_statement_id) bucket
    is covered by a *paid* payout, the split is rejected with ValueError.

    Returns the target payee row.
    """
    # Ownership check on source payee
    payee_res = db.table("royalty_payees").select("*").eq("id", payee_id).eq("user_id", user_id).execute()
    payee_rows = payee_res.data or []
    if not payee_rows:
        raise PermissionError("Payee not found or not owned by caller")

    # Load the selected lines owned by this user
    lines_res = db.table("royalty_lines").select("*").eq("user_id", user_id).in_("id", line_ids).execute()
    selected_lines = lines_res.data or []
    # Filter to only lines that belong to the caller (belt-and-suspenders)
    selected_lines = [l for l in selected_lines if l.get("user_id") == user_id and l.get("id") in line_ids]

    if not selected_lines:
        # Nothing to split — still upsert target and return it
        target_id = upsert_payee(db, user_id, new_display_name)
        target_res = db.table("royalty_payees").select("*").eq("id", target_id).execute()
        return (target_res.data or [{}])[0]

    # Paid-bucket guard
    # Build set of (payee_id, royalty_statement_id, project_id) buckets from selected lines
    # project_id defaults to "" to match coverage rows that lack a project_id
    buckets_to_check = {
        (l.get("payee_id"), l.get("royalty_statement_id"), l.get("project_id") or "") for l in selected_lines
    }

    # Load all payouts for this user to resolve coverage status
    all_payouts = _load_payouts(db, user_id)
    payout_by_id = {p["id"]: p for p in all_payouts}

    # Load coverage for the source payee
    coverage = _load_coverage_for_payee(db, payee_id)

    for cov in coverage:
        stmt_id = cov.get("royalty_statement_id")
        proj_id = cov.get("project_id") or ""
        payout_id_cov = cov.get("payout_id")
        payout = payout_by_id.get(payout_id_cov, {})
        if payout.get("status") == "paid":
            # Check if any of the selected lines fall into this paid (payee, statement, project) bucket
            if (payee_id, stmt_id, proj_id) in buckets_to_check:
                raise ValueError("cannot split lines already settled by a paid invoice")

    # All clear — upsert the target payee
    target_id = upsert_payee(db, user_id, new_display_name)

    # Unique-identity guard: an EXISTING target payee may already hold a line
    # with the same (statement, project, song, type) — reassigning would violate
    # the royalty_lines identity index and surface as an opaque 500. NULL keys
    # are skipped (the index treats NULLs as distinct, so they cannot collide).
    selected_ids = {l["id"] for l in selected_lines}

    def _identity(l):
        return (l.get("royalty_statement_id"), l.get("project_id"), l.get("song_key"), l.get("royalty_type_key"))

    target_lines = (
        db.table("royalty_lines").select("*").eq("user_id", user_id).eq("payee_id", target_id).execute()
    ).data or []
    taken = {
        _identity(l)
        for l in target_lines
        if l["id"] not in selected_ids and l.get("song_key") and l.get("royalty_type_key")
    }
    if any(_identity(l) in taken for l in selected_lines):
        raise ValueError(
            "That person already has a royalty entry for this song on this statement — remove or merge it first"
        )

    # Lock the reassigned lines so recalculations keep routing this party's
    # money to the split payee instead of re-inserting under the original.
    # Chained splits: an already-locked line KEEPS its original locked_party_key
    # — overwriting it with the intermediate payee's name would break the match
    # against the calculator's party and silently undo the split on re-run.
    source_party_key = payee_rows[0].get("normalized_name") or ""
    fresh_ids = [l["id"] for l in selected_lines if not l.get("locked_party_key")]
    keep_ids = [l["id"] for l in selected_lines if l.get("locked_party_key")]
    if fresh_ids:
        db.table("royalty_lines").update(
            {"payee_id": target_id, "payee_locked": True, "locked_party_key": source_party_key}
        ).in_("id", fresh_ids).execute()
    if keep_ids:
        db.table("royalty_lines").update({"payee_id": target_id, "payee_locked": True}).in_("id", keep_ids).execute()

    # Return the target payee row
    target_res = db.table("royalty_payees").select("*").eq("id", target_id).execute()
    return (target_res.data or [{}])[0]


def delete_project_royalty_entries(db, user_id: str, project_id: str) -> dict:
    """Delete a project's royalty entries: its royalty_calculations, royalty_lines,
    and payout coverage.

    Note: royalty_lines.calculation_id is ON DELETE SET NULL (so the ledger survives
    the deploy cache-wipe), which means lines are NOT swept by the calc deletion —
    we delete them explicitly by project_id. This also clears any post-cache-wipe
    lines whose calculation_id is null. royalty_payouts are intentionally kept; their
    orphan_state is derived at read time.

    Returns {"deleted_calculations": N, "project_id": project_id}.
    """
    # Find the project's calculation ids (scoped to both project and user)
    calc_res = (
        db.table("royalty_calculations").select("id").eq("project_id", project_id).eq("user_id", user_id).execute()
    )
    calc_ids = [r["id"] for r in (calc_res.data or [])]

    # Delete the project's payout coverage — scoped to the user's own payouts.
    # History-record every doomed row first: a manual purge is a destructive,
    # user-triggered action outside the ledger's normal gates, so it must leave
    # an audit trail just like the automated sync paths do.
    user_payouts = _load_payouts(db, user_id)
    payout_ids = [p["id"] for p in user_payouts]
    if payout_ids:
        doomed_cov = (
            db.table("royalty_payout_coverage")
            .select("*")
            .eq("project_id", project_id)
            .in_("payout_id", payout_ids)
            .execute()
        ).data or []
        for cov in doomed_cov:
            history.record(db, user_id, "manual_purge", dict(cov), "manual_purge")
        db.table("royalty_payout_coverage").delete().eq("project_id", project_id).in_("payout_id", payout_ids).execute()

    # Delete the project's royalty lines directly — under ON DELETE SET NULL they
    # outlive their calc, so the calc deletion would not remove them (and this also
    # sweeps null-calc lines left behind by a cache wipe).
    doomed_lines = (
        db.table("royalty_lines").select("*").eq("project_id", project_id).eq("user_id", user_id).execute()
    ).data or []
    for line in doomed_lines:
        history.record(db, user_id, "manual_purge", dict(line), "manual_purge")
    db.table("royalty_lines").delete().eq("project_id", project_id).eq("user_id", user_id).execute()

    # Delete the calculations (the cache rows)
    if calc_ids:
        db.table("royalty_calculations").delete().in_("id", calc_ids).execute()

    return {"deleted_calculations": len(calc_ids), "project_id": project_id}


def list_payouts(db, user_id: str) -> list[dict]:
    """Return all payouts for user_id with derived orphan_state."""
    res = db.table("royalty_payouts").select("*").eq("user_id", user_id).execute()
    payouts = res.data or []
    return [_attach_orphan_state(db, p) for p in payouts]


def get_payout(db, user_id: str, payout_id: str) -> dict:
    """Return a single payout for user_id with derived orphan_state.

    Raises PermissionError (→ 404) if not found or not owned by caller.
    """
    res = db.table("royalty_payouts").select("*").eq("id", payout_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payout not found")
    return _attach_orphan_state(db, rows[0])


# ---------------------------------------------------------------------------
# PayPal checkout for payouts
# ---------------------------------------------------------------------------


class PayoutStateError(ValueError):
    """Payout is in a state that doesn't allow the requested action (→ 409)."""


def _load_owned_payout(db, user_id: str, payout_id: str) -> dict:
    res = db.table("royalty_payouts").select("*").eq("id", payout_id).eq("user_id", user_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payout not found")
    return rows[0]


def _mark_paid_via_paypal(db, user_id: str, payout_id: str, capture_id: str) -> dict:
    update_res = (
        db.table("royalty_payouts")
        .update(
            {
                "status": "paid",
                "paid_at": "now()",
                "payment_method": "paypal",
                "paypal_capture_id": capture_id,
            }
        )
        .eq("id", payout_id)
        .eq("user_id", user_id)
        .execute()
    )
    return (update_res.data or [{}])[0]


def _extract_capture(order: dict) -> dict:
    """Pull the first capture out of an order payload; {} if none exists yet."""
    units = order.get("purchase_units") or []
    captures = ((units[0].get("payments") or {}).get("captures") or []) if units else []
    return captures[0] if captures else {}


def get_paid_payout(db, user_id: str, payout_id: str) -> dict:
    """Return a payout that must be paid (receipts only exist for settled payments).

    Raises PermissionError (→ 404) if not found or not owned by caller,
    PayoutStateError (→ 409) if the payout is not paid yet.
    """
    payout = _load_owned_payout(db, user_id, payout_id)
    if payout.get("status") != "paid":
        raise PayoutStateError("Receipts are only available for paid payouts")
    return payout


def create_paypal_order_for_payout(db, user_id: str, payout_id: str) -> dict:
    """Create a PayPal checkout order for a draft payout.

    The order pays the payee's email directly; the caller approves it with
    their own PayPal account. The payout stays 'draft' until capture succeeds.

    Raises PermissionError (→ 404), PayoutStateError (→ 409),
    ValueError (→ 400, actionable message), PayPalError (→ 502).
    """
    payout = _load_owned_payout(db, user_id, payout_id)
    if payout.get("status") != "draft":
        raise PayoutStateError("Only draft payouts can be paid with PayPal")

    # Reconcile the crash window: a previous capture may have completed on
    # PayPal's side without our status update landing. Never let that payout
    # be paid twice.
    existing_order_id = payout.get("paypal_order_id")
    if existing_order_id:
        try:
            existing = paypal_client.get_order(existing_order_id)
        except paypal_client.PayPalError:
            existing = None  # expired/voided order — safe to create a fresh one
        if existing and existing.get("status") == "COMPLETED":
            capture = _extract_capture(existing)
            _mark_paid_via_paypal(db, user_id, payout_id, capture.get("id"))
            raise PayoutStateError("This payout was already paid via PayPal")

    payee_res = db.table("royalty_payees").select("*").eq("id", payout["payee_id"]).execute()
    payee_rows = payee_res.data or []
    payee = payee_rows[0] if payee_rows else {}
    payee_email = (payee.get("email") or "").strip()
    if not payee_email:
        raise ValueError("This payee has no email address yet. Add one in the Parties tab, then try again.")

    currency = (payout.get("pay_currency") or "").upper()
    if currency not in paypal_client.PAYPAL_SUPPORTED_CURRENCIES:
        raise ValueError(
            f"PayPal doesn't support payments in {currency}. "
            "Change this payee's payout currency in the Parties tab and create a new payout."
        )

    amount_value = paypal_client.format_amount(payout["total_amount"], currency)
    display_name = payee.get("display_name") or "collaborator"
    order = paypal_client.create_order(
        payee_email=payee_email,
        amount_value=amount_value,
        currency=currency,
        reference_id=payout_id,
        description=payout.get("note") or f"Royalty payout — {display_name}",
    )

    # Persist the order id only — payment_method stays 'manual' until capture,
    # so an abandoned order followed by manual Mark-paid never reads as PayPal.
    db.table("royalty_payouts").update({"paypal_order_id": order["id"]}).eq("id", payout_id).eq(
        "user_id", user_id
    ).execute()

    return {"paypal_order_id": order["id"]}


def capture_paypal_order_for_payout(db, user_id: str, payout_id: str) -> dict:
    """Capture the payout's PayPal order and mark the payout paid.

    The order id always comes from the payout row, never from the client, so
    a caller can't capture someone else's order onto their payout. Safe to
    call twice (double-click, retry after crash).

    Raises PermissionError (→ 404), PayoutStateError (→ 409),
    ValueError (→ 400), PayPalError (→ 502 / payment not completed).
    """
    payout = _load_owned_payout(db, user_id, payout_id)

    if payout.get("status") == "paid":
        if payout.get("payment_method") == "paypal":
            return payout  # idempotent: capture already recorded
        raise PayoutStateError("This payout was already marked paid manually")

    order_id = payout.get("paypal_order_id")
    if not order_id:
        raise ValueError("No PayPal payment was started for this payout.")

    try:
        order = paypal_client.capture_order(order_id)
    except paypal_client.PayPalError as exc:
        if exc.issue != "ORDER_ALREADY_CAPTURED":
            raise
        # Captured previously (e.g. double-click) — recover the capture details.
        order = paypal_client.get_order(order_id)

    capture = _extract_capture(order)
    if capture.get("status") != "COMPLETED":
        raise paypal_client.PayPalError(
            "PayPal hasn't completed this payment yet — check your PayPal account and try again.",
            issue=capture.get("status"),
        )

    # Never mark paid on a mismatched amount or currency.
    amount = capture.get("amount") or {}
    expected_currency = (payout.get("pay_currency") or "").upper()
    expected_value = paypal_client.format_amount(payout["total_amount"], expected_currency)
    if amount.get("currency_code") != expected_currency or amount.get("value") != expected_value:
        raise paypal_client.PayPalError(
            f"PayPal capture amount mismatch for payout {payout_id}: "
            f"expected {expected_value} {expected_currency}, got {amount.get('value')} {amount.get('currency_code')}"
        )

    return _mark_paid_via_paypal(db, user_id, payout_id, capture.get("id"))
