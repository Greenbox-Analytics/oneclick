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

from oneclick.royalties import fx
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


def _load_projects(db, user_id: str) -> dict[str, str]:
    """Return {project_id: name} for all projects visible to user_id."""
    res = db.table("projects").select("id, name").eq("user_id", user_id).execute()
    rows = res.data or []
    return {r["id"]: r.get("name", "") for r in rows}


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


def _aggregate_payee_buckets(payee: dict, lines: list[dict], payouts: list[dict], coverage: list[dict], db, base: str):
    """Return aggregated totals (base-currency and payout-currency) for one payee.

    Per-bucket steps:
      1. earned_b  = Σ amount_owed  (statement ccy)
      2. paid_b    = Σ covered_amount where payout.status == 'paid'  (statement ccy)
      3. drafted_b = Σ covered_amount where payout.status == 'draft' (statement ccy)
      4. owed_b    = max(0, earned_b - paid_b - drafted_b)           (statement ccy, CLAMPED)
      5. Convert each of earned_b/paid_b/drafted_b/owed_b → base → accumulate
      6. Convert each of earned_b/paid_b/drafted_b/owed_b → payout_currency → accumulate
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
    earned_r = paid_r = drafted_r = owed_r = 0.0
    # Native (payout_currency) accumulators
    earned_n = paid_n = drafted_n = owed_n = 0.0

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

        # Convert each component → reporting base (on_missing="none" → skip unconvertible buckets)
        earned_conv = fx.convert(db, earned_b, ccy, base, on_missing="none")
        if earned_conv is None:
            unconvertible_count += 1
            continue
        paid_conv = fx.convert(db, paid_b, ccy, base, on_missing="none")
        drafted_conv = fx.convert(db, drafted_b, ccy, base, on_missing="none")
        owed_conv = fx.convert(db, owed_b, ccy, base, on_missing="none")

        earned_r += earned_conv
        paid_r += paid_conv if paid_conv is not None else 0.0
        drafted_r += drafted_conv if drafted_conv is not None else 0.0
        owed_r += owed_conv if owed_conv is not None else 0.0

        # Convert each component → payout_currency (native) — use default "amount" fallback
        earned_n += fx.convert(db, earned_b, ccy, payout_ccy)
        paid_n += fx.convert(db, paid_b, ccy, payout_ccy)
        drafted_n += fx.convert(db, drafted_b, ccy, payout_ccy)
        owed_n += fx.convert(db, owed_b, ccy, payout_ccy)

        if proj_id:
            project_ids.add(proj_id)

    return {
        "earned": earned_r,
        "paid": paid_r,
        "drafted": drafted_r,
        "owed": owed_r,
        "earned_native": earned_n,
        "paid_native": paid_n,
        "drafted_native": drafted_n,
        "owed_native": owed_n,
        "project_ids": project_ids,
        "unconvertible_count": unconvertible_count,
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
        if owed > 0:
            status = "owed"
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
                earned_native=totals["earned_native"],
                paid_native=totals["paid_native"],
                drafted_native=totals["drafted_native"],
                owed_native=totals["owed_native"],
                unconvertible_count=totals["unconvertible_count"],
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
    if owed > 0:
        status = "owed"
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
        earned_native=totals["earned_native"],
        paid_native=totals["paid_native"],
        drafted_native=totals["drafted_native"],
        owed_native=totals["owed_native"],
        unconvertible_count=totals["unconvertible_count"],
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

    # Project names
    proj_res = db.table("projects").select("id, name").execute()
    project_name_map: dict[str, str] = {r["id"]: r.get("name", "") for r in (proj_res.data or [])}

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

            if owed_b > 0:
                st_state = "owed"
            elif drafted_b > 0:
                st_state = "scheduled"
            else:
                st_state = "settled"

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

        for (stmt_id, proj_id), slines in stmt_lines.items():
            ccy = (slines[0].get("statement_currency") or base).upper()
            earned_b = sum(float(sl.get("amount_owed") or 0) for sl in slines)
            paid_b = cov_paid.get((stmt_id, proj_id), 0.0)
            drafted_b = cov_drafted.get((stmt_id, proj_id), 0.0)
            owed_b = max(0.0, earned_b - paid_b - drafted_b)

            if owed_b > 0:
                state = "owed"
            elif drafted_b > 0:
                state = "scheduled"
            else:
                state = "settled"

            earned_base = fx.convert(db, earned_b, ccy, base)
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
            )
        )

    return PeriodLedger(base=base, rows=rows_out).model_dump()


# ---------------------------------------------------------------------------
# Shared owed-bucket helper (used by payee_summary and payout creation)
# ---------------------------------------------------------------------------


def payee_owed_buckets(db, user_id: str, payee_id: str) -> list[dict]:
    """Return the owed buckets for a single payee, filtered to owed_b > 0.

    Each bucket dict contains:
      royalty_statement_id, project_id, calculation_id,
      ccy (statement currency), owed_b (statement-ccy amount, > 0),
      lines (the bucket's royalty_lines rows).

    Coverage is loaded from royalty_payout_coverage for this payee, indexed
    by (statement, project) to compute the per-bucket clamp identical to _aggregate_payee_buckets.
    """
    # Load payee lines for this payee only
    lines_res = db.table("royalty_lines").select("*").eq("user_id", user_id).eq("payee_id", payee_id).execute()
    all_lines = lines_res.data or []
    if not all_lines:
        return []

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

    # Group lines by (statement, project) bucket
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for line in all_lines:
        stmt_id = line.get("royalty_statement_id", "")
        proj_id = line.get("project_id", "")
        buckets[(stmt_id, proj_id)].append(line)

    result = []
    for (stmt_id, proj_id), bucket_lines in buckets.items():
        ccy = (bucket_lines[0].get("statement_currency") or "USD").upper()
        earned_b = sum(float(bl.get("amount_owed") or 0) for bl in bucket_lines)
        paid_b = cov_paid.get((stmt_id, proj_id), 0.0)
        drafted_b = cov_drafted.get((stmt_id, proj_id), 0.0)
        owed_b = max(0.0, earned_b - paid_b - drafted_b)

        if owed_b <= 0:
            continue

        project_id = proj_id
        calc_id = bucket_lines[0].get("calculation_id")

        result.append(
            {
                "royalty_statement_id": stmt_id,
                "project_id": project_id,
                "calculation_id": calc_id,
                "ccy": ccy,
                "owed_b": owed_b,
                "lines": bucket_lines,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Payout creation and management
# ---------------------------------------------------------------------------


def create_payouts(
    db,
    user_id: str,
    payee_ids: list[str],
    idempotency_key: str | None,
    note: str | None,
) -> list[dict]:
    """Create draft payouts for one or more payees.

    For each payee_id:
    - Assert the payee belongs to the caller.
    - Compute owed buckets; skip if empty (no $0 invoices).
    - Idempotency: if derived_key already exists, return the existing payout.
    - Insert royalty_payouts + royalty_payout_coverage rows.

    CRITICAL invariant: coverage.covered_amount is stored in STATEMENT currency (owed_b),
    NOT in the converted payout currency. Only total_amount and snapshot *_pay_ccy fields
    are in payout currency.
    """
    today = date.today().isoformat()
    results = []

    # Load all projects once (for names in snapshot)
    proj_res = db.table("projects").select("id, name").eq("user_id", user_id).execute()
    project_name_map: dict[str, str] = {r["id"]: r.get("name", "") for r in (proj_res.data or [])}

    for payee_id in payee_ids:
        # Ownership check
        payee_res = db.table("royalty_payees").select("*").eq("id", payee_id).eq("user_id", user_id).execute()
        payee_rows = payee_res.data or []
        if not payee_rows:
            raise PermissionError(f"Payee {payee_id} not found or not owned by caller")
        payee = payee_rows[0]

        # Compute owed buckets
        buckets = payee_owed_buckets(db, user_id, payee_id)
        if not buckets:
            continue  # skip — no $0 invoices

        payout_ccy = (payee.get("payout_currency") or "USD").upper()

        # Idempotency: derive a per-payee key and check for existing payout
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
    res = db.table("royalty_payouts").select("*").eq("id", payout_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payout not found")
    payout = rows[0]
    if payout.get("user_id") != user_id:
        raise PermissionError("Not your payout")

    update_res = (
        db.table("royalty_payouts").update({"status": "paid", "paid_at": "now()"}).eq("id", payout_id).execute()
    )
    updated = (update_res.data or [{}])[0]
    return updated


def cancel_payout(db, user_id: str, payout_id: str) -> None:
    """Cancel (delete) a draft payout. Coverage cascades on delete.

    Raises PermissionError for ownership violations.
    Raises ValueError if payout is not in 'draft' status.
    """
    res = db.table("royalty_payouts").select("*").eq("id", payout_id).execute()
    rows = res.data or []
    if not rows:
        raise PermissionError("Payout not found")
    payout = rows[0]
    if payout.get("user_id") != user_id:
        raise PermissionError("Not your payout")
    if payout.get("status") != "draft":
        raise ValueError("only drafts can be canceled")

    db.table("royalty_payouts").delete().eq("id", payout_id).execute()


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
    res = db.table("royalty_payees").select("*").eq("id", payee_id).execute()
    rows = res.data or []
    if not rows or rows[0].get("user_id") != user_id:
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
    payee_res = db.table("royalty_payees").select("*").eq("id", payee_id).execute()
    payee_rows = payee_res.data or []
    if not payee_rows or payee_rows[0].get("user_id") != user_id:
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
    # Build set of (payee_id, royalty_statement_id) buckets from selected lines
    buckets_to_check = {(l.get("payee_id"), l.get("royalty_statement_id")) for l in selected_lines}

    # Load all payouts for this user to resolve coverage status
    all_payouts = _load_payouts(db, user_id)
    payout_by_id = {p["id"]: p for p in all_payouts}

    # Load coverage for the source payee
    coverage = _load_coverage_for_payee(db, payee_id)

    for cov in coverage:
        stmt_id = cov.get("royalty_statement_id")
        payout_id_cov = cov.get("payout_id")
        payout = payout_by_id.get(payout_id_cov, {})
        if payout.get("status") == "paid":
            # Check if any of the selected lines fall into this paid bucket
            if (payee_id, stmt_id) in buckets_to_check:
                raise ValueError("cannot split lines already settled by a paid invoice")

    # All clear — upsert the target payee
    target_id = upsert_payee(db, user_id, new_display_name)

    # Reassign the selected lines
    selected_ids = [l["id"] for l in selected_lines]
    db.table("royalty_lines").update({"payee_id": target_id}).in_("id", selected_ids).execute()

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

    # Delete the project's payout coverage
    db.table("royalty_payout_coverage").delete().eq("project_id", project_id).execute()

    # Delete the project's royalty lines directly — under ON DELETE SET NULL they
    # outlive their calc, so the calc deletion would not remove them (and this also
    # sweeps null-calc lines left behind by a cache wipe).
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
