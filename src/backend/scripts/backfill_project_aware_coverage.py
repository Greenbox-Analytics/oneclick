"""Backfill royalty_payout_coverage rows to be project-aware.

BEST-EFFORT NOTICE — READ BEFORE APPLYING
==========================================
Historically, ``royalty_payout_coverage`` wrote ONE row per (payout, statement)
using the first line's project_id and the full bucket covered_amount.  If any
statement spanned multiple projects, that single row mis-attributes per-project
coverage.

This script re-derives those rows by splitting the original ``covered_amount``
*proportionally* to each project's earned amount (``Σ royalty_lines.amount_owed``
per project for that (payee, statement) pair).  The split is an **approximate
retroactive heuristic** — the true per-project split is unknowable if prior
partial coverage existed (because the original partial payment could have been
preferentially applied to one project).

**What is preserved exactly:** the per-(payout, statement) ``covered_amount``
total is unchanged, so no ``paid_total`` or headline payout figure is affected.

**What is approximate:** per-project attribution.

**YOU MUST review the --dry-run output before applying ``--yes``.**

Usage::

    poetry run python scripts/backfill_project_aware_coverage.py --dry-run
    poetry run python scripts/backfill_project_aware_coverage.py --yes

Requires VITE_SUPABASE_URL + VITE_SUPABASE_SECRET_KEY in env (same as
``clear_oneclick_cache.py``).
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env at import time so that tests can monkeypatch os.environ AFTER
# import and reliably override (or delete) individual variables without
# load_dotenv() re-populating them inside _get_supabase().
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # In environments where python-dotenv is absent, rely on real env.


def _get_supabase():
    """Lazy supabase client.  Patched in tests."""
    from supabase import create_client

    url = os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_SECRET_KEY")
    if not url or not key:
        print(
            "ERROR: VITE_SUPABASE_URL / VITE_SUPABASE_SECRET_KEY not set",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Core backfill logic (pure-ish, accepts a supabase client for testability)
# ---------------------------------------------------------------------------


def _load_all_payouts(sb):
    """Return every payout row (no user filter — service-role client)."""
    res = sb.table("royalty_payouts").select("id,payee_id").execute()
    return res.data or []


def _load_coverage_for_payout(sb, payout_id: str):
    """Return all coverage rows for a payout."""
    res = (
        sb.table("royalty_payout_coverage")
        .select("payout_id,payee_id,royalty_statement_id,covered_amount,project_id")
        .eq("payout_id", payout_id)
        .execute()
    )
    return res.data or []


def _load_lines_for_payee_statement(sb, payee_id: str, statement_id: str):
    """Return royalty_lines for a (payee, statement) pair."""
    res = (
        sb.table("royalty_lines")
        .select("project_id,amount_owed")
        .eq("payee_id", payee_id)
        .eq("royalty_statement_id", statement_id)
        .execute()
    )
    return res.data or []


def _proportional_split(covered_amount: float, project_earned: dict[str, float]) -> dict[str, float]:
    """Split ``covered_amount`` proportionally to ``project_earned`` totals.

    If total earned is zero (edge case: all lines have amount_owed=0), falls
    back to an equal split.  The sum of returned values equals ``covered_amount``
    exactly (last project absorbs floating-point rounding residue).
    """
    projects = list(project_earned.keys())
    total_earned = sum(project_earned.values())

    if not projects:
        return {}

    if total_earned == 0.0:
        # Equal split fallback
        per = covered_amount / len(projects)
        return {p: per for p in projects}

    result = {}
    allocated = 0.0
    for i, proj in enumerate(projects):
        if i == len(projects) - 1:
            # Last project absorbs any floating-point residue
            result[proj] = round(covered_amount - allocated, 10)
        else:
            share = round((project_earned[proj] / total_earned) * covered_amount, 10)
            result[proj] = share
            allocated += share
    return result


def find_affected_payouts(sb):
    """Identify coverage rows that need re-deriving.

    Returns a list of dicts::

        {
            "payout_id": str,
            "payee_id": str,
            "statement_id": str,
            "original_row": dict,          # the single existing coverage row
            "proposed_rows": list[dict],   # replacement rows (one per project)
        }

    Only includes entries where the statement's lines span >1 distinct project_id
    and coverage currently has exactly one row for that (payout, statement).
    """
    affected = []
    payouts = _load_all_payouts(sb)

    for payout in payouts:
        payout_id = payout["id"]
        payee_id = payout["payee_id"]

        coverage_rows = _load_coverage_for_payout(sb, payout_id)

        # Group existing coverage by statement_id
        by_stmt: dict[str, list[dict]] = defaultdict(list)
        for row in coverage_rows:
            by_stmt[row["royalty_statement_id"]].append(row)

        for stmt_id, rows in by_stmt.items():
            # Only process statements that currently have exactly ONE coverage row
            if len(rows) != 1:
                continue

            original_row = rows[0]
            covered_amount = original_row["covered_amount"]

            lines = _load_lines_for_payee_statement(sb, payee_id, stmt_id)

            # Aggregate earned per project
            project_earned: dict[str, float] = defaultdict(float)
            for line in lines:
                pid = line["project_id"]
                if pid:
                    project_earned[pid] += line["amount_owed"]

            # Single-project statements don't need re-deriving
            if len(project_earned) <= 1:
                continue

            split = _proportional_split(covered_amount, dict(project_earned))

            proposed_rows = [
                {
                    "payout_id": payout_id,
                    "payee_id": payee_id,
                    "royalty_statement_id": stmt_id,
                    "project_id": proj_id,
                    "covered_amount": amount,
                }
                for proj_id, amount in split.items()
            ]

            affected.append(
                {
                    "payout_id": payout_id,
                    "payee_id": payee_id,
                    "statement_id": stmt_id,
                    "original_row": original_row,
                    "proposed_rows": proposed_rows,
                }
            )

    return affected


def apply_backfill(sb, affected: list[dict]) -> int:
    """Delete the old single-row and insert per-project rows for each affected entry.

    Returns count of (payout, statement) pairs rewritten.
    """
    count = 0
    for entry in affected:
        payout_id = entry["payout_id"]
        stmt_id = entry["statement_id"]

        # Delete the stale single-project coverage row
        sb.table("royalty_payout_coverage").delete().eq("payout_id", payout_id).eq(
            "royalty_statement_id", stmt_id
        ).execute()

        # Insert one row per project
        sb.table("royalty_payout_coverage").insert(entry["proposed_rows"]).execute()

        count += 1
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill royalty_payout_coverage to be project-aware (one row per project per statement)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print affected payouts and proposed changes without modifying the database (default-safe mode).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply the backfill.  Without this flag the script is read-only.",
    )
    args = parser.parse_args(argv)

    sb = _get_supabase()

    print("Scanning royalty_payout_coverage for mixed-project statements…")
    affected = find_affected_payouts(sb)

    if not affected:
        print("0 mixed-project payouts — nothing to backfill.")
        return 0

    for entry in affected:
        print(f"\n  payout={entry['payout_id']}  statement={entry['statement_id']}  payee={entry['payee_id']}")
        print(
            f"    CURRENT : project={entry['original_row']['project_id']}"
            f"  covered_amount={entry['original_row']['covered_amount']}"
        )
        print("    PROPOSED:")
        for row in entry["proposed_rows"]:
            print(f"      project={row['project_id']}  covered_amount={row['covered_amount']}")

    if args.dry_run or not args.yes:
        print(
            f"\nDRY RUN: {len(affected)} (payout, statement) pair(s) would be rewritten."
            "\nRe-run with --yes to apply.  Review the output above first."
        )
        return 0

    count = apply_backfill(sb, affected)
    print(f"\nBackfill complete: {count} (payout, statement) pair(s) rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
