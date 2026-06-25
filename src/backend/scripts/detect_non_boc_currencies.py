"""Detect royalty ledger data in a currency the Bank of Canada can't convert.

Read-only GATE to run BEFORE removing the fawazahmed0 FX fallback (BoC-only FX).
BoC publishes FX{CODE}CAD series for a fixed set of currencies; anything else
(e.g. NGN, AED) has no BoC rate and would otherwise be folded into a base-currency
sum at 1:1 — a large overstatement. This scans for any such data so the pivot is a
verified decision, not a silent one.

Scans:
  - royalty_lines.statement_currency
  - royalty_payouts.pay_currency
  - royalty_payees.payout_currency
  - orphan coverage: royalty_payout_coverage rows whose statement has no surviving
    royalty_lines (so their currency can't be inferred — flagged for manual review)

Usage (read-only, no flags):
    poetry run python scripts/detect_non_boc_currencies.py

Requires VITE_SUPABASE_URL + VITE_SUPABASE_SECRET_KEY in env.
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Currencies the Bank of Canada publishes (FX{CODE}CAD series), plus CAD itself.
BOC_SUPPORTED = {
    "AUD",
    "BRL",
    "CHF",
    "CNY",
    "EUR",
    "GBP",
    "HKD",
    "IDR",
    "INR",
    "JPY",
    "KRW",
    "MXN",
    "MYR",
    "NOK",
    "NZD",
    "PEN",
    "PLN",
    "RUB",
    "SAR",
    "SEK",
    "SGD",
    "THB",
    "TRY",
    "TWD",
    "USD",
    "VND",
    "ZAR",
    "CAD",
}


def _get_supabase():
    """Lazy supabase client. Patched in tests."""
    from supabase import create_client

    url = os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_SECRET_KEY")
    if not url or not key:
        print("ERROR: VITE_SUPABASE_URL / VITE_SUPABASE_SECRET_KEY not set", file=sys.stderr)
        raise SystemExit(2)
    return create_client(url, key)


def _distinct(db, table: str, col: str) -> set[str]:
    res = db.table(table).select(col).execute()
    return {(row.get(col) or "").upper() for row in (res.data or []) if row.get(col)}


def scan(db) -> dict:
    """Return {'non_boc': {col: [codes]}, 'orphan_coverage': [statement_ids]}."""
    sources = {
        "royalty_lines.statement_currency": ("royalty_lines", "statement_currency"),
        "royalty_payouts.pay_currency": ("royalty_payouts", "pay_currency"),
        "royalty_payees.payout_currency": ("royalty_payees", "payout_currency"),
    }
    non_boc: dict[str, list[str]] = {}
    for label, (table, col) in sources.items():
        offenders = sorted(c for c in _distinct(db, table, col) if c not in BOC_SUPPORTED)
        if offenders:
            non_boc[label] = offenders

    # Orphan coverage: coverage statements with no surviving lines (currency unknowable).
    cov = db.table("royalty_payout_coverage").select("royalty_statement_id").execute()
    cov_stmts = {r["royalty_statement_id"] for r in (cov.data or []) if r.get("royalty_statement_id")}
    orphan = []
    for sid in cov_stmts:
        lines = db.table("royalty_lines").select("id").eq("royalty_statement_id", sid).limit(1).execute()
        if not (lines.data or []):
            orphan.append(sid)

    return {"non_boc": non_boc, "orphan_coverage": sorted(orphan)}


def main(argv: list[str] | None = None) -> int:
    db = _get_supabase()
    result = scan(db)
    non_boc, orphan = result["non_boc"], result["orphan_coverage"]

    if not non_boc and not orphan:
        print("0 non-BoC rows and 0 orphan-coverage statements — safe to proceed (BoC-only FX).")
        return 0

    print("NON-BoC CURRENCY DATA FOUND — do NOT remove the FX fallback until resolved:")
    for label, codes in non_boc.items():
        print(f"  {label}: {', '.join(codes)}")
    if orphan:
        print(f"  orphan coverage (statement has no lines; currency unknowable): {len(orphan)} statement(s)")
        for sid in orphan:
            print(f"    - {sid}")
    print("\nThese rows have no BoC rate. Resolve before P1.1 (keep a conversion path, or exclude them).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
