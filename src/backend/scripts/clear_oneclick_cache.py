"""Truncate OneClick cache tables (royalty_calculations + junction).

Used by CI/CD after a deploy whose diff touched src/backend/oneclick/**, so
users don't see stale results from the previous calculation logic. Also
runnable locally:

    poetry run python scripts/clear_oneclick_cache.py --dry-run
    poetry run python scripts/clear_oneclick_cache.py --yes

Requires VITE_SUPABASE_URL + VITE_SUPABASE_SECRET_KEY in env (same as
grant_pro.py).
"""

import argparse
import os
import sys
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
    """Lazy supabase client. Patched in tests."""
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


def _count(supabase, table: str) -> int:
    res = supabase.table(table).select("*", count="exact").execute()
    return getattr(res, "count", 0) or 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wipe OneClick cache tables")
    parser.add_argument("--dry-run", action="store_true", help="Print counts, do not delete")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args(argv)

    sb = _get_supabase()

    junction_before = _count(sb, "royalty_calculation_contracts")
    calc_before = _count(sb, "royalty_calculations")

    if args.dry_run:
        print(f"DRY RUN: would delete {junction_before} junction rows and {calc_before} royalty_calculations rows")
        return 0

    if not args.yes:
        confirm = input(f"Delete {calc_before} calc rows + {junction_before} junction rows? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return 1

    # Junction has ON DELETE CASCADE from parent, so deleting parent rows sweeps
    # the junction automatically. We still issue the junction delete first as a
    # defensive belt-and-suspenders move.
    sb.table("royalty_calculation_contracts").delete().neq(
        "calculation_id", "00000000-0000-0000-0000-000000000000"
    ).execute()
    sb.table("royalty_calculations").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    print(f"Deleted {calc_before} royalty_calculations rows, {junction_before} junction rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
