"""OneClick Earnings Breakdown — per-calculation aggregations across the
dimensional fields preserved on `royalty_statement_rows` (vendor, country,
month, delivery format).

Read-only endpoints; mounted under `/oneclick` alongside the share router.
"""

import sys
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id  # noqa: E402

router = APIRouter()

DIMENSIONS = {
    "country": "country",
    "vendor": "vendor",
    "format": "delivery_format",
    "month": "sale_date",
}


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


def _bucket_key(dimension: str, raw_value):
    """Map a raw row value to its bucket key for the requested dimension.

    Month is bucketed YYYY-MM. Unknown / null values land in an explicit
    `"Unknown"` bucket so the UI surfaces them rather than dropping rows silently.
    """
    if raw_value is None or raw_value == "":
        return "Unknown"
    if dimension == "month":
        # sale_date arrives as ISO YYYY-MM-DD; first 7 chars is YYYY-MM.
        s = str(raw_value)
        return s[:7] if len(s) >= 7 else "Unknown"
    return str(raw_value)


@router.get("/calculations/{calculation_id}/breakdown")
async def get_earnings_breakdown(
    calculation_id: str,
    dimension: str = Query(..., description="One of: country, month, format, vendor"),
    user_id: str = Depends(get_current_user_id),
):
    """Aggregate net_payable across statement rows for a calculation by dimension.

    Response shape:
        {
            "dimension": "country",
            "total": 1234.56,
            "row_count": 4200,
            "rows": [
                {"key": "United States", "net_payable": 812.10, "row_count": 1800, "percent_of_total": 65.8},
                ...
            ]
        }
    """
    if dimension not in DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension '{dimension}'. Must be one of: {', '.join(DIMENSIONS)}",
        )

    supabase = _get_supabase()

    # Owner check — RLS would block cross-user reads, but failing fast with a
    # clear 404 is friendlier than the empty-array surprise the policy returns.
    calc_res = (
        supabase.table("royalty_calculations")
        .select("id, user_id")
        .eq("id", calculation_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not calc_res.data:
        raise HTTPException(status_code=404, detail="Calculation not found")

    column = DIMENSIONS[dimension]
    rows_res = (
        supabase.table("royalty_statement_rows")
        .select(f"{column}, net_payable")
        .eq("calculation_id", calculation_id)
        .execute()
    )
    raw_rows = rows_res.data or []

    if not raw_rows:
        return {
            "dimension": dimension,
            "total": 0.0,
            "row_count": 0,
            "rows": [],
        }

    bucket_total: dict[str, float] = defaultdict(float)
    bucket_count: dict[str, int] = defaultdict(int)
    for row in raw_rows:
        amount = row.get("net_payable")
        if amount is None:
            continue
        key = _bucket_key(dimension, row.get(column))
        bucket_total[key] += float(amount)
        bucket_count[key] += 1

    grand_total = sum(bucket_total.values())

    # Month dimension sorts chronologically; others sort by amount descending so
    # the visual story is "where the money comes from."
    if dimension == "month":
        sorted_keys = sorted(bucket_total.keys())
    else:
        sorted_keys = sorted(bucket_total.keys(), key=lambda k: bucket_total[k], reverse=True)

    rows = [
        {
            "key": key,
            "net_payable": round(bucket_total[key], 4),
            "row_count": bucket_count[key],
            "percent_of_total": (round(bucket_total[key] / grand_total * 100, 2) if grand_total else 0.0),
        }
        for key in sorted_keys
    ]

    return {
        "dimension": dimension,
        "total": round(grand_total, 4),
        "row_count": sum(bucket_count.values()),
        "rows": rows,
    }
