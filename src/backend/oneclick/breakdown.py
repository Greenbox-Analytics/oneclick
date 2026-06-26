"""OneClick Earnings Breakdown — per-calculation aggregations across the
dimensional fields preserved on `royalty_statement_rows` (vendor, country,
month, delivery format).

Read-only endpoints; mounted under `/oneclick` alongside the share router.
"""

import sys
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

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
    "year": "sale_date",
}

# Time dimensions sort chronologically; everything else by amount descending.
_TIME_DIMENSIONS = {"month", "year"}

# Section title shown in the exported PDF for each dimension.
_PDF_SECTION_TITLES = {
    "month": "Earnings by Time Period",
    "year": "Earnings by Time Period",
    "vendor": "Earnings by Vendor",
    "country": "Earnings by Country",
    "format": "Earnings by Source",
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
    if dimension == "year":
        # First 4 chars is YYYY.
        s = str(raw_value)
        return s[:4] if len(s) >= 4 else "Unknown"
    return str(raw_value)


def _verify_calc_owner(supabase, calculation_id: str, user_id: str) -> None:
    """Raise 404 unless `calculation_id` belongs to `user_id`.

    RLS would block cross-user reads, but failing fast with a clear 404 is
    friendlier than the empty-array surprise the policy returns.
    """
    calc_res = (
        supabase.table("royalty_calculations")
        .select("id, user_id")
        .eq("id", calculation_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not calc_res.data:
        raise HTTPException(status_code=404, detail="Calculation not found")


def _calculation_track_label(supabase, calculation_id: str) -> str | None:
    """Build a human label of the contract track(s) for a calculation, e.g.
    "Like That" or "Like That, Home +2 more", or None if unavailable.

    Sourced from the saved calculation's payments — each payment's `song_title`
    is the contract work title, so this names the track(s) the contract covers.
    """
    res = supabase.table("royalty_calculations").select("results").eq("id", calculation_id).execute()
    results = (res.data or [{}])[0].get("results") or {}
    payments = results.get("payments", []) or []

    titles: list[str] = []
    seen: set[str] = set()
    for p in payments:
        title = (p.get("song_title") or "").strip()
        if title and title.lower() not in seen:
            seen.add(title.lower())
            titles.append(title)

    if not titles:
        return None
    if len(titles) <= 3:
        return ", ".join(titles)
    return f"{', '.join(titles[:3])} +{len(titles) - 3} more"


def _aggregate_breakdown(supabase, calculation_id: str, dimension: str) -> dict:
    """Aggregate net_payable across a calculation's statement rows by dimension.

    Returns the response shape documented on `get_earnings_breakdown`. Shared by
    the JSON endpoint and the PDF export.
    """
    column = DIMENSIONS[dimension]
    rows_res = (
        supabase.table("royalty_statement_rows")
        .select(f"{column}, net_payable")
        .eq("calculation_id", calculation_id)
        .execute()
    )
    raw_rows = rows_res.data or []

    if not raw_rows:
        return {"dimension": dimension, "total": 0.0, "row_count": 0, "rows": []}

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

    # Time dimensions sort chronologically; others by amount descending so the
    # visual story is "where the money comes from."
    if dimension in _TIME_DIMENSIONS:
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
    _verify_calc_owner(supabase, calculation_id, user_id)
    return _aggregate_breakdown(supabase, calculation_id, dimension)


@router.get("/calculations/{calculation_id}/breakdown/pdf")
async def export_breakdown_pdf(
    calculation_id: str,
    time_grain: str = Query("month", description="Time-period grain for the PDF: month or year"),
    user_id: str = Depends(get_current_user_id),
):
    """Export the full earnings breakdown (time period, vendor, country, source)
    as a single multi-section PDF.
    """
    if time_grain not in _TIME_DIMENSIONS:
        time_grain = "month"

    supabase = _get_supabase()
    _verify_calc_owner(supabase, calculation_id, user_id)

    # Section order: time period first, then the where-it-came-from dimensions.
    section_dimensions = [time_grain, "vendor", "country", "format"]
    sections = [(dim, _aggregate_breakdown(supabase, calculation_id, dim)) for dim in section_dimensions]

    track_label = _calculation_track_label(supabase, calculation_id)
    pdf_buffer = _generate_breakdown_pdf(sections, track_label)
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="earnings-breakdown.pdf"'},
    )


def _generate_breakdown_pdf(sections: list[tuple[str, dict]], track_label: str | None = None):
    """Build a multi-section earnings-breakdown PDF and return a BytesIO buffer.

    Each section is a (dimension, aggregation) pair. Reuses the table styling
    established by the OneClick share PDF for visual consistency. When
    `track_label` is given, the document title reads "<Track> — Earnings Breakdown".
    """
    import io
    from datetime import datetime
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("BreakdownTitle", parent=styles["Title"], fontSize=18, spaceAfter=12)

    # Escape the dynamic track name — it can contain &, <, > that would otherwise
    # break the reportlab Paragraph markup.
    prefix = escape(track_label) if track_label else "OneClick"
    elements = [
        Paragraph(f"{prefix} &mdash; Earnings Breakdown", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles["Normal"]),
        Spacer(1, 0.3 * inch),
    ]

    for dimension, agg in sections:
        section_title = _PDF_SECTION_TITLES.get(dimension, dimension.title())
        elements.append(Paragraph(f"{section_title} &mdash; ${agg['total']:,.2f}", styles["Heading2"]))
        elements.append(Spacer(1, 0.1 * inch))

        if not agg["rows"]:
            elements.append(Paragraph("No data for this dimension.", styles["Normal"]))
            elements.append(Spacer(1, 0.25 * inch))
            continue

        table_data = [["Category", "Net Payable", "% of total"]]
        for r in agg["rows"]:
            table_data.append(
                [
                    str(r["key"]),
                    f"${r['net_payable']:,.2f}",
                    f"{r['percent_of_total']:.2f}%",
                ]
            )

        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A154B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer
