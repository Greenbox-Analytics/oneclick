"""FastAPI router for expense-report exports (PDF + Excel).

One endpoint, ``GET /expenses/export``, builds a downloadable report from the
same cross-project expense summary the Expense Tracker page renders, scoped by
the project/category filters the user has applied on screen.
"""

import re
import sys
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_id
from expenses.common import category_label, filter_expense_rows
from expenses.report_pdf import generate_expense_report_pdf
from expenses.report_xlsx import generate_expense_report_xlsx
from projects.service import get_expenses_summary

router = APIRouter()

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


def _safe_filename_part(text: str | None) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", text or "").strip("_") or "report"


@router.get("/export")
async def export_expenses(
    format: str = Query("pdf"),
    project_id: str | None = Query(None),
    category: str | None = Query(None),
    user_id: str = Depends(get_current_user_id),
):
    """Download the (optionally filtered) expense report as PDF or XLSX.

    ``project_id`` / ``category`` mirror the on-screen filters — omit them for
    an overall report. Rows are scoped to projects the caller is a member of by
    ``get_expenses_summary`` (RLS parity with the Expense Tracker).
    """
    fmt = (format or "pdf").lower()
    if fmt not in ("pdf", "xlsx"):
        raise HTTPException(status_code=400, detail="format must be 'pdf' or 'xlsx'")

    rows = await get_expenses_summary(_get_supabase(), user_id)
    rows = filter_expense_rows(rows, project_id, category)

    if project_id:
        scope_label = next((r.get("project_name") for r in rows if r.get("project_name")), None) or "Selected project"
    else:
        scope_label = "All projects"
    cat_label = category_label(category) if category else "All categories"
    generated_on = date.today().isoformat()

    try:
        if fmt == "xlsx":
            buffer = generate_expense_report_xlsx(rows, scope_label, cat_label, generated_on)
            media_type = XLSX_MEDIA_TYPE
        else:
            buffer = generate_expense_report_pdf(rows, scope_label, cat_label, generated_on)
            media_type = "application/pdf"
    except Exception as e:  # noqa: BLE001 — surface any generation failure as a 500
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

    analytics_capture(
        user_id,
        "expense_report_exported",
        {
            "tool": "expense_tracker",
            "format": fmt,
            "scope": "project" if project_id else "all",
            "category_filtered": bool(category),
            "expense_count": len(rows),
        },
    )

    filename = f"Expense_Report_{_safe_filename_part(scope_label)}.{fmt}"
    return StreamingResponse(
        buffer,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
