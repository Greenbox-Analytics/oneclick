"""OneClick results sharing - PDF generation and export to Drive/Slack."""

import io
import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from integrations.oauth import get_valid_token

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


class ShareRequest(BaseModel):
    target: str  # "drive" or "slack"
    artist_name: str
    payments: list[dict]
    total_payments: float
    message: str | None = None  # dashboard result message, shown in the PDF header
    channel_id: str | None = None
    folder_id: str | None = None


# Brand palette mirroring the dashboard's chart colors (hsl greens -> hex),
# assigned by descending payout like the UI; grey slice for unallocated revenue.
_CHART_COLORS = ["#30A66B", "#2C6D4D", "#47D18C", "#40AA95", "#3FAF7F", "#2F7D57", "#7FCBA6", "#33705C"]
_UNALLOCATED_COLOR = "#46534D"
_BRAND_DARK = colors.HexColor("#1E4634")
_ROW_ALT = colors.HexColor("#F4F7F5")
_MUTED = colors.HexColor("#6B7280")


def _payment_gross(p: dict) -> float:
    return float(p.get("gross_amount") or p.get("total_royalty") or 0)


def _payment_net(p: dict) -> float:
    return float(p.get("net_amount") or p.get("gross_amount") or p.get("total_royalty") or 0)


def _generate_pdf(artist_name: str, payments: list[dict], total: float, message: str | None = None) -> io.BytesIO:
    """Generate the OneClick results PDF mirroring the dashboard view: stat
    cards, the payee distribution donut + legend, and the full breakdown table.

    ``total`` is the total amount owed in dollars. Everything else is derived
    from the payment rows the same way the dashboard derives it — no amounts
    are recomputed, only summed.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("RRTitle", parent=styles["Title"], fontSize=19, spaceAfter=4, alignment=0)
    sub_style = ParagraphStyle("RRSub", parent=styles["Normal"], fontSize=9.5, textColor=_MUTED)
    section_style = ParagraphStyle("RRSection", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=8)
    stat_label = ParagraphStyle("RRStatLabel", parent=styles["Normal"], fontSize=8, textColor=_MUTED)
    stat_value = ParagraphStyle("RRStatValue", parent=styles["Normal"], fontSize=16, leading=20, spaceBefore=2)
    stat_value_accent = ParagraphStyle("RRStatValueA", parent=stat_value, textColor=_BRAND_DARK)
    elements = []

    # ---- header ----
    elements.append(Paragraph("Royalty Calculation Results", title_style))
    generated = f"Generated {datetime.now().strftime('%B %d, %Y')} &mdash; {artist_name}"
    elements.append(Paragraph(f"{message} &bull; {generated}" if message else generated, sub_style))
    elements.append(Spacer(1, 0.22 * inch))

    # ---- stat cards (derived exactly like the dashboard) ----
    songs = {p.get("song_title", "") for p in payments}
    payees = {p.get("party_name", "") for p in payments}
    # Total revenue = each song's statement earnings, counted once per song.
    revenue_by_song: dict[str, float] = {}
    for p in payments:
        revenue_by_song.setdefault(p.get("song_title", ""), float(p.get("total_royalty") or 0))
    total_revenue = sum(revenue_by_song.values())

    stat_cells = [
        [
            Paragraph("SONGS PROCESSED", stat_label),
            Paragraph("TOTAL PAYEES", stat_label),
            Paragraph("TOTAL REVENUE", stat_label),
            Paragraph("TOTAL TO PAY", stat_label),
        ],
        [
            Paragraph(str(len(songs)), stat_value),
            Paragraph(str(len(payees)), stat_value),
            Paragraph(f"${total_revenue:,.2f}", stat_value),
            Paragraph(f"${total:,.2f}", stat_value_accent),
        ],
    ]
    stat_table = Table(stat_cells, colWidths=[1.85 * inch] * 4)
    stat_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (0, -1), 0.75, colors.HexColor("#E2E8F0")),
                ("BOX", (1, 0), (1, -1), 0.75, colors.HexColor("#E2E8F0")),
                ("BOX", (2, 0), (2, -1), 0.75, colors.HexColor("#E2E8F0")),
                ("BOX", (3, 0), (3, -1), 0.75, colors.HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ]
        )
    )
    elements.append(stat_table)

    # ---- distribution: donut + legend, mirroring the dashboard ----
    payee_totals: dict[str, float] = {}
    for p in payments:
        payee_totals[p.get("party_name", "")] = payee_totals.get(p.get("party_name", ""), 0) + float(
            p.get("amount_to_pay") or 0
        )
    segments = sorted(payee_totals.items(), key=lambda kv: kv[1], reverse=True)
    seg_colors = {name: _CHART_COLORS[i % len(_CHART_COLORS)] for i, (name, _) in enumerate(segments)}
    allocated = sum(v for _, v in segments)
    unallocated = max(0.0, total_revenue - allocated)
    if unallocated > 0.005:
        segments.append(("Unallocated", unallocated))
        seg_colors["Unallocated"] = _UNALLOCATED_COLOR

    if segments and sum(v for _, v in segments) > 0:
        elements.append(Paragraph("Distribution by payee", section_style))

        drawing = Drawing(2.3 * inch, 2.3 * inch)
        pie = Pie()
        pie.x = 15
        pie.y = 15
        pie.width = 2.0 * inch
        pie.height = 2.0 * inch
        pie.data = [v for _, v in segments]
        pie.labels = None
        pie.simpleLabels = True
        pie.slices.strokeWidth = 0.75
        pie.slices.strokeColor = colors.white
        for i, (name, _) in enumerate(segments):
            pie.slices[i].fillColor = colors.HexColor(seg_colors[name])
        try:  # donut hole (reportlab >= 3.5); harmless plain pie on older versions
            pie.innerRadiusFraction = 0.55
        except (AttributeError, ValueError):
            pass
        drawing.add(pie)

        seg_total = sum(v for _, v in segments)
        legend_cell = ParagraphStyle("RRLegendCell", parent=styles["Normal"], fontSize=8, leading=10)
        legend_rows = [["", "Payee", "Share", "Payout"]]
        for name, value in segments:
            legend_rows.append(
                ["", Paragraph(name, legend_cell), f"{(value / seg_total) * 100:.1f}%", f"${value:,.2f}"]
            )
        legend = Table(legend_rows, colWidths=[0.18 * inch, 2.5 * inch, 0.7 * inch, 0.9 * inch])
        legend_style = [
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), _MUTED),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#E2E8F0")),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        for i, (name, _) in enumerate(segments, start=1):
            legend_style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(seg_colors[name])))
        legend.setStyle(TableStyle(legend_style))

        dist = Table([[drawing, legend]], colWidths=[2.5 * inch, 4.6 * inch])
        dist.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        elements.append(dist)

    # ---- full breakdown table (same columns as the dashboard) ----
    if payments:
        elements.append(Paragraph("Royalty Breakdown", section_style))
        # Text columns use Paragraph cells so long names WRAP inside their
        # column instead of overflowing into the next one.
        cell_style = ParagraphStyle("RRCell", parent=styles["Normal"], fontSize=7.5, leading=9.5)
        table_data = [["Song", "Payee", "Role", "Type", "Basis", "Gross", "Expenses", "Net", "Share", "Amount owed"]]
        for p in payments:
            is_net = p.get("basis") == "net"
            expenses = float(p.get("expenses_applied") or 0)
            table_data.append(
                [
                    Paragraph(str(p.get("song_title", "")), cell_style),
                    Paragraph(str(p.get("party_name", "")), cell_style),
                    Paragraph(str(p.get("role", "")), cell_style),
                    Paragraph(str(p.get("royalty_type", "")), cell_style),
                    "Net" if is_net else "Gross",
                    f"${_payment_gross(p):,.2f}",
                    f"-${expenses:,.2f}" if is_net and expenses > 0 else "—",
                    f"${_payment_net(p):,.2f}",
                    f"{p.get('percentage', 0):g}%",
                    f"${p.get('amount_to_pay', 0):,.2f}",
                ]
            )
        table_data.append(["", "", "", "", "", "", "", "", "Total owed", f"${total:,.2f}"])

        col_w = [0.85, 1.25, 0.72, 0.72, 0.45, 0.62, 0.62, 0.62, 0.45, 0.8]
        t = Table(table_data, repeatRows=1, colWidths=[w * inch for w in col_w])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _BRAND_DARK),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("ALIGN", (5, 0), (-1, -1), "RIGHT"),
                    ("GRID", (0, 0), (-1, -2), 0.5, colors.HexColor("#E2E8F0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, _ROW_ALT]),
                    ("FONTNAME", (8, -1), (-1, -1), "Helvetica-Bold"),
                    ("LINEABOVE", (0, -1), (-1, -1), 0.75, _BRAND_DARK),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        elements.append(t)

    doc.build(elements)
    buffer.seek(0)
    return buffer


class ExportPdfRequest(BaseModel):
    artist_name: str = "Artist"
    payments: list[dict]
    total_payments: float  # total amount owed in dollars (sum of amount_to_pay)
    message: str | None = None  # dashboard result message, shown in the PDF header


@router.post("/export-pdf")
async def export_pdf(body: ExportPdfRequest, user_id: str = Depends(get_current_user_id)):
    """Generate the OneClick royalty-results PDF and return it as a download.

    Reuses the same PDF builder as /share; the data comes from the caller's own
    calculation payload (same trust model as /share), auth required.
    """
    pdf_buffer = _generate_pdf(body.artist_name, body.payments, body.total_payments, message=body.message)
    filename = f"OneClick_Royalties_{datetime.now().strftime('%Y%m%d')}.pdf"
    return Response(
        content=pdf_buffer.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/share")
async def share_results(body: ShareRequest, user_id: str = Depends(get_current_user_id)):
    """Share OneClick results as PDF to Google Drive or Slack."""
    supabase = _get_supabase()
    pdf_buffer = _generate_pdf(body.artist_name, body.payments, body.total_payments, message=body.message)
    filename = f"OneClick_Royalties_{body.artist_name}_{datetime.now().strftime('%Y%m%d')}.pdf"

    if body.target == "drive":
        token = await get_valid_token(supabase, user_id, "google_drive")
        if not token:
            raise HTTPException(status_code=401, detail="Google Drive not connected")

        from integrations.google_drive.service import export_pdf_to_drive

        drive_file = await export_pdf_to_drive(token, pdf_buffer.read(), filename, body.folder_id)
        return {"success": True, "target": "drive", "file": drive_file}

    elif body.target == "slack":
        token = await get_valid_token(supabase, user_id, "slack")
        if not token:
            raise HTTPException(status_code=401, detail="Slack not connected")

        from integrations.slack.service import upload_file_to_channel

        channel_id = body.channel_id
        if not channel_id:
            raise HTTPException(status_code=400, detail="No Slack channel specified")

        result = await upload_file_to_channel(
            token,
            channel_id,
            pdf_buffer.read(),
            filename,
            f"OneClick royalty results for {body.artist_name}",
        )
        return {"success": True, "target": "slack", "result": result}

    raise HTTPException(status_code=400, detail="Invalid target. Use 'drive' or 'slack'.")
