"""OneClick results sharing - PDF generation and export to Drive/Slack."""

import io
import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
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
    channel_id: str | None = None
    folder_id: str | None = None


def _generate_pdf(artist_name: str, payments: list[dict], total: float) -> io.BytesIO:
    """Generate a PDF of OneClick royalty results."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=18, spaceAfter=12)
    elements = []

    elements.append(Paragraph(f"OneClick Royalty Report &mdash; {artist_name}", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Total Payments: ${total:,.2f}", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    if payments:
        table_data = [["Song", "Party", "Role", "Type", "%", "Amount"]]
        for p in payments:
            table_data.append(
                [
                    str(p.get("song_title", "")),
                    str(p.get("party_name", "")),
                    str(p.get("role", "")),
                    str(p.get("royalty_type", "")),
                    f"{p.get('percentage', 0):.1f}%",
                    f"${p.get('amount_to_pay', 0):,.2f}",
                ]
            )

        t = Table(table_data, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A154B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F9FAFB")],
                    ),
                ]
            )
        )
        elements.append(t)

    doc.build(elements)
    buffer.seek(0)
    return buffer


@router.post("/share")
async def share_results(body: ShareRequest, user_id: str = Depends(get_current_user_id)):
    """Share OneClick results as PDF to Google Drive or Slack."""
    supabase = _get_supabase()
    pdf_buffer = _generate_pdf(body.artist_name, body.payments, body.total_payments)
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
