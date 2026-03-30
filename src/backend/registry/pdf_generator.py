"""Proof-of-ownership PDF with approval status per stakeholder."""

import io
import hashlib
from datetime import datetime, timezone

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

BRAND = colors.HexColor("#1a3a2a")
GREEN = colors.HexColor("#16a34a")
RED = colors.HexColor("#dc2626")
AMBER = colors.HexColor("#d97706")


def generate_proof_of_ownership_pdf(work_data: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    elements = []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=24, spaceAfter=4, textColor=BRAND, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#555"), alignment=TA_CENTER, spaceAfter=20)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=14, textColor=BRAND, spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888"), leading=11)

    collaborators = work_data.get("collaborators", [])
    stake_approval = {}
    email_approval = {}
    for c in collaborators:
        if c.get("stake_id"):
            stake_approval[c["stake_id"]] = {"status": c["status"], "name": c["name"]}
        if c.get("email"):
            email_approval[c["email"].lower()] = {"status": c["status"], "name": c["name"]}

    # Header
    elements.append(Paragraph("PROOF OF OWNERSHIP", title_style))
    elements.append(Paragraph("Rights & Ownership Registry Certificate", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=BRAND, spaceAfter=20))

    # Status banner
    work_status = (work_data.get("status") or "draft").replace("_", " ").title()
    status_color = {"Registered": GREEN, "Disputed": RED, "Pending Approval": AMBER}.get(work_status, colors.HexColor("#666"))
    elements.append(Paragraph(f"<b>Registry Status: <font color='{status_color}'>{work_status}</font></b>", body_style))
    elements.append(Spacer(1, 8))

    # Work details
    elements.append(Paragraph("Work Details", section_style))
    details = [
        ["Title:", work_data.get("title", "—")],
        ["Type:", (work_data.get("work_type") or "single").replace("_", " ").title()],
        ["ISRC:", work_data.get("isrc") or "—"],
        ["ISWC:", work_data.get("iswc") or "—"],
        ["UPC:", work_data.get("upc") or "—"],
        ["Release Date:", str(work_data.get("release_date") or "—")],
    ]
    dt = Table(details, colWidths=[1.5 * inch, 5 * inch])
    dt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(dt)
    elements.append(Spacer(1, 12))

    # Ownership with approval
    stakes = work_data.get("stakes", [])

    def build_stakes_section(label, stake_list):
        elements.append(Paragraph(f"{label} Ownership", section_style))
        if not stake_list:
            elements.append(Paragraph(f"No {label.lower()} ownership recorded.", body_style))
            elements.append(Spacer(1, 8))
            return

        header = ["Holder", "Role", "%", "Publisher/Label", "Approval"]
        rows = [header]
        for s in stake_list:
            approval = "—"
            sid = s.get("id")
            hemail = (s.get("holder_email") or "").lower()
            if sid in stake_approval:
                approval = stake_approval[sid]["status"].title()
            elif hemail and hemail in email_approval:
                approval = email_approval[hemail]["status"].title()
            rows.append([
                s.get("holder_name", ""), s.get("holder_role", ""),
                f"{s.get('percentage', 0):.2f}%",
                s.get("publisher_or_label") or "—", approval,
            ])
        total = sum(s.get("percentage", 0) for s in stake_list)
        rows.append(["", "TOTAL", f"{total:.2f}%", "", ""])

        col_widths = [1.5 * inch, 1.0 * inch, 0.8 * inch, 1.5 * inch, 1.2 * inch]
        tbl = Table(rows, colWidths=col_widths)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), BRAND),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ("ALIGN", (4, 0), (4, -1), "CENTER"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 1, colors.black),
            ("GRID", (0, 0), (-1, -2), 0.5, colors.HexColor("#ccc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#999")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]
        for row_idx in range(1, len(rows) - 1):
            approval_text = rows[row_idx][4]
            if approval_text == "Confirmed":
                style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), GREEN))
            elif approval_text == "Disputed":
                style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), RED))
            elif approval_text == "Invited":
                style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), AMBER))
        tbl.setStyle(TableStyle(style_cmds))
        elements.append(tbl)
        elements.append(Spacer(1, 12))

    master = [s for s in stakes if s.get("stake_type") == "master"]
    pub = [s for s in stakes if s.get("stake_type") == "publishing"]
    build_stakes_section("Master", master)
    build_stakes_section("Publishing", pub)

    # Licensing
    licenses = work_data.get("licenses", [])
    elements.append(Paragraph("Licensing Rights", section_style))
    if not licenses:
        elements.append(Paragraph("No licensing rights recorded.", body_style))
    else:
        rows = [["Type", "Licensee", "Territory", "Start", "End", "Status"]]
        for lic in licenses:
            rows.append([
                (lic.get("license_type") or "").replace("_", " ").title(),
                lic.get("licensee_name", ""), lic.get("territory", ""),
                str(lic.get("start_date", "—")), str(lic.get("end_date") or "Perpetual"),
                (lic.get("status") or "active").title(),
            ])
        tbl = Table(rows, colWidths=[1.0*inch, 1.4*inch, 1.0*inch, 0.9*inch, 0.9*inch, 0.8*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(tbl)
    elements.append(Spacer(1, 12))

    # Agreements
    agreements = work_data.get("agreements", [])
    elements.append(Paragraph("Agreement History", section_style))
    if not agreements:
        elements.append(Paragraph("No agreements recorded.", body_style))
    else:
        for agr in agreements:
            agr_type = (agr.get("agreement_type") or "").replace("_", " ").title()
            parties_list = agr.get("parties") or []
            party_names = ", ".join(p.get("name", "") for p in parties_list) if parties_list else "—"
            elements.append(Paragraph(f"<b>{agr.get('title', '')}</b> — {agr_type}", body_style))
            elements.append(Paragraph(f"Effective: {agr.get('effective_date', '—')} | Recorded: {agr.get('created_at', '—')}", small_style))
            elements.append(Paragraph(f"Parties: {party_names}", small_style))
            if agr.get("document_hash"):
                elements.append(Paragraph(f"Hash: {agr['document_hash']}", small_style))
            elements.append(Spacer(1, 6))

    # Footer
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccc"), spaceAfter=8))
    content_str = f"{work_data.get('id', '')}|{work_data.get('title', '')}|{generated_at}"
    doc_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

    elements.append(Paragraph(
        "This certificate reflects the ownership and rights information as recorded in the "
        "Msanii Rights & Ownership Registry. Stakeholder approval status indicates whether "
        "each party has confirmed their stake. A 'Registered' status means all parties have agreed.",
        ParagraphStyle("Disc", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888"), leading=11),
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        f"Generated: {generated_at} | Document ID: {doc_hash}",
        ParagraphStyle("Foot", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#aaa"), alignment=TA_CENTER),
    ))
    elements.append(Paragraph(
        "Msanii Rights & Ownership Registry",
        ParagraphStyle("Brand", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#aaa"), alignment=TA_CENTER),
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
