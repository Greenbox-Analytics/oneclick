import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def generate_split_sheet_pdf(
    work_title: str,
    work_type: str,
    split_type: str,
    date: str,
    contributors: list[dict],
) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    elements = []

    brand = colors.HexColor("#1a3a2a")

    title_style = ParagraphStyle(
        "SplitSheetTitle",
        parent=styles["Title"],
        fontSize=28,
        spaceAfter=4,
        textColor=brand,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SplitSheetSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=brand,
        spaceBefore=16,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        leading=11,
        spaceBefore=20,
    )

    needs_pub = split_type in ("publishing", "both")
    needs_master = split_type in ("master", "both")

    work_type_display = work_type.upper() if work_type else "SINGLE"

    # --- Header ---
    elements.append(Paragraph("SPLIT SHEET AGREEMENT", title_style))
    elements.append(Paragraph("Music Work Royalty Split Agreement", subtitle_style))
    elements.append(
        HRFlowable(width="100%", thickness=2, color=brand, spaceAfter=20)
    )

    # --- Section 1: Parties ---
    elements.append(Paragraph("Section 1: Parties to This Agreement", section_heading_style))
    elements.append(
        Paragraph(
            f"This Split Sheet Agreement (the \"Agreement\") is entered into as of {date}, "
            f"by and between the following parties:",
            body_style,
        )
    )
    elements.append(Spacer(1, 8))

    for c in contributors:
        name = c.get("name", "")
        role = c.get("role", "")
        publisher = c.get("publisher_or_label", "") or ""
        ipi = c.get("ipi_number", "") or ""
        party_line = f"<b>{name}</b> (hereinafter referred to as \"{role}\")"
        if publisher:
            party_line += f", affiliated with {publisher}"
        if ipi:
            party_line += f", IPI/CAE #{ipi}"
        elements.append(Paragraph(party_line, body_style))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 8))

    # --- Section 2: Musical Work ---
    elements.append(Paragraph("Section 2: Musical Work", section_heading_style))
    elements.append(
        Paragraph(
            f"The parties agree that this Agreement pertains to the following musical work:",
            body_style,
        )
    )
    elements.append(Spacer(1, 8))

    work_details = [
        ["Work Title:", work_title],
        ["Work Type:", f"{work_type_display} (song)"],
        ["Date:", date],
    ]
    work_table = Table(work_details, colWidths=[1.5 * inch, 5 * inch])
    work_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ])
    )
    elements.append(work_table)
    elements.append(Spacer(1, 12))

    # --- Section 3: Royalty Splits ---
    elements.append(Paragraph("Section 3: Royalty Percentage Splits", section_heading_style))

    def build_royalty_section(royalty_type_label, royalty_type_key, pct_key):
        """Build a royalty split section with clear, extractable language."""
        elements.append(Spacer(1, 4))
        elements.append(
            Paragraph(
                f"<b>{royalty_type_label} Royalties:</b> The parties agree to the following "
                f"{royalty_type_key.lower()} royalty percentage splits for the work titled "
                f"\"{work_title}\":",
                body_style,
            )
        )
        elements.append(Spacer(1, 8))

        # Explicit text for each party's share (easy for LLM to parse)
        for c in contributors:
            name = c.get("name", "")
            role = c.get("role", "")
            pct = c.get(pct_key, 0) or 0
            elements.append(
                Paragraph(
                    f"• {name} (\"{role}\") shall receive <b>{pct:.2f}%</b> of all "
                    f"{royalty_type_key.lower()} royalties.",
                    body_style,
                )
            )
        elements.append(Spacer(1, 8))

        # Table format
        header_row = ["Party Name", "Role", f"{royalty_type_label} Royalty %"]
        table_data = [header_row]

        for c in contributors:
            pct = c.get(pct_key, 0) or 0
            table_data.append([
                c.get("name", ""),
                c.get("role", ""),
                f"{pct:.2f}%",
            ])

        total_pct = sum((c.get(pct_key, 0) or 0) for c in contributors)
        table_data.append(["", "TOTAL", f"{total_pct:.2f}%"])

        col_widths = [2.5 * inch, 2.0 * inch, 2.0 * inch]
        tbl = Table(table_data, colWidths=col_widths)

        cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a2a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -2), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (2, 1), (2, -1), "CENTER"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 1, colors.black),
            ("GRID", (0, 0), (-1, -2), 0.5, colors.HexColor("#cccccc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#999999")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        for row_idx in range(1, len(table_data) - 1):
            if row_idx % 2 == 0:
                cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f5f5f5")))

        tbl.setStyle(TableStyle(cmds))
        elements.append(tbl)
        elements.append(Spacer(1, 16))

    if needs_pub:
        build_royalty_section("Publishing", "Publishing", "publishing_percentage")

    if needs_master:
        build_royalty_section("Master", "Master", "master_percentage")

    # --- Section 4: Signatures ---
    elements.append(Paragraph("Section 4: Signatures", section_heading_style))
    elements.append(
        Paragraph(
            "By signing below, each party acknowledges and agrees to the royalty "
            "percentage splits as outlined in Section 3 of this Agreement.",
            body_style,
        )
    )
    elements.append(Spacer(1, 12))

    for c in contributors:
        name = c.get("name", "")
        role = c.get("role", "")
        sig_data = [
            [f"Name: {name}", f"Role: {role}"],
            ["Signature: ___________________________", "Date: _______________"],
        ]
        sig_table = Table(sig_data, colWidths=[3.5 * inch, 3.5 * inch])
        sig_table.setStyle(
            TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(sig_table)
        elements.append(Spacer(1, 16))

    # --- Disclaimer ---
    elements.append(
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=8)
    )
    elements.append(
        Paragraph(
            "This split sheet agreement documents the agreed-upon royalty ownership percentages "
            "for the above-referenced musical work. All parties acknowledge and agree to the "
            "royalty splits as outlined. This document is not a substitute for a legally binding "
            "contract and all parties are advised to seek independent legal counsel.",
            disclaimer_style,
        )
    )
    elements.append(
        Paragraph(
            f"Generated by Msanii — {date}",
            ParagraphStyle(
                "Footer",
                parent=styles["Normal"],
                fontSize=8,
                textColor=colors.HexColor("#aaaaaa"),
                alignment=TA_CENTER,
                spaceBefore=12,
            ),
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer
