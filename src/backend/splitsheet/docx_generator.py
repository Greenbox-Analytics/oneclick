import io
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def generate_split_sheet_docx(
    work_title: str,
    work_type: str,
    split_type: str,
    date: str,
    contributors: list[dict],
) -> io.BytesIO:
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    brand_color = RGBColor(0x1A, 0x3A, 0x2A)
    gray_color = RGBColor(0x55, 0x55, 0x55)

    needs_pub = split_type in ("publishing", "both")
    needs_master = split_type in ("master", "both")

    work_type_display = work_type.upper() if work_type else "SINGLE"

    def _add_shading(cell, fill_color):
        shading_element = cell._element.makeelement(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}shd", {}
        )
        shading_element.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}fill", fill_color
        )
        shading_element.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "clear"
        )
        tcPr = cell._element.get_or_add_tcPr()
        tcPr.append(shading_element)

    # --- Header ---
    title = doc.add_heading("SPLIT SHEET AGREEMENT", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = brand_color
        run.font.size = Pt(28)

    subtitle = doc.add_paragraph("Music Work Royalty Split Agreement")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in subtitle.runs:
        run.font.color.rgb = gray_color
        run.font.size = Pt(12)

    doc.add_paragraph("_" * 85)

    # --- Section 1: Parties ---
    heading = doc.add_heading("Section 1: Parties to This Agreement", level=2)
    for run in heading.runs:
        run.font.color.rgb = brand_color

    p = doc.add_paragraph()
    run = p.add_run(
        f'This Split Sheet Agreement (the "Agreement") is entered into as of {date}, '
        f'by and between the following parties:'
    )
    run.font.size = Pt(10)

    doc.add_paragraph()

    for c in contributors:
        name = c.get("name", "")
        role = c.get("role", "")
        publisher = c.get("publisher_or_label", "") or ""
        ipi = c.get("ipi_number", "") or ""

        p = doc.add_paragraph()
        run = p.add_run(f"{name}")
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(f' (hereinafter referred to as "{role}")')
        run.font.size = Pt(10)
        if publisher:
            run = p.add_run(f", affiliated with {publisher}")
            run.font.size = Pt(10)
        if ipi:
            run = p.add_run(f", IPI/CAE #{ipi}")
            run.font.size = Pt(10)

    doc.add_paragraph()

    # --- Section 2: Musical Work ---
    heading = doc.add_heading("Section 2: Musical Work", level=2)
    for run in heading.runs:
        run.font.color.rgb = brand_color

    p = doc.add_paragraph()
    run = p.add_run(
        "The parties agree that this Agreement pertains to the following musical work:"
    )
    run.font.size = Pt(10)

    doc.add_paragraph()

    details = [
        ("Work Title:", work_title),
        ("Work Type:", f"{work_type_display} (song)"),
        ("Date:", date),
    ]
    for label, value in details:
        p = doc.add_paragraph()
        run = p.add_run(f"{label}  ")
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(value)
        run.font.size = Pt(10)

    doc.add_paragraph()

    # --- Section 3: Royalty Splits ---
    heading = doc.add_heading("Section 3: Royalty Percentage Splits", level=2)
    for run in heading.runs:
        run.font.color.rgb = brand_color

    def build_royalty_section(royalty_type_label, royalty_type_key, pct_key):
        # Explicit prose for each party's share
        p = doc.add_paragraph()
        run = p.add_run(f"{royalty_type_label} Royalties:")
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(
            f" The parties agree to the following {royalty_type_key.lower()} royalty "
            f'percentage splits for the work titled "{work_title}":'
        )
        run.font.size = Pt(10)

        doc.add_paragraph()

        for c in contributors:
            name = c.get("name", "")
            role = c.get("role", "")
            pct = c.get(pct_key, 0) or 0
            p = doc.add_paragraph(style="List Bullet")
            run = p.add_run(
                f'{name} ("{role}") shall receive {pct:.2f}% of all '
                f"{royalty_type_key.lower()} royalties."
            )
            run.font.size = Pt(10)

        doc.add_paragraph()

        # Table
        headers = ["Party Name", "Role", f"{royalty_type_label} Royalty %"]
        table = doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table Grid"

        header_row = table.rows[0]
        for i, text in enumerate(headers):
            cell = header_row.cells[i]
            cell.text = text
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            _add_shading(cell, "1A3A2A")

        for idx, c in enumerate(contributors, 1):
            pct = c.get(pct_key, 0) or 0
            row = table.add_row()
            values = [c.get("name", ""), c.get("role", ""), f"{pct:.2f}%"]
            for i, val in enumerate(values):
                cell = row.cells[i]
                cell.text = val
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
            if idx % 2 == 0:
                for cell in row.cells:
                    _add_shading(cell, "F5F5F5")

        total_pct = sum((c.get(pct_key, 0) or 0) for c in contributors)
        total_row = table.add_row()
        total_row.cells[1].text = "TOTAL"
        total_row.cells[2].text = f"{total_pct:.2f}%"
        for cell in total_row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(9)

        doc.add_paragraph()

    if needs_pub:
        build_royalty_section("Publishing", "Publishing", "publishing_percentage")

    if needs_master:
        build_royalty_section("Master", "Master", "master_percentage")

    # --- Section 4: Signatures ---
    heading = doc.add_heading("Section 4: Signatures", level=2)
    for run in heading.runs:
        run.font.color.rgb = brand_color

    p = doc.add_paragraph()
    run = p.add_run(
        "By signing below, each party acknowledges and agrees to the royalty "
        "percentage splits as outlined in Section 3 of this Agreement."
    )
    run.font.size = Pt(10)

    doc.add_paragraph()

    for c in contributors:
        name = c.get("name", "")
        role = c.get("role", "")

        p = doc.add_paragraph()
        run = p.add_run(f"Name: {name}")
        run.font.size = Pt(10)
        p.add_run("          ")
        run = p.add_run(f"Role: {role}")
        run.font.size = Pt(10)

        p = doc.add_paragraph()
        run = p.add_run("Signature: ___________________________")
        run.font.size = Pt(10)
        p.add_run("          ")
        run = p.add_run("Date: _______________")
        run.font.size = Pt(10)

        doc.add_paragraph()

    # --- Disclaimer ---
    doc.add_paragraph("_" * 85)
    disclaimer = doc.add_paragraph(
        "This split sheet agreement documents the agreed-upon royalty ownership percentages "
        "for the above-referenced musical work. All parties acknowledge and agree to the "
        "royalty splits as outlined. This document is not a substitute for a legally binding "
        "contract and all parties are advised to seek independent legal counsel."
    )
    for run in disclaimer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        run.italic = True

    footer = doc.add_paragraph(f"Generated by Msanii \u2014 {date}")
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
