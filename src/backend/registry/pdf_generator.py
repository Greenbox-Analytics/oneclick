"""Work metadata PDF export with approval status per stakeholder."""

import hashlib
import html
import io
from datetime import UTC, datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BRAND = colors.HexColor("#1a3a2a")
GREEN = colors.HexColor("#16a34a")
RED = colors.HexColor("#dc2626")
AMBER = colors.HexColor("#d97706")


def _wrap(value, style):
    """Wrap a cell value in a Paragraph so long text wraps inside its column.

    Bare strings don't wrap in fixed-width Table cells and overflow into the
    next column. Values are escaped because Paragraph parses XML-ish markup
    (names/titles can legitimately contain '&' or '<').
    """
    text = str(value) if value not in (None, "") else "—"
    return Paragraph(html.escape(text), style)


WITHHELD_TEXT = "Withheld"


def generate_proof_of_ownership_pdf(work_data: dict, hidden_parties: set[str] | None = None) -> io.BytesIO:
    # Parties (by holder_name) whose ownership percentages should be redacted in
    # this export — the exporter chose not to disclose their splits. The party is
    # still listed, but its % (and section total) show a "Withheld" message.
    hidden_parties = hidden_parties or set()
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
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontSize=24, spaceAfter=4, textColor=BRAND, alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555"),
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=14, textColor=BRAND, spaceBefore=16, spaceAfter=8
    )
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888"), leading=11
    )
    cell_style = ParagraphStyle("Cell", parent=body_style, fontSize=9, leading=12)
    lic_cell_style = ParagraphStyle("LicCell", parent=body_style, fontSize=8, leading=10)

    collaborators = work_data.get("collaborators", [])
    stake_approval = {}
    email_approval = {}
    for c in collaborators:
        if c.get("stake_id"):
            stake_approval[c["stake_id"]] = {"status": c["status"], "name": c["name"]}
        if c.get("email"):
            email_approval[c["email"].lower()] = {"status": c["status"], "name": c["name"]}

    # Header
    elements.append(Paragraph("WORK METADATA", title_style))
    elements.append(Paragraph("Metadata Registry Certificate", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=BRAND, spaceAfter=20))

    # Status banner
    work_status = (work_data.get("status") or "draft").replace("_", " ").title()
    status_color = {"Registered": GREEN, "Disputed": RED, "Pending Approval": AMBER}.get(
        work_status, colors.HexColor("#666")
    )
    elements.append(Paragraph(f"<b>Registry Status: <font color='{status_color}'>{work_status}</font></b>", body_style))
    elements.append(Spacer(1, 8))

    # Work details
    elements.append(Paragraph("Work Details", section_style))
    details = [
        ["Title:", _wrap(work_data.get("title", "—"), body_style)],
        ["Type:", (work_data.get("work_type") or "single").replace("_", " ").title()],
        ["ISRC:", _wrap(work_data.get("isrc"), body_style)],
        ["ISWC:", _wrap(work_data.get("iswc"), body_style)],
        ["UPC:", _wrap(work_data.get("upc"), body_style)],
        ["Release Date:", str(work_data.get("release_date") or "—")],
    ]
    dt = Table(details, colWidths=[1.5 * inch, 5 * inch])
    dt.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(dt)
    elements.append(Spacer(1, 12))

    # Ownership with approval
    stakes = work_data.get("stakes", [])

    def build_stakes_section(label, stake_list, heading=None):
        elements.append(Paragraph(heading or f"{label} Ownership", section_style))
        if not stake_list:
            elements.append(Paragraph(f"No {label.lower()} ownership recorded.", body_style))
            elements.append(Spacer(1, 8))
            return

        header = ["Holder", "Role", "%", "Publisher/Label", "Approval"]
        rows = [header]
        section_has_hidden = False
        for s in stake_list:
            approval = "—"
            sid = s.get("id")
            hemail = (s.get("holder_email") or "").lower()
            if sid in stake_approval:
                approval = stake_approval[sid]["status"].title()
            elif hemail and hemail in email_approval:
                approval = email_approval[hemail]["status"].title()
            # Redact this party's split if the exporter withheld it: keep the
            # holder/role/approval so the party is still listed, but replace the
            # percentage (and publisher/label, which can imply the split) with a
            # "Withheld" message.
            is_hidden = s.get("holder_name") in hidden_parties
            if is_hidden:
                section_has_hidden = True
            pct_cell = WITHHELD_TEXT if is_hidden else f"{s.get('percentage', 0):.2f}%"
            publisher_cell = (
                _wrap(WITHHELD_TEXT, cell_style) if is_hidden else _wrap(s.get("publisher_or_label"), cell_style)
            )
            # Free-text cells are wrapped so long names can't overlap the next
            # column; % and Approval stay plain strings — the per-row TEXTCOLOR
            # styling below only applies to string cells.
            rows.append(
                [
                    _wrap(s.get("holder_name", ""), cell_style),
                    _wrap(s.get("holder_role", ""), cell_style),
                    pct_cell,
                    publisher_cell,
                    approval,
                ]
            )
        # When any party in this section is withheld, redact the total too —
        # otherwise a recipient could derive a hidden split by subtracting the
        # visible ones from the total.
        if section_has_hidden:
            rows.append(["", "TOTAL", WITHHELD_TEXT, "", ""])
        else:
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
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
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

    # SoundExchange shares are paid directly by SoundExchange, so they're shown
    # in their own section — never folded into the master totals above — and
    # only when the work actually has some (most works have none).
    soundexchange = [s for s in stakes if s.get("stake_type") == "soundexchange"]
    if soundexchange:
        build_stakes_section("SoundExchange", soundexchange, heading="SoundExchange Royalties")
        elements.append(
            Paragraph(
                "SoundExchange royalties (US non-interactive digital performance) are "
                "collected and paid directly by SoundExchange. They are tracked "
                "separately and are not counted in the master ownership totals above.",
                small_style,
            )
        )
        elements.append(Spacer(1, 8))

    # Explain the "Withheld" cells whenever the exporter hid one or more splits.
    if hidden_parties:
        elements.append(
            Paragraph(
                f"Splits marked '{WITHHELD_TEXT}' were hidden by the exporter and are not disclosed in this document.",
                small_style,
            )
        )
        elements.append(Spacer(1, 8))

    # Licensing
    licenses = work_data.get("licenses", [])
    elements.append(Paragraph("Licensing Rights", section_style))
    if not licenses:
        elements.append(Paragraph("No licensing rights recorded.", body_style))
    else:
        rows = [["Type", "Licensee", "Territory", "Start", "End", "Status"]]
        for lic in licenses:
            rows.append(
                [
                    (lic.get("license_type") or "").replace("_", " ").title(),
                    _wrap(lic.get("licensee_name", ""), lic_cell_style),
                    _wrap(lic.get("territory", ""), lic_cell_style),
                    str(lic.get("start_date", "—")),
                    str(lic.get("end_date") or "Perpetual"),
                    (lic.get("status") or "active").title(),
                ]
            )
        tbl = Table(rows, colWidths=[1.0 * inch, 1.4 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccc")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
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
            # Escape interpolated values — Paragraph parses markup and titles/names
            # can contain '&' or '<'.
            party_names = html.escape(party_names)
            elements.append(Paragraph(f"<b>{html.escape(agr.get('title', ''))}</b> — {agr_type}", body_style))
            elements.append(
                Paragraph(
                    f"Effective: {agr.get('effective_date', '—')} | Recorded: {agr.get('created_at', '—')}", small_style
                )
            )
            elements.append(Paragraph(f"Parties: {party_names}", small_style))
            if agr.get("document_hash"):
                elements.append(Paragraph(f"Hash: {agr['document_hash']}", small_style))
            elements.append(Spacer(1, 6))

    # Signatures — one block per unique stakeholder across all stake types, so
    # the same person holding both a master and a publishing stake signs once.
    elements.append(Paragraph("Signatures", section_style))
    elements.append(
        Paragraph(
            "By signing below, each party confirms that the ownership splits and "
            "rights information recorded in this document are accurate.",
            body_style,
        )
    )
    elements.append(Spacer(1, 12))

    signers = {}
    for s in stakes:
        holder_name = (s.get("holder_name") or "").strip()
        if not holder_name:
            continue
        key = ((s.get("holder_email") or "").strip().lower() or holder_name.lower(), holder_name.lower())
        role = (s.get("holder_role") or "").strip()
        signer = signers.setdefault(key, {"name": holder_name, "roles": []})
        if role and role not in signer["roles"]:
            signer["roles"].append(role)

    if not signers:
        elements.append(Paragraph("No stakeholders recorded for this work.", body_style))
    for signer in signers.values():
        role_text = ", ".join(signer["roles"]) or "—"
        sig_rows = [
            [_wrap(f"Name: {signer['name']}", body_style), _wrap(f"Role: {role_text}", body_style)],
            ["Signature: ___________________________", "Date: _______________"],
        ]
        sig_table = Table(sig_rows, colWidths=[3.5 * inch, 3.5 * inch])
        sig_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(sig_table)
        elements.append(Spacer(1, 16))

    # Footer
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccc"), spaceAfter=8))
    content_str = f"{work_data.get('id', '')}|{work_data.get('title', '')}|{generated_at}"
    doc_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

    elements.append(
        Paragraph(
            "This document reflects the metadata, ownership splits, and rights information "
            "recorded for this work in the Msanii Metadata Registry. Stakeholder approval status "
            "indicates whether each party has confirmed their stake. A 'Registered' status means "
            "all parties have agreed.",
            ParagraphStyle("Disc", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888"), leading=11),
        )
    )
    elements.append(Spacer(1, 4))
    elements.append(
        Paragraph(
            f"Generated: {generated_at} | Document ID: {doc_hash}",
            ParagraphStyle(
                "Foot", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#aaa"), alignment=TA_CENTER
            ),
        )
    )
    elements.append(
        Paragraph(
            "Msanii Metadata Registry",
            ParagraphStyle(
                "Brand", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#aaa"), alignment=TA_CENTER
            ),
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer
