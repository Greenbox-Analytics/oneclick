"""PDF documents for royalty payouts: analysis breakdown and payment receipt.

Both builders render from a royalty_payouts row (dict) and its
breakdown_snapshot, tolerating older/orphaned payouts whose snapshots may be
empty or missing optional fields. Money strings are quantized via
paypal_client.format_amount, which is currency-agnostic and zero-decimal-safe
(JPY/HUF/TWD) — never format royalty amounts with raw floats.
"""

import io
from decimal import Decimal

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

from oneclick.royalties import paypal_client

BRAND = colors.HexColor("#1a3a2a")
MUTED = colors.HexColor("#555555")
FAINT = colors.HexColor("#888888")


def fmt_money(amount, currency: str) -> str:
    """Format an amount as '1,234.56 USD' (zero-decimal currencies as '1,235 JPY')."""
    try:
        quantized = paypal_client.format_amount(float(amount), currency)
    except (ValueError, TypeError):
        # format_amount rejects non-positive amounts; render them plainly.
        zero = "0" if (currency or "").upper() in paypal_client.ZERO_DECIMAL_CURRENCIES else "0.00"
        return f"{zero} {currency or ''}".strip()
    return f"{Decimal(quantized):,} {currency}"


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    return str(iso)[:10]


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PayoutTitle", parent=base["Title"], fontSize=24, spaceAfter=4, textColor=BRAND, alignment=TA_CENTER
        ),
        "subtitle": ParagraphStyle(
            "PayoutSubtitle", parent=base["Normal"], fontSize=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=16
        ),
        "section": ParagraphStyle(
            "PayoutSection", parent=base["Heading2"], fontSize=13, textColor=BRAND, spaceBefore=14, spaceAfter=6
        ),
        "body": ParagraphStyle("PayoutBody", parent=base["Normal"], fontSize=10, leading=14),
        "small": ParagraphStyle(
            "PayoutSmall", parent=base["Normal"], fontSize=8, textColor=FAINT, leading=11, spaceBefore=14
        ),
        "cell": ParagraphStyle("PayoutCell", parent=base["Normal"], fontSize=8.5, leading=11),
    }


def _doc(buffer: io.BytesIO) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )


def _header(elements: list, styles: dict, title: str, subtitle: str) -> None:
    elements.append(Paragraph(title, styles["title"]))
    elements.append(Paragraph(subtitle, styles["subtitle"]))
    elements.append(HRFlowable(width="100%", thickness=2, color=BRAND, spaceAfter=16))


def _facts_table(rows: list[tuple[str, str]]) -> Table:
    table = Table([[label, value] for label, value in rows], colWidths=[1.8 * inch, 5.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TEXTCOLOR", (0, 0), (0, -1), BRAND),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _payment_method_label(payout: dict) -> str:
    if payout.get("payment_method") == "paypal":
        capture_id = payout.get("paypal_capture_id")
        return f"PayPal (transaction {capture_id})" if capture_id else "PayPal"
    return "Manual transfer"


def generate_breakdown_pdf(payout: dict) -> io.BytesIO:
    """Full payout analysis: per-project statement tables with royalty lines."""
    buffer = io.BytesIO()
    doc = _doc(buffer)
    styles = _styles()
    elements: list = []

    snapshot = payout.get("breakdown_snapshot") or {}
    payee = snapshot.get("payee") or {}
    payee_name = payee.get("display_name") or "Unknown payee"
    pay_ccy = payout.get("pay_currency") or payee.get("payout_currency") or ""
    status = (payout.get("status") or "draft").capitalize()

    _header(elements, styles, "PAYOUT ANALYSIS", "Royalty payout breakdown")

    facts = [
        ("Payee", payee_name),
        ("Status", status),
        ("Created", _fmt_date(payout.get("created_at"))),
    ]
    if payout.get("paid_at"):
        facts.append(("Paid", _fmt_date(payout.get("paid_at"))))
    facts.append(("Total payout", fmt_money(payout.get("total_amount", 0), pay_ccy)))
    if payout.get("note"):
        facts.append(("Note", str(payout["note"])))
    elements.append(_facts_table(facts))

    projects = snapshot.get("projects") or []
    if not projects:
        elements.append(Spacer(1, 12))
        elements.append(
            Paragraph(
                "Breakdown details are not available for this payout.",
                styles["body"],
            )
        )
    for project in projects:
        elements.append(Paragraph(project.get("name") or "Untitled project", styles["section"]))
        for stmt in project.get("statements") or []:
            stmt_ccy = stmt.get("statement_currency") or ""
            period = f"{_fmt_date(stmt.get('period_start'))} – {_fmt_date(stmt.get('period_end'))}"
            stmt_total = stmt.get("statement_total")
            meta = f"Statement period: {period}"
            if stmt_total is not None:
                meta += f" &nbsp;·&nbsp; Statement total: {fmt_money(stmt_total, stmt_ccy)}"
            elements.append(Paragraph(meta, styles["body"]))
            elements.append(Spacer(1, 4))

            header = ["Song", "Role", "Type", "Share %", f"Owed ({stmt_ccy})", f"Pay ({pay_ccy})"]
            rows = [header]
            for line in stmt.get("lines") or []:
                pct = line.get("percentage")
                rows.append(
                    [
                        Paragraph(str(line.get("song") or "—"), styles["cell"]),
                        str(line.get("role") or "—"),
                        str(line.get("royalty_type") or "—"),
                        f"{pct:g}%" if pct is not None else "—",
                        fmt_money(line.get("amount_owed", 0), stmt_ccy),
                        fmt_money(line.get("amount_pay_ccy", 0), pay_ccy),
                    ]
                )
            rows.append(
                [
                    "Subtotal",
                    "",
                    "",
                    "",
                    fmt_money(stmt.get("payee_subtotal_owed", 0), stmt_ccy),
                    fmt_money(stmt.get("payee_subtotal_pay_ccy", 0), pay_ccy),
                ]
            )
            table = Table(
                rows, colWidths=[1.9 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch, 1.3 * inch, 1.3 * inch], repeatRows=1
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f4f7f5")]),
                        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8efeb")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                        ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 10))

    # Grand total + FX footnote
    elements.append(HRFlowable(width="100%", thickness=1, color=BRAND, spaceBefore=6, spaceAfter=8))
    total = snapshot.get("total_pay_ccy", payout.get("total_amount", 0))
    elements.append(Paragraph(f"<b>Total payout: {fmt_money(total, pay_ccy)}</b>", styles["body"]))

    fx = snapshot.get("fx") or {}
    rates = fx.get("rates_used") or {}
    if rates:
        pairs = " · ".join(f"{pair}: {rate}" for pair, rate in rates.items())
        elements.append(
            Paragraph(
                f"FX rates as of {fx.get('rate_date') or '—'}: {pairs}",
                styles["small"],
            )
        )
    elements.append(
        Paragraph(f"Generated by Msanii OneClick · Payout reference {payout.get('id') or '—'}", styles["small"])
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer


def generate_receipt_pdf(payout: dict, payer_name: str | None = None) -> io.BytesIO:
    """One-page payment receipt for a paid payout."""
    buffer = io.BytesIO()
    doc = _doc(buffer)
    styles = _styles()
    elements: list = []

    snapshot = payout.get("breakdown_snapshot") or {}
    payee = snapshot.get("payee") or {}
    payee_name = payee.get("display_name") or "Unknown payee"
    pay_ccy = payout.get("pay_currency") or payee.get("payout_currency") or ""

    _header(elements, styles, "PAYMENT RECEIPT", "Royalty payout confirmation")

    facts = [
        ("Paid to", payee_name),
        ("Amount", fmt_money(payout.get("total_amount", 0), pay_ccy)),
        ("Payment date", _fmt_date(payout.get("paid_at"))),
        ("Payment method", _payment_method_label(payout)),
    ]
    if payer_name:
        facts.append(("Paid by", payer_name))
    facts.append(("Payout reference", payout.get("id") or "—"))
    if payout.get("note"):
        facts.append(("Note", str(payout["note"])))
    elements.append(_facts_table(facts))

    projects = snapshot.get("projects") or []
    if projects:
        elements.append(Paragraph("Payment breakdown by project", styles["section"]))
        rows = [["Project", f"Amount ({pay_ccy})"]]
        for project in projects:
            proj_total = sum(stmt.get("payee_subtotal_pay_ccy", 0) or 0 for stmt in project.get("statements") or [])
            rows.append([project.get("name") or "Untitled project", fmt_money(proj_total, pay_ccy)])
        rows.append(["Total", fmt_money(payout.get("total_amount", 0), pay_ccy)])
        table = Table(rows, colWidths=[4.5 * inch, 2.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f4f7f5")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(table)

    elements.append(
        Paragraph(
            "This receipt confirms a royalty payment recorded in Msanii OneClick. "
            "For a full line-by-line analysis, see the payout breakdown document.",
            styles["small"],
        )
    )

    elements.append(
        Paragraph(
            "<b>Please verify:</b> This is a system-generated record of a payment marked as "
            "completed in Msanii OneClick — it is not proof that funds were received. Always "
            "confirm the transaction independently (e.g. through your bank or PayPal) to ensure "
            "the payment actually reached the recipient.",
            styles["small"],
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer
