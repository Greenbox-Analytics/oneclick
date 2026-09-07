"""Downloadable PDF for the usage payloads built by orgs.service.org_usage_rollup.

ONE renderer behind three routes — GET /orgs/{id}/usage/report.pdf, its
Msanii-admin twin, and GET /me/api-usage/report.pdf — so the three can never
drift apart. `sections` is what makes the personal report multi-org: one
sub-heading + chart + key/folder tables per org, no member table.

Bucketing mirrors src/lib/orgUsage.ts (bucketSeries/bucketLabel) exactly: days
for 7d/14d/mtd, ISO weeks for 1y, months for all; day and week windows are
gap-filled from `since` to today so a quiet day reads as zero. Keep the two in
step or the PDF and the on-screen card will disagree about the same window.
"""

import io
import re
from datetime import UTC, date, datetime, timedelta
from html import escape

from fastapi import Response
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
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
MUTED = colors.HexColor("#666666")
SHADE = colors.HexColor("#f2f5f3")
RULE = colors.HexColor("#dddddd")

# The backend's copy of src/lib/usageTools.ts: (id, label, actions, colour).
# Both are exhaustive over the credit actions; an action neither knows about is
# ignored everywhere rather than crashing or landing in a fifth bucket.
TOOLS = (
    ("oneclick", "OneClick", ("oneclick_run", "partner_oneclick_run"), "#5fbf7a"),
    ("registry", "Registry", ("registry_parse", "partner_registry_parse"), "#6ea8f5"),
    ("splitsheet", "Split sheet", ("split_sheet", "partner_split_sheet"), "#4fc3c8"),
    ("zoe", "Zoe", ("zoe_message", "partner_zoe_message"), "#b48ef0"),
)
_ACTION_TOOL = {action: tool[0] for tool in TOOLS for action in tool[2]}

CONTENT_WIDTH = 504  # letter minus the 0.75in margins


# ---- pure helpers (unit-tested; no reportlab) ---------------------------------


def _empty_tools() -> dict[str, int]:
    return {t[0]: 0 for t in TOOLS}


def bucket_of(day: str, range_: str) -> str:
    """The chart bucket a UTC day falls in: its Monday for 1y, the 1st of its
    month for all, itself otherwise."""
    if range_ == "1y":
        d = date.fromisoformat(day[:10])
        return (d - timedelta(days=d.weekday())).isoformat()
    if range_ == "all":
        return f"{day[:7]}-01"
    return day[:10]


def bucket_label(bucket: str, range_: str) -> str:
    d = date.fromisoformat(bucket)
    return f"{d:%b} {d.year}" if range_ == "all" else f"{d:%b} {d.day}"


def bucket_series(
    series: list[dict], range_: str, since: str | None, today: str | None = None
) -> list[dict[str, object]]:
    """[{bucket, label, tools, credits}] — the chart's x axis. Day and week
    windows are filled from `since` to `today` (so gaps read as zero); all-time
    lists only the months that actually carry spend."""
    today = today or datetime.now(UTC).date().isoformat()
    buckets: dict[str, dict[str, int]] = {}
    if since and range_ != "all":
        step = timedelta(days=7 if range_ == "1y" else 1)
        end = date.fromisoformat(today[:10])
        cursor = date.fromisoformat(bucket_of(since, range_))
        while cursor <= end:
            buckets[cursor.isoformat()] = _empty_tools()
            cursor += step
    for day in series or []:
        tools = buckets.setdefault(bucket_of(day["day"], range_), _empty_tools())
        for a in day.get("actions") or []:
            tool_id = _ACTION_TOOL.get(a.get("action"))
            if tool_id:
                tools[tool_id] += a.get("credits", 0)
    return [
        {"bucket": b, "label": bucket_label(b, range_), "tools": t, "credits": sum(t.values())}
        for b, t in sorted(buckets.items())
    ]


def merge_series(series_list) -> list[dict]:
    """Fold several per-day series (per key, or per org) into one, same wire
    shape. Used for the personal report's per-org and overall charts."""
    days: dict[str, dict[str, dict]] = {}
    for series in series_list:
        for day in series or []:
            bucket = days.setdefault(day["day"], {})
            for a in day.get("actions") or []:
                slot = bucket.setdefault(a["action"], {"action": a["action"], "credits": 0, "runs": 0})
                slot["credits"] += a.get("credits", 0)
                slot["runs"] += a.get("runs", 0)
    return [
        {"day": d, "actions": sorted(acts.values(), key=lambda a: -a["credits"])} for d, acts in sorted(days.items())
    ]


def _fold(actions) -> dict[str, int]:
    tools = _empty_tools()
    for a in actions or []:
        tool_id = _ACTION_TOOL.get(a.get("action"))
        if tool_id:
            tools[tool_id] += a.get("credits", 0)
    return tools


def _fold_series(series) -> dict[str, int]:
    tools = _empty_tools()
    for day in series or []:
        for tool_id, credits in _fold(day.get("actions")).items():
            tools[tool_id] += credits
    return tools


def _totals(series) -> tuple[int, int]:
    credits = runs = 0
    for day in series or []:
        for a in day.get("actions") or []:
            credits += a.get("credits", 0)
            runs += a.get("runs", 0)
    return credits, runs


def _fmt_day(d: date) -> str:
    return f"{d:%b} {d.day}, {d.year}"


def window_label(range_: str, since: str | None, today: date | None = None) -> str:
    if not since or range_ == "all":
        return "All time"
    return f"{_fmt_day(date.fromisoformat(since[:10]))} – {_fmt_day(today or datetime.now(UTC).date())}"


def report_filename(name: str, range_: str, today: date | None = None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40].strip("-") or "team"
    return f"usage-report-{slug}-{range_}-{(today or datetime.now(UTC).date()).isoformat()}.pdf"


def pdf_response(pdf: bytes, name: str, range_: str) -> Response:
    """The one download response shape all three routes return."""
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{report_filename(name, range_)}"'},
    )


# ---- styles -------------------------------------------------------------------

_base = getSampleStyleSheet()
_TITLE = ParagraphStyle("UsageTitle", parent=_base["Title"], fontSize=22, alignment=0, spaceAfter=2, textColor=BRAND)
_SUB = ParagraphStyle("UsageSub", parent=_base["Normal"], fontSize=11, textColor=MUTED, spaceAfter=2)
_META = ParagraphStyle("UsageMeta", parent=_base["Normal"], fontSize=8, textColor=MUTED)
# keepWithNext: a section heading must never orphan at the foot of a page.
_H2 = ParagraphStyle(
    "UsageH2", parent=_base["Heading2"], fontSize=13, textColor=BRAND, spaceBefore=16, spaceAfter=6, keepWithNext=1
)
_H3 = ParagraphStyle(
    "UsageH3", parent=_base["Heading3"], fontSize=11, textColor=BRAND, spaceBefore=14, spaceAfter=4, keepWithNext=1
)
_BODY = ParagraphStyle("UsageBody", parent=_base["Normal"], fontSize=9, leading=12)
_TILE = ParagraphStyle("UsageTile", parent=_base["Normal"], fontSize=8, leading=13)
_TH = ParagraphStyle(
    "UsageTH", parent=_base["Normal"], fontSize=8, leading=10, textColor=colors.white, fontName="Helvetica-Bold"
)
_THR = ParagraphStyle("UsageTHR", parent=_TH, alignment=2)
_TD = ParagraphStyle("UsageTD", parent=_base["Normal"], fontSize=8, leading=10)
_TDR = ParagraphStyle("UsageTDR", parent=_TD, alignment=2)


def _table(headers: list[str], rows: list[list], widths: list[float], right: set[int]) -> Table:
    """Paragraph cells (so long labels wrap), repeating header, zebra rows."""
    data = [[Paragraph(escape(str(h)), _THR if i in right else _TH) for i, h in enumerate(headers)]]
    data += [[Paragraph(escape(str(c)), _TDR if i in right else _TD) for i, c in enumerate(r)] for r in rows]
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
    ]
    style += [("BACKGROUND", (0, i), (-1, i), SHADE) for i in range(2, len(data), 2)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def _tile(label: str, value, sub: str = "") -> Paragraph:
    return Paragraph(
        f'<font size="7" color="#666666">{escape(label.upper())}</font><br/>'
        f'<font size="16" color="#1a3a2a"><b>{escape(str(value))}</b></font><br/>'
        f'<font size="7" color="#666666">{escape(sub) if sub else "&nbsp;"}</font>',
        _TILE,
    )


def _chart(buckets: list[dict], width: float = CONTENT_WIDTH) -> Drawing | None:
    """Stacked bars, one series per tool that has spend. None when nothing was
    spent in the window — the caller prints a line instead."""
    active = [t for t in TOOLS if any(b["tools"][t[0]] for b in buckets)]
    if not active or not buckets:
        return None
    drawing = Drawing(width, 215)
    chart = VerticalBarChart()
    chart.x, chart.y = 40, 54
    chart.width, chart.height = width - 60, 145
    chart.data = [[b["tools"][t[0]] for b in buckets] for t in active]
    chart.categoryAxis.style = "stacked"
    # Thin the labels out rather than letting them collide.
    step = max(1, -(-len(buckets) // 14))
    chart.categoryAxis.categoryNames = [b["label"] if i % step == 0 else "" for i, b in enumerate(buckets)]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 6
    chart.categoryAxis.labels.dy = -2
    chart.valueAxis.valueMin = 0
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 6
    chart.barSpacing = 0
    chart.groupSpacing = 3
    for i, tool in enumerate(active):
        chart.bars[i].fillColor = colors.HexColor(tool[3])
        chart.bars[i].strokeWidth = 0.25
        chart.bars[i].strokeColor = colors.white
    drawing.add(chart)

    legend = Legend()
    legend.x, legend.y = 40, 10
    legend.boxAnchor = "sw"
    legend.alignment = "right"  # text to the RIGHT of the swatch
    legend.columnMaximum = 1  # one item per column => a single horizontal row
    legend.deltax = 85
    legend.fontName, legend.fontSize = "Helvetica", 7
    legend.dxTextSpace = 4
    legend.dx = legend.dy = 6
    legend.strokeWidth = 0
    legend.strokeColor = None
    legend.colorNamePairs = [(colors.HexColor(t[3]), t[1]) for t in active]
    drawing.add(legend)
    return drawing


def _chart_block(series, range_: str, since: str | None, today: str | None) -> list:
    drawing = _chart(bucket_series(series, range_, since, today))
    if drawing is None:
        return [Paragraph("No credits used in this window.", _BODY), Spacer(1, 6)]
    return [drawing, Spacer(1, 6)]


def _by_tool_rows(series) -> list[list]:
    tools = _fold_series(series)
    total = sum(tools.values())
    return [[t[1], f"{tools[t[0]]:,}", f"{(tools[t[0]] / total * 100 if total else 0):.0f}%"] for t in TOOLS]


def _member_rows(seats: list[dict]) -> tuple[list[str], list[list], list[float], set[int]]:
    rows = []
    for s in seats:
        tools = _fold(s.get("byAction"))
        rows.append(
            {
                "email": s.get("email") or "Unknown",
                "role": (s.get("role") or "—").capitalize(),
                "total": (s.get("spentThisPeriod") or 0) + (s.get("apiCredits") or 0),
                "runs": sum(a.get("runs", 0) for a in s.get("byAction") or []),
                "tools": tools,
            }
        )
    rows.sort(key=lambda r: -r["total"])
    # Only tools with spend somewhere in this table earn a column.
    active = [t for t in TOOLS if any(r["tools"][t[0]] for r in rows)]
    headers = ["Member", "Role", "Credits", "Runs"] + [t[1] for t in active]
    body = [
        [r["email"], r["role"], f"{r['total']:,}", f"{r['runs']:,}"] + [f"{r['tools'][t[0]]:,}" for t in active]
        for r in rows
    ]
    fixed = [150, 60, 55, 45]
    tool_width = (CONTENT_WIDTH - sum(fixed)) / len(active) if active else 0
    widths = fixed + [tool_width] * len(active)
    if not active:
        widths = [220, 90, 100, 94]
    return headers, body, widths, set(range(2, len(headers)))


def render_org_report(title: str, payload: dict) -> bytes:
    """An org_usage_rollup payload straight to PDF — the adapter both
    org-usage routes (the admin console's and the Msanii-admin twin's) share so
    the two can't map the payload differently."""
    return render_usage_report(
        title=title,
        range_=payload.get("range", "mtd"),
        since=payload.get("since"),
        series=payload.get("series") or [],
        seats=payload.get("seats"),
        by_key=payload.get("byKey") or [],
        by_folder=payload.get("byFolder") or [],
    )


def render_usage_report(
    *,
    title: str,
    range_: str,
    since: str | None,
    series: list,
    seats: list | None,
    by_key: list,
    by_folder: list,
    subtitle: str = "Usage report",
    sections: list[dict] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """The usage PDF.

    `series`/`seats` drive the header tiles and the By tool table. `sections`
    (each {heading, series, seats, by_key, by_folder}) splits the body — the
    personal report passes one per org, everything else leaves it None and the
    flat `series`/`seats`/`by_key`/`by_folder` become the single implicit
    section. A section with a heading carries its own chart; the implicit one
    charts at the top instead.
    """
    generated_at = generated_at or datetime.now(UTC)
    today = generated_at.astimezone(UTC).date()
    today_iso = today.isoformat()
    multi = sections is not None
    if not multi:
        sections = [{"heading": None, "series": series, "seats": seats, "by_key": by_key, "by_folder": by_folder}]

    credits, runs = _totals(series)
    tools = _fold_series(series)
    top = max(TOOLS, key=lambda t: tools[t[0]]) if credits else None

    story: list = [
        Paragraph(escape(title or "Team"), _TITLE),
        Paragraph(escape(subtitle), _SUB),
        Paragraph(
            f"{escape(window_label(range_, since, today))} &nbsp;·&nbsp; "
            f"Generated {generated_at.astimezone(UTC):%b %d, %Y %H:%M} UTC",
            _META,
        ),
        HRFlowable(width="100%", thickness=2, color=BRAND, spaceBefore=8, spaceAfter=12),
    ]

    tiles = [
        _tile("Credits used", f"{credits:,}"),
        _tile("Runs", f"{runs:,}"),
        _tile(
            "Most used tool",
            top[1] if top else "—",
            f"{tools[top[0]] / credits * 100:.0f}% of credits" if top else "",
        ),
    ]
    if seats is not None:
        active_members = sum(1 for s in seats if (s.get("spentThisPeriod") or 0) + (s.get("apiCredits") or 0) > 0)
        tiles.append(_tile("Active members", active_members, f"of {len(seats)}"))
    tile_table = Table([tiles], colWidths=[CONTENT_WIDTH / len(tiles)] * len(tiles))
    tile_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story += [tile_table, Spacer(1, 4)]

    if not multi:
        story.append(Paragraph("Credits over time", _H2))
        story += _chart_block(series, range_, since, today_iso)

    story.append(Paragraph("By tool", _H2))
    story.append(_table(["Tool", "Credits", "Share"], _by_tool_rows(series), [200, 152, 152], {1, 2}))

    for section in sections:
        heading = section.get("heading")
        if heading:
            story.append(Paragraph(escape(heading), _H2))
            story += _chart_block(section.get("series"), range_, since, today_iso)

        section_seats = section.get("seats")
        if section_seats is not None:
            story.append(Paragraph("By member", _H3 if heading else _H2))
            headers, rows, widths, right = _member_rows(section_seats)
            story.append(_table(headers, rows, widths, right))

        story.append(Paragraph("By key", _H3 if heading else _H2))
        story.append(
            _table(
                ["Key", "Prefix", "Status", "Folder", "Credits", "Runs", "Last used"],
                [
                    [
                        k.get("label") or "Untitled",
                        f"{k.get('keyPrefix') or ''}…",
                        (k.get("status") or "—").capitalize(),
                        k.get("folderName") or "No folder",
                        f"{k.get('credits', 0):,}",
                        f"{k.get('runs', 0):,}",
                        (k.get("lastUsedAt") or "")[:10] or "—",
                    ]
                    for k in section.get("by_key") or []
                ],
                [104, 96, 56, 88, 50, 40, 70],
                {4, 5},
            )
        )

        story.append(Paragraph("By folder", _H3 if heading else _H2))
        story.append(
            _table(
                ["Folder", "Keys", "Credits", "Runs"],
                [
                    [
                        f.get("name") or "No folder",
                        f"{f.get('keys', 0):,}",
                        f"{f.get('credits', 0):,}",
                        f"{f.get('runs', 0):,}",
                    ]
                    for f in section.get("by_folder") or []
                ],
                [264, 80, 80, 80],
                {1, 2, 3},
            )
        )

    buffer = io.BytesIO()
    SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title=f"{title} — {subtitle}",
    ).build(story)
    return buffer.getvalue()
