"""The downloadable usage PDF (orgs/usage_report.py) and the three routes that
serve it.

pypdf/pdfplumber are not installed, so the content assertions turn PDF
compression off (`rl_config.pageCompression = 0`) and look for the section
headings in the raw content stream. The bucketing that decides the chart's x
axis is a pure function and is unit-tested directly against the rules in
src/lib/orgUsage.ts, which the PDF must mirror.
"""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from orgs import usage_report as ur

ORG = "20000000-0000-0000-0000-0000000000d1"


@pytest.fixture(autouse=True)
def _readable_pdfs(monkeypatch):
    """Uncompressed content streams so `b"By member" in pdf` means something."""
    from reportlab import rl_config

    monkeypatch.setattr(rl_config, "pageCompression", 0)


def _day(day, *actions):
    return {"day": day, "actions": [{"action": a, "credits": c, "runs": r} for a, c, r in actions]}


SERIES = [
    _day("2026-09-02", ("oneclick_run", 60, 2), ("zoe_message", 5, 1)),
    _day("2026-09-05", ("registry_parse", 30, 1)),
    _day("2026-09-06", ("partner_split_sheet", 20, 1)),
]
SEATS = [
    {
        "email": "ada@label.test",
        "role": "admin",
        "spentThisPeriod": 65,
        "apiCredits": 20,
        "byAction": [
            {"action": "oneclick_run", "credits": 60, "runs": 2},
            {"action": "partner_split_sheet", "credits": 20, "runs": 1},
            {"action": "zoe_message", "credits": 5, "runs": 1},
        ],
    },
    {"email": None, "role": "member", "spentThisPeriod": 0, "apiCredits": 0, "byAction": []},
]
KEYS = [
    {
        "keyId": "k1",
        "label": "Prod",
        "keyPrefix": "mk_live_aaaa",
        "status": "active",
        "folderName": "Ingest",
        "credits": 20,
        "runs": 1,
        "lastUsedAt": "2026-09-06T10:00:00+00:00",
        "byAction": [{"action": "partner_split_sheet", "credits": 20, "runs": 1}],
        "series": [_day("2026-09-06", ("partner_split_sheet", 20, 1))],
    },
    {
        "keyId": "k2",
        "label": "Idle",
        "keyPrefix": "mk_live_bbbb",
        "status": "revoked",
        "folderName": None,
        "credits": 0,
        "runs": 0,
        "lastUsedAt": None,
        "byAction": [],
        "series": [],
    },
]
FOLDERS = [
    {"folderId": "f1", "name": "Ingest", "keys": 1, "credits": 20, "runs": 1, "series": KEYS[0]["series"]},
    {"folderId": None, "name": "No folder", "keys": 1, "credits": 0, "runs": 0, "series": []},
]


# ---- bucketing (the pure half) -------------------------------------------------


def test_day_buckets_are_gap_filled_from_since_to_today():
    out = ur.bucket_series(SERIES, "mtd", "2026-09-01T00:00:00+00:00", "2026-09-06")
    assert [b["bucket"] for b in out] == [f"2026-09-0{d}" for d in range(1, 7)]
    assert [b["credits"] for b in out] == [0, 65, 0, 0, 30, 20]
    assert [b["label"] for b in out][:2] == ["Sep 1", "Sep 2"]
    # Product and partner actions fold into the same tool.
    assert out[1]["tools"] == {"oneclick": 60, "registry": 0, "splitsheet": 0, "zoe": 5}
    assert out[5]["tools"]["splitsheet"] == 20


def test_without_a_floor_only_days_with_spend_are_listed():
    out = ur.bucket_series(SERIES, "mtd", None, "2026-09-06")
    assert [b["bucket"] for b in out] == ["2026-09-02", "2026-09-05", "2026-09-06"]


def test_1y_buckets_start_on_monday_and_step_a_week():
    # 2026-09-02 is a Wednesday; its ISO week starts Monday 2026-08-31.
    assert ur.bucket_of("2026-09-02", "1y") == "2026-08-31"
    assert ur.bucket_of("2026-08-31", "1y") == "2026-08-31"
    assert ur.bucket_of("2026-09-06", "1y") == "2026-08-31"  # Sunday still that week
    out = ur.bucket_series(SERIES, "1y", "2026-08-20T00:00:00+00:00", "2026-09-10")
    assert [b["bucket"] for b in out] == ["2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07"]
    assert [b["credits"] for b in out] == [0, 0, 115, 0]  # every day above is in one week
    assert out[2]["label"] == "Aug 31"


def test_all_time_lists_only_months_with_spend():
    series = SERIES + [_day("2026-07-14", ("zoe_message", 5, 1))]
    out = ur.bucket_series(series, "all", "2026-01-01T00:00:00+00:00", "2026-09-06")
    assert [(b["bucket"], b["label"], b["credits"]) for b in out] == [
        ("2026-07-01", "Jul 2026", 5),
        ("2026-09-01", "Sep 2026", 115),
    ]


def test_unknown_actions_are_ignored_not_fatal():
    out = ur.bucket_series([_day("2026-09-02", ("something_new", 99, 1), ("zoe_message", 5, 1))], "mtd", None)
    assert out[0]["credits"] == 5 and sum(out[0]["tools"].values()) == 5


def test_merge_series_folds_per_key_series_into_one():
    merged = ur.merge_series(
        [
            [_day("2026-09-02", ("zoe_message", 5, 1))],
            [_day("2026-09-02", ("zoe_message", 5, 2)), _day("2026-09-03", ("oneclick_run", 30, 1))],
            None,
        ]
    )
    assert merged == [
        _day("2026-09-02", ("zoe_message", 10, 3)),
        _day("2026-09-03", ("oneclick_run", 30, 1)),
    ]


def test_window_and_filename_labels():
    today = date(2026, 9, 6)
    assert ur.window_label("mtd", "2026-09-01T00:00:00+00:00", today) == "Sep 1, 2026 – Sep 6, 2026"
    assert ur.window_label("all", "2026-09-01T00:00:00+00:00", today) == "All time"
    assert ur.window_label("7d", None, today) == "All time"
    assert ur.report_filename("Greenbox Records!", "mtd", today) == "usage-report-greenbox-records-mtd-2026-09-06.pdf"
    assert ur.report_filename("", "7d", today) == "usage-report-team-7d-2026-09-06.pdf"
    assert ur.report_filename("!!!", "7d", today) == "usage-report-team-7d-2026-09-06.pdf"
    assert len(ur.report_filename("x" * 80, "all", today).split("-all-")[0]) == len("usage-report-") + 40


# ---- rendering -----------------------------------------------------------------


def _render(**over):
    kwargs = {
        "title": "Greenbox Records",
        "range_": "mtd",
        "since": "2026-09-01T00:00:00+00:00",
        "series": SERIES,
        "seats": SEATS,
        "by_key": KEYS,
        "by_folder": FOLDERS,
        "generated_at": datetime(2026, 9, 6, 12, 0, tzinfo=UTC),
    }
    kwargs.update(over)
    return ur.render_usage_report(**kwargs)


def test_full_report_renders_every_section():
    pdf = _render()
    assert pdf[:4] == b"%PDF"
    for chunk in (b"Greenbox Records", b"Credits over time", b"By tool", b"By member", b"By key", b"By folder"):
        assert chunk in pdf
    assert b"Unknown" in pdf  # the seat with no email
    assert b"No credits used in this window." not in pdf


def test_empty_payload_still_renders():
    pdf = ur.render_usage_report(
        title="Quiet Team", range_="mtd", since=None, series=[], seats=None, by_key=[], by_folder=[]
    )
    assert pdf[:4] == b"%PDF"
    assert b"No credits used in this window." in pdf
    assert b"By member" not in pdf


def test_personal_variant_has_sections_and_no_member_table():
    sections = [
        {
            "heading": "Greenbox Records",
            "series": KEYS[0]["series"],
            "seats": None,
            "by_key": KEYS,
            "by_folder": FOLDERS,
        },
        {"heading": "Other Label", "series": [], "seats": None, "by_key": [], "by_folder": []},
    ]
    pdf = ur.render_usage_report(
        title="My API usage",
        range_="mtd",
        since="2026-09-01T00:00:00+00:00",
        series=ur.merge_series(s["series"] for s in sections),
        seats=None,
        by_key=[],
        by_folder=[],
        sections=sections,
    )
    assert pdf[:4] == b"%PDF"
    assert b"By member" not in pdf and b"Active members" not in pdf
    assert b"Other Label" in pdf
    # The quiet org gets the fallback line instead of a chart.
    assert b"No credits used in this window." in pdf


def test_long_windows_render():
    """400 days of spend through both the week and the month bucketers — the
    label thinning and the axis have to cope with ~57 weeks / 14 months."""
    start = date(2025, 8, 1)
    series = [
        _day((start + timedelta(days=i)).isoformat(), ("oneclick_run", 30 + i, 1))
        for i in range(400)
        if i % 3  # gaps, so the fill has something to do
    ]
    for range_ in ("1y", "all"):
        pdf = ur.render_usage_report(
            title="Busy Label",
            range_=range_,
            since="2025-08-01T00:00:00+00:00",
            series=series,
            seats=SEATS,
            by_key=KEYS,
            by_folder=FOLDERS,
        )
        assert pdf[:4] == b"%PDF" and len(pdf) > 3000


def test_member_columns_drop_tools_with_no_spend():
    """Column counting off the raw PDF is hopeless (the legend and the By tool
    table repeat every name), so check the table builder directly."""
    zoe_only = [
        {
            "email": "a@b.test",
            "role": "member",
            "spentThisPeriod": 5,
            "apiCredits": 0,
            "byAction": [{"action": "zoe_message", "credits": 5, "runs": 1}],
        }
    ]
    headers, rows, widths, right = ur._member_rows(zoe_only)
    assert headers == ["Member", "Role", "Credits", "Runs", "Zoe"]
    assert rows == [["a@b.test", "Member", "5", "1", "5"]]
    assert len(widths) == len(headers) and right == {2, 3, 4}

    # Sorted by total desc; total = product + API spend, email falls back.
    headers, rows, _, _ = ur._member_rows(SEATS)
    assert headers == ["Member", "Role", "Credits", "Runs", "OneClick", "Split sheet", "Zoe"]
    assert rows[0][:4] == ["ada@label.test", "Admin", "85", "4"]
    assert rows[1][:4] == ["Unknown", "Member", "0", "0"]

    # No spend anywhere: no tool columns at all, and the widths still add up.
    headers, _, widths, _ = ur._member_rows([{"email": "x@y.z", "role": "member"}])
    assert headers == ["Member", "Role", "Credits", "Runs"]
    assert sum(widths) == ur.CONTENT_WIDTH


# ---- routes ---------------------------------------------------------------------

PAYLOAD = {
    "range": "mtd",
    "since": "2026-09-01T00:00:00+00:00",
    "series": SERIES,
    "seats": SEATS,
    "byKey": KEYS,
    "byFolder": FOLDERS,
}


def _assert_pdf_download(resp, expect_slug):
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert f'attachment; filename="usage-report-{expect_slug}-' in resp.headers["content-disposition"]
    assert resp.content[:4] == b"%PDF"


@pytest.fixture
def licensing(monkeypatch):
    # /orgs/* 404s without the flag (orgs.router.require_licensing).
    monkeypatch.setenv("LICENSING_ENABLED", "true")


def test_org_report_route_returns_a_pdf(client, mock_supabase, licensing):
    from tests.conftest import MockQueryBuilder, _default_table_side_effect

    def _side(name):
        if name != "organizations":
            return _default_table_side_effect(name)
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=[{"name": "Greenbox Records"}], count=1)
        return b

    mock_supabase.table.side_effect = _side
    with patch("orgs.router.service.get_org_usage", new=AsyncMock(return_value=PAYLOAD)) as svc:
        resp = client.get(f"/orgs/{ORG}/usage/report.pdf?range=7d")
    _assert_pdf_download(resp, "greenbox-records")
    assert svc.call_args.kwargs["range_"] == "7d"


def test_org_report_route_rejects_an_unknown_range(client, licensing):
    with patch("orgs.router.service.get_org_usage", new=AsyncMock(return_value=PAYLOAD)) as svc:
        resp = client.get(f"/orgs/{ORG}/usage/report.pdf?range=30d")
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "invalid_range"
    svc.assert_not_called()


def test_org_report_route_reuses_the_admin_gate(client, licensing):
    """No second authz check here — whatever get_org_usage raises is the answer."""
    denied = AsyncMock(side_effect=HTTPException(status_code=403, detail="Admin access required"))
    with patch("orgs.router.service.get_org_usage", new=denied):
        assert client.get(f"/orgs/{ORG}/usage/report.pdf").status_code == 403


@pytest.fixture
def admin_client(client):
    import main
    from subscriptions.admin_auth import require_admin

    async def _pass():
        return "admin@example.com"

    main.app.dependency_overrides[require_admin] = _pass
    yield client
    main.app.dependency_overrides.pop(require_admin, None)


def test_admin_report_route_returns_a_pdf(admin_client, monkeypatch):
    async def fake_rollup(db, org_id, range_="mtd"):
        assert (org_id, range_) == (ORG, "1y")
        return {**PAYLOAD, "range": "1y"}

    monkeypatch.setattr("orgs.service.org_usage_rollup", fake_rollup)
    _assert_pdf_download(admin_client.get(f"/admin/orgs/{ORG}/usage/report.pdf?range=1y"), "team")


def test_admin_report_route_rejects_an_unknown_range(admin_client, monkeypatch):
    called = MagicMock()
    monkeypatch.setattr("orgs.service.org_usage_rollup", called)
    resp = admin_client.get(f"/admin/orgs/{ORG}/usage/report.pdf?range=30d")
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "invalid_range"
    called.assert_not_called()


def test_admin_report_route_requires_a_msanii_admin(client):
    assert client.get(f"/admin/orgs/{ORG}/usage/report.pdf").status_code in (401, 403)


def test_my_report_route_renders_one_section_per_org(client, monkeypatch):
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    payload = {
        "range": "mtd",
        "orgs": [
            {
                "orgId": ORG,
                "orgName": "Greenbox Records",
                "since": "2026-09-01T00:00:00+00:00",
                "credits": 20,
                "runs": 1,
                "byKey": KEYS,
                "byFolder": FOLDERS,
            }
        ],
    }
    with patch("subscriptions.router._my_api_usage", new=AsyncMock(return_value=payload)):
        resp = client.get("/me/api-usage/report.pdf")
    _assert_pdf_download(resp, "my-api-usage")
    assert b"Greenbox Records" in resp.content and b"By member" not in resp.content


def test_my_report_route_is_still_a_pdf_with_the_flags_off(client, monkeypatch):
    """Same gate result as the JSON's empty `orgs` — an empty report, not a 404."""
    monkeypatch.delenv("CREDITS_ENABLED", raising=False)
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)
    with patch("orgs.service.org_usage_rollup") as rollup:
        resp = client.get("/me/api-usage/report.pdf")
    _assert_pdf_download(resp, "my-api-usage")
    rollup.assert_not_called()
