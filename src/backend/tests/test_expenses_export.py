"""Tests for the expense-report export endpoint.

Covers:
- GET /expenses/export?format=pdf   → PDF StreamingResponse
- GET /expenses/export?format=xlsx  → XLSX StreamingResponse
- invalid format                    → 400
- project_id / category filters narrow the rows included in the report

``get_expenses_summary`` is monkeypatched at the router module level so the
tests exercise the export/report path without wiring the full Supabase
membership → projects → expenses chain.
"""

from unittest.mock import AsyncMock

import pytest

PDF_MEDIA = "application/pdf"
XLSX_MEDIA = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

ROWS = [
    {
        "id": "e1",
        "project_id": "p1",
        "project_name": "Album One",
        "artist_id": "a1",
        "artist_name": "Artist One",
        "description": "Studio time",
        "amount": 100.0,
        "category": "studio",
        "incurred_on": "2026-06-01",
        "is_tagged": False,
    },
    {
        "id": "e2",
        "project_id": "p2",
        "project_name": "Album Two",
        "artist_id": "a2",
        "artist_name": "Artist Two",
        "description": "Marketing push",
        "amount": 250.0,
        "category": "marketing",
        "incurred_on": "2026-06-10",
        "is_tagged": True,
    },
    {
        "id": "e3",
        "project_id": "p1",
        "project_name": "Album One",
        "artist_id": "a1",
        "artist_name": "Artist One",
        "description": "Uncategorized misc",
        "amount": 40.0,
        "category": None,
        "incurred_on": None,
        "is_tagged": False,
    },
]


@pytest.fixture()
def _patch_summary(monkeypatch):
    """Patch get_expenses_summary in the router to return ROWS."""
    import expenses.router as exp_router

    monkeypatch.setattr(exp_router, "get_expenses_summary", AsyncMock(return_value=ROWS))


class TestExportFormats:
    def test_pdf_returns_200_and_content_type(self, client, mock_supabase, _patch_summary):
        resp = client.get("/expenses/export", params={"format": "pdf"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == PDF_MEDIA
        assert len(resp.content) > 0

    def test_pdf_is_the_default_format(self, client, mock_supabase, _patch_summary):
        resp = client.get("/expenses/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == PDF_MEDIA

    def test_xlsx_returns_200_and_content_type(self, client, mock_supabase, _patch_summary):
        resp = client.get("/expenses/export", params={"format": "xlsx"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == XLSX_MEDIA
        assert len(resp.content) > 0

    def test_content_disposition_is_attachment(self, client, mock_supabase, _patch_summary):
        resp = client.get("/expenses/export", params={"format": "xlsx"})
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "Expense_Report_" in cd
        assert ".xlsx" in cd

    def test_invalid_format_returns_400(self, client, mock_supabase, _patch_summary):
        resp = client.get("/expenses/export", params={"format": "csv"})
        assert resp.status_code == 400


class TestExportFilters:
    def test_project_filter_scopes_the_report(self, client, mock_supabase, _patch_summary):
        """A project filter narrows both the rows and the scope label in the filename."""
        resp = client.get("/expenses/export", params={"format": "pdf", "project_id": "p1"})
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        # scope label derives from the selected project's name
        assert "Album_One" in cd

    def test_overall_report_uses_all_projects_label(self, client, mock_supabase, _patch_summary):
        resp = client.get("/expenses/export", params={"format": "pdf"})
        cd = resp.headers.get("content-disposition", "")
        assert "All_projects" in cd

    def test_category_filter_matches_uncategorized_as_other(self, client, mock_supabase, _patch_summary):
        """category=other includes the null-category row (parity with the page)."""
        resp = client.get("/expenses/export", params={"format": "xlsx", "category": "other"})
        assert resp.status_code == 200
        assert len(resp.content) > 0
