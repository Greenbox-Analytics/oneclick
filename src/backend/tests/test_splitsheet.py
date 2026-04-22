"""Tests for split sheet generation endpoint.

Covers:
- POST /splitsheet/generate  returns PDF StreamingResponse
- POST /splitsheet/generate  returns DOCX StreamingResponse
- POST /splitsheet/generate  returns 400 when no contributors
- POST /splitsheet/generate  returns 422 on invalid payload
"""

CONTRIBUTOR = {
    "name": "Jane Doe",
    "role": "Songwriter",
    "publishing_percentage": 50.0,
    "master_percentage": 50.0,
    "publisher_or_label": "Indie Label",
    "ipi_number": "00123456789",
}

BASE_PAYLOAD = {
    "work_title": "Test Track",
    "work_type": "single",
    "split_type": "both",
    "date": "2026-04-10",
    "format": "pdf",
    "contributors": [CONTRIBUTOR],
}


class TestGenerateSplitSheetPDF:
    """POST /splitsheet/generate with format=pdf produces a valid PDF."""

    def test_returns_200_with_pdf_content_type(self, client, mock_supabase):
        """Successful PDF generation returns 200 and correct content-type."""
        response = client.post("/splitsheet/generate", json=BASE_PAYLOAD)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"

    def test_response_body_is_non_empty(self, client, mock_supabase):
        """PDF body must be non-empty bytes."""
        response = client.post("/splitsheet/generate", json=BASE_PAYLOAD)

        assert response.status_code == 200
        assert len(response.content) > 0

    def test_content_disposition_uses_safe_filename(self, client, mock_supabase):
        """Content-Disposition header contains the sanitised work title and .pdf ext."""
        response = client.post("/splitsheet/generate", json=BASE_PAYLOAD)

        assert response.status_code == 200
        cd = response.headers.get("content-disposition", "")
        assert "Split_Sheet_" in cd
        assert ".pdf" in cd

    def test_multiple_contributors(self, client, mock_supabase):
        """Works with multiple contributors."""
        contributor_2 = {
            "name": "John Smith",
            "role": "Producer",
            "publishing_percentage": 50.0,
            "master_percentage": 50.0,
        }
        payload = {**BASE_PAYLOAD, "contributors": [CONTRIBUTOR, contributor_2]}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 200
        assert len(response.content) > 0

    def test_special_characters_in_title_are_sanitised(self, client, mock_supabase):
        """Special characters in work_title are replaced with underscores in filename."""
        payload = {**BASE_PAYLOAD, "work_title": "My Song: Version #2 (Remix)"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 200
        cd = response.headers.get("content-disposition", "")
        # Colons, hashes, parens and spaces should be sanitised
        assert ":" not in cd
        assert "#" not in cd

    def test_minimal_contributor_fields(self, client, mock_supabase):
        """Contributor with only required fields (name, role) succeeds."""
        minimal_contributor = {"name": "Solo Artist", "role": "Composer"}
        payload = {**BASE_PAYLOAD, "contributors": [minimal_contributor]}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 200


class TestGenerateSplitSheetDOCX:
    """POST /splitsheet/generate with format=docx produces a valid DOCX."""

    def test_returns_200_with_docx_content_type(self, client, mock_supabase):
        """Successful DOCX generation returns 200 and correct content-type."""
        payload = {**BASE_PAYLOAD, "format": "docx"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 200
        assert (
            response.headers["content-type"]
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_response_body_is_non_empty(self, client, mock_supabase):
        """DOCX body must be non-empty bytes."""
        payload = {**BASE_PAYLOAD, "format": "docx"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 200
        assert len(response.content) > 0

    def test_content_disposition_has_docx_extension(self, client, mock_supabase):
        """Content-Disposition header contains .docx extension."""
        payload = {**BASE_PAYLOAD, "format": "docx"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 200
        cd = response.headers.get("content-disposition", "")
        assert ".docx" in cd


class TestGenerateSplitSheetValidation:
    """Input validation for POST /splitsheet/generate."""

    def test_returns_400_when_contributors_empty(self, client, mock_supabase):
        """Returns 400 when contributors list is empty."""
        payload = {**BASE_PAYLOAD, "contributors": []}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 400
        assert "contributor" in response.json()["detail"].lower()

    def test_returns_422_when_contributors_missing(self, client, mock_supabase):
        """Returns 422 when contributors field is absent."""
        payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "contributors"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 422

    def test_returns_422_when_work_title_missing(self, client, mock_supabase):
        """Returns 422 when work_title is absent."""
        payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "work_title"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 422

    def test_returns_422_when_date_missing(self, client, mock_supabase):
        """Returns 422 when date is absent."""
        payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "date"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 422

    def test_defaults_format_to_pdf_when_not_specified(self, client, mock_supabase):
        """Omitting format defaults to PDF generation."""
        payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "format"}

        response = client.post("/splitsheet/generate", json=payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
