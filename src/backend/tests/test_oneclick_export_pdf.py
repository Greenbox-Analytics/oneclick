"""POST /oneclick/export-pdf — returns the royalty-results PDF as a download.

The endpoint reuses /share's _generate_pdf; these tests pin the download
contract (status, content type, attachment disposition, real PDF bytes).
"""

PAYLOAD = {
    "artist_name": "Artist",
    "message": "Successfully calculated 2 royalty payments",
    "payments": [
        {
            "song_title": "Home",
            "party_name": "Alice",
            "role": "Artist",
            "royalty_type": "Streaming",
            "percentage": 50.0,
            "total_royalty": 13.94,
            "amount_to_pay": 6.97,
            "basis": "gross",
            "gross_amount": 13.94,
            "expenses_applied": 0.0,
            "net_amount": 13.94,
        },
        {
            "song_title": "Home",
            "party_name": "Bob",
            "role": "Producer",
            "royalty_type": "Streaming",
            "percentage": 5.0,
            "total_royalty": 13.94,
            "amount_to_pay": 0.7,
            "basis": "net",
            "gross_amount": 13.94,
            "expenses_applied": 2.0,
            "net_amount": 11.94,
        },
    ],
    "total_payments": 7.67,
}


def test_export_pdf_returns_pdf_download(client):
    resp = client.post("/oneclick/export-pdf", json=PAYLOAD)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert resp.content.startswith(b"%PDF")


def test_export_pdf_with_empty_payments_still_builds(client):
    resp = client.post("/oneclick/export-pdf", json={**PAYLOAD, "payments": []})
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF")


def test_export_pdf_without_message_still_builds(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "message"}
    resp = client.post("/oneclick/export-pdf", json=payload)
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF")


def test_export_pdf_handles_sparse_legacy_rows(client):
    # Older cached results lack basis/gross/net fields — generator must not raise.
    sparse = {
        "artist_name": "Artist",
        "payments": [
            {
                "song_title": "Old Song",
                "party_name": "Carol",
                "role": "Writer",
                "royalty_type": "publishing",
                "percentage": 25.0,
                "amount_to_pay": 2.5,
            }
        ],
        "total_payments": 2.5,
    }
    resp = client.post("/oneclick/export-pdf", json=sparse)
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF")
