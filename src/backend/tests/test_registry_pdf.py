"""Tests for the registry work-metadata PDF export.

Focus: the Signatures section — every unique stakeholder gets a signature
block, deduplicated across stake types.
"""

import fitz

from registry.pdf_generator import generate_proof_of_ownership_pdf


def _pdf_text(work_data: dict) -> str:
    buffer = generate_proof_of_ownership_pdf(work_data)
    with fitz.open(stream=buffer.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _base_work(stakes: list[dict]) -> dict:
    return {
        "id": "work-1",
        "title": "Test Song",
        "work_type": "single",
        "status": "pending_approval",
        "stakes": stakes,
        "collaborators": [],
        "licenses": [],
        "agreements": [],
    }


class TestSignatureSection:
    def test_signature_section_present_with_signer_blocks(self):
        text = _pdf_text(
            _base_work(
                [
                    {"holder_name": "Alice", "holder_role": "Artist", "stake_type": "master", "percentage": 60.0},
                    {"holder_name": "Bob", "holder_role": "Producer", "stake_type": "master", "percentage": 40.0},
                ]
            )
        )
        assert "Signatures" in text
        assert "By signing below" in text
        assert "Name: Alice" in text
        assert "Name: Bob" in text
        assert text.count("Signature:") == 2
        assert text.count("Date: ____") == 2

    def test_same_holder_across_stake_types_signs_once_with_merged_roles(self):
        text = _pdf_text(
            _base_work(
                [
                    {
                        "holder_name": "Alice",
                        "holder_email": "alice@example.com",
                        "holder_role": "Artist",
                        "stake_type": "master",
                        "percentage": 100.0,
                    },
                    {
                        "holder_name": "Alice",
                        "holder_email": "Alice@Example.com",
                        "holder_role": "Songwriter",
                        "stake_type": "publishing",
                        "percentage": 100.0,
                    },
                ]
            )
        )
        assert text.count("Name: Alice") == 1
        assert text.count("Signature:") == 1
        assert "Artist, Songwriter" in text

    def test_no_stakeholders_shows_placeholder_instead_of_blocks(self):
        text = _pdf_text(_base_work([]))
        assert "Signatures" in text
        assert "No stakeholders recorded for this work." in text
        assert "Signature:" not in text
