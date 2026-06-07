"""Tests for page-marker injection / detection / stripping in PDF->markdown."""

import os
import tempfile

import fitz  # PyMuPDF, already a dependency

from zoe_chatbot.helpers import (
    markdown_has_page_markers,
    pdf_to_markdown,
    strip_page_markers,
)


def _make_pdf(pages: list[str]) -> str:
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body)
    path = os.path.join(tempfile.gettempdir(), "_pagemarker_test.pdf")
    doc.save(path)
    doc.close()
    return path


def test_pdf_to_markdown_injects_one_marker_per_page():
    path = _make_pdf(["Royalties are paid on net receipts.", "Either party may terminate."])
    try:
        md = pdf_to_markdown(path)
    finally:
        os.remove(path)
    assert "[[PAGE 1]]" in md
    assert "[[PAGE 2]]" in md
    assert "[[PAGE 3]]" not in md
    assert md.count("[[PAGE ") == 2
    assert md.index("[[PAGE 1]]") < md.index("[[PAGE 2]]")


def test_markdown_has_page_markers():
    assert markdown_has_page_markers("intro\n\n[[PAGE 1]]\n\nbody") is True
    assert markdown_has_page_markers("legacy markdown, no markers") is False
    assert markdown_has_page_markers("") is False
    assert markdown_has_page_markers(None) is False


def test_strip_page_markers():
    md = "intro\n\n[[PAGE 1]]\n\nbody one\n\n[[PAGE 2]]\n\nbody two"
    out = strip_page_markers(md)
    assert "[[PAGE" not in out
    assert "body one" in out and "body two" in out
    assert strip_page_markers("") == ""
    assert strip_page_markers(None) == ""
