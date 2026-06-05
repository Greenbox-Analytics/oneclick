"""Tests for offset-based, section-aware book chunking."""

import tiktoken

from knowledge.chunking import (
    PageText,
    _page_at,
    _sections_with_offsets,
    section_aware_chunks,
)


def test_page_at_maps_offsets():
    boundaries = [(0, 1), (100, 2), (250, 3)]
    assert _page_at(boundaries, 0) == 1
    assert _page_at(boundaries, 99) == 1
    assert _page_at(boundaries, 100) == 2
    assert _page_at(boundaries, 249) == 2
    assert _page_at(boundaries, 300) == 3


def test_sections_with_offsets_breadcrumb_and_contiguous():
    text = "# Part V\n## Record Deals\nbody one\n### Royalties\nbody two\n"
    sections = _sections_with_offsets(text)
    paths = [p for p, _, _ in sections]
    assert "Part V ▸ Record Deals" in paths
    assert "Part V ▸ Record Deals ▸ Royalties" in paths
    # sections tile the text with no gaps
    for (_, _, end), (_, start, _) in zip(sections, sections[1:], strict=False):
        assert end == start


def test_chunk_starting_on_page_keeps_start_page_across_boundary():
    # Page 1 long enough to force a split; pages end in \n like real pymupdf4llm output.
    pages = [
        PageText(1, "# Royalties\n" + "Royalty detail sentence. " * 40 + "\n"),
        PageText(2, "Continued royalty detail on the next page. " * 5 + "\n"),
    ]
    chunks = section_aware_chunks(pages, source="passman", book_title="X", chunk_tokens=30, overlap_tokens=0)
    assert chunks[0].page_start == 1
    assert all(c.page_start <= c.page_end for c in chunks)
    assert max(c.page_end for c in chunks) == 2


def test_top_of_page_heading_is_detected_and_attributed():
    # Page 2 begins with a heading. With the newline separator the heading is genuinely
    # detected (not glued mid-line), and the page-1 content stays page 1.
    pages = [
        PageText(1, "Plain page one content about masters.\n"),
        PageText(2, "# New Chapter\nFresh content about publishing.\n"),
    ]
    chunks = section_aware_chunks(pages, source="passman", book_title="X", chunk_tokens=40, overlap_tokens=0)
    page_one = [c for c in chunks if "masters" in c.text]
    assert page_one and all(c.page_start == 1 and c.page_end == 1 for c in page_one)
    page_two = [c for c in chunks if "publishing" in c.text]
    assert page_two and all(c.page_start == 2 for c in page_two)
    assert any("New Chapter" in c.section_path for c in page_two)  # heading genuinely detected


def test_heading_only_section_is_dropped():
    pages = [PageText(1, "# Part V\n## Record Deals\nRoyalties are paid on net.\n")]
    chunks = section_aware_chunks(pages, source="passman", book_title="X")
    assert all(c.text.strip() not in {"# Part V", "## Record Deals"} for c in chunks)
    assert any("Part V ▸ Record Deals" in c.section_path for c in chunks)


def test_section_path_strips_markdown_emphasis():
    # pymupdf4llm keeps **bold** / _italic_ markers on headings; section_path must be clean.
    pages = [PageText(1, "# **Songwriting**\n## **MORE ABOUT MECHANICAL ROYALTIES**\nbody text here.\n")]
    chunks = section_aware_chunks(pages, source="passman", book_title="X")
    joined = " | ".join(c.section_path for c in chunks)
    assert "*" not in joined and "_" not in joined
    assert any("MORE ABOUT MECHANICAL ROYALTIES" in c.section_path for c in chunks)


def test_page_offset_applied():
    pages = [PageText(15, "# Intro\nPrinted page one content.")]
    chunks = section_aware_chunks(pages, source="passman", book_title="X", page_offset=14)
    assert chunks[0].page_start == 1  # 15 - 14
    assert all(c.page_start >= 1 for c in chunks)


def test_no_empty_chunks_and_token_count():
    enc = tiktoken.get_encoding("cl100k_base")
    pages = [PageText(1, "# A\nSome content here.\n\n   \n\n## B\nMore content.")]
    chunks = section_aware_chunks(pages, source="passman", book_title="X")
    assert chunks
    for c in chunks:
        assert c.text.strip() != ""
        assert c.token_count == len(enc.encode(c.text))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_pages_out_of_order_are_handled():
    # Caller passes pages out of order; output must still attribute correctly.
    pages = [
        PageText(2, "Second page about publishing.\n"),
        PageText(1, "First page about masters.\n"),
    ]
    chunks = section_aware_chunks(pages, source="passman", book_title="X", chunk_tokens=6, overlap_tokens=0)
    masters = [c for c in chunks if "masters" in c.text]
    publishing = [c for c in chunks if "publishing" in c.text]
    assert masters and all(c.page_start == 1 for c in masters)
    assert publishing and all(c.page_start == 2 for c in publishing)
