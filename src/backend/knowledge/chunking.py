"""Offset-based, section-aware chunking for long-form reference PDFs.

Page attribution is computed from character offsets (not in-text sentinels), so it
is correct at page boundaries and when a page begins with a heading. Pure and
unit-testable — PDF extraction lives in `extract_pages`, kept thin and separate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

TOKENIZER = "cl100k_base"
_HEADER = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
# Markdown emphasis markers pymupdf4llm keeps on headings (**bold**, _italic_, `code`).
_EMPHASIS = re.compile(r"\*{1,3}|_{1,3}|`+")


def _clean_heading(title: str) -> str:
    """Strip markdown emphasis from a heading so section_path breadcrumbs read cleanly."""
    return _EMPHASIS.sub("", title).strip()


@dataclass
class PageText:
    """One page of extracted markdown. `page_number` is the 1-based PDF position."""

    page_number: int
    markdown: str


@dataclass
class Chunk:
    text: str
    section_path: str
    page_start: int
    page_end: int
    chunk_index: int
    token_count: int


def extract_pages(pdf_path: str) -> list[PageText]:
    """Extract per-page markdown using pymupdf4llm.

    `page_chunks=True` returns one dict per page in order; we number pages by
    position (1-based) so the result is deterministic.
    """
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    out: list[PageText] = []
    for i, page in enumerate(pages):
        if not isinstance(page, dict):
            raise TypeError(f"Unexpected pymupdf4llm page type: {type(page)}")
        md = page.get("text", "")
        out.append(PageText(page_number=i + 1, markdown=md))
    return out


def _build_combined(pages: list[PageText]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate page markdown and record (start_char, page_number) boundaries.

    A newline is inserted between pages that don't already end in one, so a heading
    at the top of a page lands at the start of a line and is detected as a header
    (not glued mid-line onto the previous page's last line).
    """
    parts: list[str] = []
    boundaries: list[tuple[int, int]] = []
    offset = 0
    for p in pages:
        boundaries.append((offset, p.page_number))
        parts.append(p.markdown)
        offset += len(p.markdown)
        if not p.markdown.endswith("\n"):
            parts.append("\n")
            offset += 1
    return "".join(parts), boundaries


def _page_at(boundaries: list[tuple[int, int]], char_offset: int) -> int:
    """Page number in effect at `char_offset` (last boundary whose start <= offset)."""
    page = boundaries[0][1]
    for start, page_no in boundaries:
        if start <= char_offset:
            page = page_no
        else:
            break
    return page


def _sections_with_offsets(combined: str) -> list[tuple[str, int, int]]:
    """Split into (section_path, start_offset, end_offset) by scanning ATX headers.

    A header line begins its own section (so the heading text travels with the
    content beneath it). Text before the first header is one path-less section.
    """
    results: list[tuple[str, int, int]] = []
    stack: dict[int, str] = {}
    cur_path = ""
    cur_start = 0
    offset = 0
    for line in combined.splitlines(keepends=True):
        m = _HEADER.match(line)
        if m:
            if offset > cur_start:
                results.append((cur_path, cur_start, offset))
            level = len(m.group(1))
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = _clean_heading(m.group(2))
            cur_path = " ▸ ".join(stack[k] for k in sorted(stack))
            cur_start = offset
        offset += len(line)
    if offset > cur_start:
        results.append((cur_path, cur_start, offset))
    return results


def _is_heading_only(text: str) -> bool:
    """True if the text is nothing but heading line(s) — drop these (the breadcrumb
    is preserved on the child sections that carry actual content)."""
    body = "\n".join(line for line in text.splitlines() if not _HEADER.match(line))
    return not body.strip()


def section_aware_chunks(
    pages: list[PageText],
    source: str,
    book_title: str,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
    page_offset: int = 0,
) -> list[Chunk]:
    """Split per-page markdown into section-bounded, token-sized chunks.

    Page ranges are derived from global character offsets, so they stay correct at
    boundaries. `page_offset` converts PDF position to printed page (PDF page 15 with
    offset 14 -> printed page 1), clamped to >= 1.
    """
    if not pages:
        return []

    pages = sorted(pages, key=lambda p: p.page_number)
    combined, boundaries = _build_combined(pages)
    sections = _sections_with_offsets(combined)
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER,
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        add_start_index=True,
    )
    enc = tiktoken.get_encoding(TOKENIZER)

    chunks: list[Chunk] = []
    idx = 0
    for section_path, sec_start, sec_end in sections:
        section_text = combined[sec_start:sec_end]
        for doc in token_splitter.create_documents([section_text]):
            piece = doc.page_content
            clean = piece.strip()
            if not clean or _is_heading_only(clean):
                continue
            lead = len(piece) - len(piece.lstrip())
            clean_start = sec_start + doc.metadata["start_index"] + lead
            clean_end = clean_start + len(clean)
            pdf_start = _page_at(boundaries, clean_start)
            pdf_end = _page_at(boundaries, max(clean_start, clean_end - 1))
            chunks.append(
                Chunk(
                    text=clean,
                    section_path=section_path,
                    page_start=max(1, pdf_start - page_offset),
                    page_end=max(1, pdf_end - page_offset),
                    chunk_index=idx,
                    token_count=len(enc.encode(clean)),
                )
            )
            idx += 1
    return chunks
