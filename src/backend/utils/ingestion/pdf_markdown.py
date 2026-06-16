"""PDF→markdown conversion with `[[PAGE n]]` navigation markers.

The markers let Zoe attribute an answer to a specific page so the viewer can jump
to it. They are hidden from the answer model and from OneClick (see
`strip_page_markers`) — they must never reach user-facing prose.
"""

import re

import pymupdf4llm

# Matches a page marker plus any blank lines immediately around it.
_PAGE_MARKER_RE = re.compile(r"\n*\[\[PAGE \d+\]\]\n*")


def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF to markdown, injecting a `[[PAGE n]]` navigation marker
    (1-based) at the start of each page. Markers let Zoe attribute an answer to a
    page so the viewer can jump to it. They are hidden from the answer model and
    from OneClick (see strip_page_markers) and must never reach user-facing prose.
    """
    print(f"Converting PDF to markdown: {pdf_path}")
    # page_chunks=True -> one dict per page, in page order. NOTE: metadata["page"]
    # is None in pymupdf4llm 1.27.2.2, so page number = list index + 1.
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    parts = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        parts.append(f"\n\n[[PAGE {i + 1}]]\n\n{text}")
    md_text = "".join(parts).strip()
    print(f"Converted to {len(md_text)} characters of markdown ({len(chunks)} pages)")
    return md_text


def markdown_has_page_markers(markdown: str | None) -> bool:
    """True if `markdown` contains at least one `[[PAGE n]]` marker."""
    if not markdown:
        return False
    return bool(re.search(r"\[\[PAGE \d+\]\]", markdown))


def strip_page_markers(markdown: str | None) -> str:
    """Remove `[[PAGE n]]` markers (and the blank lines around them).

    Used to hide markers from consumers that must not see them: the Zoe answer
    model (leak-proofing) and all OneClick contract parsing.
    """
    if not markdown:
        return ""
    return _PAGE_MARKER_RE.sub("\n\n", markdown).strip()
