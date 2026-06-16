"""Section detection and categorization for contract markdown."""

import re

# Section category keywords for classification
SECTION_CATEGORIES = {
    "ROYALTY_CALCULATIONS": ["royalty", "compensation", "revenue share", "splits", "payment", "percentage", "master"],
    "PUBLISHING_RIGHTS": ["publishing", "songwriter", "composition"],
    "PERFORMANCE_RIGHTS": ["performance", "production services", "live"],
    "COPYRIGHT": ["copyright", "intellectual property"],
    "TERMINATION": ["termination", "term", "duration", "end date"],
    "MASTER_RIGHTS": ["master rights", "master recording"],
    "OWNERSHIP_RIGHTS": ["ownership", "synchronization", "licensing"],
    "ACCOUNTING_AND_CREDIT": ["accounting", "credit", "promotion", "audit"],
}


def split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """
    Split markdown text into sections using a 4-layer approach:
    1) Explicit headings (markdown # or numbered sections)
    2) Heuristic heading candidates (caps, bold, short lines)
    3) Semantic heading validation
    4) Content-based section inference

    Args:
        markdown_text: The markdown content

    Returns:
        List of (header, content) tuples
    """
    lines = markdown_text.splitlines()
    sections: list[tuple[str, str]] = []

    # Layer 1: Explicit Headings (markdown # or numbered like "1. DEFINITIONS")
    explicit_heading_re = re.compile(r"^(#{1,6}\s+.+|\d+(\.\d+)*[\)\.]?\s+[A-Z].+)$")

    explicit_headers = [i for i, line in enumerate(lines) if explicit_heading_re.match(line.strip())]

    if explicit_headers:
        for idx, start in enumerate(explicit_headers):
            end = explicit_headers[idx + 1] if idx + 1 < len(explicit_headers) else len(lines)
            header = lines[start].strip()
            content = "\n".join(lines[start + 1 : end]).strip()
            sections.append((header, content))
        return sections

    # Layer 2: Heuristic Header Detection (ALL CAPS, bold markdown, short title-like)
    def is_heuristic_header(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # ALL CAPS and short
        if stripped.isupper() and len(stripped.split()) <= 10:
            return True
        # Bold markdown
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped.split()) <= 12:
            return True
        # Short title-like line ending with colon
        if len(stripped.split()) <= 6 and stripped.endswith(":"):
            return True
        return False

    header_indices = [i for i, line in enumerate(lines) if is_heuristic_header(line)]

    if header_indices:
        for idx, start in enumerate(header_indices):
            end = header_indices[idx + 1] if idx + 1 < len(header_indices) else len(lines)
            header = lines[start].strip().strip("*").strip(":")
            content = "\n".join(lines[start + 1 : end]).strip()
            sections.append((header, content))
        return sections

    # Layer 3: Semantic Heading Detection (short, prominent lines validated by context)
    semantic_candidates = [
        (i, line.strip()) for i, line in enumerate(lines) if 2 <= len(line.split()) <= 8 and line.strip()
    ]

    semantic_headers = []
    for idx, text in semantic_candidates:
        if is_semantic_heading(text):
            semantic_headers.append(idx)

    if semantic_headers:
        for i, start in enumerate(semantic_headers):
            end = semantic_headers[i + 1] if i + 1 < len(semantic_headers) else len(lines)
            header = lines[start].strip()
            content = "\n".join(lines[start + 1 : end]).strip()
            sections.append((header, content))
        return sections

    # Layer 4: Content-Based Inference (detect royalty/payment sections)
    royalty_trigger_re = re.compile(r"(royalt|revenue|shall receive|net revenue|%)", re.IGNORECASE)

    inferred_starts = [i for i, line in enumerate(lines) if royalty_trigger_re.search(line)]

    if inferred_starts:
        start = inferred_starts[0]
        sections.append(("Inferred Royalty Section", "\n".join(lines[start:]).strip()))
        return sections

    # Fallback: Return entire document as one section
    return [("Full Document", markdown_text.strip())]


def is_semantic_heading(text: str) -> bool:
    """
    Determine if a text is likely a section heading using simple heuristics.

    Args:
        text: The text to evaluate

    Returns:
        True if text appears to be a heading
    """
    stripped = text.strip()

    # Numbered section patterns
    if re.match(r"^\d+\.?\s+[A-Z]", stripped):
        return True

    # All caps short line
    if stripped.isupper() and len(stripped.split()) <= 6:
        return True

    # Title case and short
    if stripped.istitle() and len(stripped.split()) <= 5:
        return True

    # Contains common section header words
    header_keywords = ["article", "section", "clause", "schedule", "exhibit", "appendix"]
    if any(kw in stripped.lower() for kw in header_keywords):
        return True

    return False


def categorize_section(section_header: str) -> str:
    """
    Categorize a section based on its header using keyword matching.

    Args:
        section_header: The header text of the section

    Returns:
        Category string (e.g., "ROYALTY_CALCULATIONS", "PUBLISHING_RIGHTS")
    """
    header_lower = section_header.lower()

    for category, keywords in SECTION_CATEGORIES.items():
        if any(keyword in header_lower for keyword in keywords):
            return category

    return "OTHER"
