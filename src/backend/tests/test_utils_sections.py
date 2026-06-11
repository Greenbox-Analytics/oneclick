"""Tests for section detection and categorization in utils/ingestion/sections.py.

These helpers had zero direct coverage before the refactor — they were exercised
only indirectly through the Zoe ingestion pipeline. Pin behavior now so future
edits to the 4-layer detector don't silently regress.
"""

from utils.ingestion.sections import (
    SECTION_CATEGORIES,
    categorize_section,
    is_semantic_heading,
    split_into_sections,
)

# ─── is_semantic_heading ─────────────────────────────────────────────────────


def test_is_semantic_heading_numbered_section():
    assert is_semantic_heading("1. DEFINITIONS") is True


def test_is_semantic_heading_all_caps_short():
    assert is_semantic_heading("ROYALTIES") is True


def test_is_semantic_heading_long_paragraph_false():
    # Long, lowercase prose with none of the header keywords (article/section/
    # clause/schedule/exhibit/appendix). Must NOT be flagged as a heading.
    long = (
        "the artist hereby grants to the company a non-exclusive perpetual right "
        "to use the recordings on any media now known or hereafter devised"
    )
    assert is_semantic_heading(long) is False


def test_is_semantic_heading_keyword_match_schedule():
    assert is_semantic_heading("Schedule A") is True


def test_is_semantic_heading_keyword_match_article():
    assert is_semantic_heading("Article 3") is True


def test_is_semantic_heading_plain_lowercase_prose_false():
    assert is_semantic_heading("the parties hereby agree") is False


# ─── categorize_section ──────────────────────────────────────────────────────


def test_categorize_section_routes_royalty_keywords():
    assert categorize_section("Royalty Calculations") == "ROYALTY_CALCULATIONS"


def test_categorize_section_routes_publishing_keywords():
    assert categorize_section("Publishing Rights") == "PUBLISHING_RIGHTS"


def test_categorize_section_unknown_returns_other():
    assert categorize_section("Force Majeure") == "OTHER"


def test_categorize_section_is_case_insensitive():
    assert categorize_section("TERMINATION CLAUSE") == "TERMINATION"


def test_section_categories_is_a_dict_of_lists():
    """Lightweight type-shape guard so consumers can rely on it."""
    assert isinstance(SECTION_CATEGORIES, dict)
    for key, val in SECTION_CATEGORIES.items():
        assert isinstance(key, str)
        assert isinstance(val, list)
        assert all(isinstance(k, str) for k in val)


# ─── split_into_sections — 4-layer detection + fallback ──────────────────────


def test_split_layer1_markdown_headings():
    md = "# Intro\nbody one\n# Royalties\nbody two"
    sections = split_into_sections(md)
    headers = [h for h, _ in sections]
    assert headers == ["# Intro", "# Royalties"]
    assert sections[0][1] == "body one"
    assert sections[1][1] == "body two"


def test_split_layer1_numbered_sections():
    md = "1. INTRODUCTION\nbody one\n2. ROYALTIES\nbody two"
    sections = split_into_sections(md)
    assert len(sections) == 2
    assert sections[0][0] == "1. INTRODUCTION"
    assert sections[1][0] == "2. ROYALTIES"


def test_split_layer2_all_caps_headers():
    """No Layer 1 hits (no #, no numbered uppercase). ALL CAPS short lines
    become Layer 2 headers."""
    md = "ROYALTIES\nbody one\nTERMINATION\nbody two"
    sections = split_into_sections(md)
    assert len(sections) == 2
    assert sections[0][0] == "ROYALTIES"
    assert sections[1][0] == "TERMINATION"


def test_split_layer4_royalty_keyword_fallback():
    """No Layer 1/2/3 hits, but content mentions a royalty trigger →
    single 'Inferred Royalty Section'."""
    md = "the artist shall receive 50% of net revenue paid quarterly to the lender per the terms herein"
    sections = split_into_sections(md)
    assert len(sections) == 1
    assert sections[0][0] == "Inferred Royalty Section"


def test_split_layer5_full_document_fallback():
    md = "this is just some plain prose lacking any structure or trigger words at all today"
    sections = split_into_sections(md)
    assert sections == [("Full Document", md)]


def test_split_handles_empty_input_with_full_document_fallback():
    """Defensive: empty string still produces one section."""
    sections = split_into_sections("")
    assert sections == [("Full Document", "")]
