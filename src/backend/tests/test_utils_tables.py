"""Tests for markdown-table detection, linearization, categorization, and
oversize-splitting in utils/ingestion/tables.py.

These helpers had zero direct coverage before the refactor.
"""

from utils.ingestion.tables import (
    TableBlock,
    categorize_table_content,
    detect_and_extract_tables,
    linearize_table,
    split_table_if_oversized,
)

# ─── detect_and_extract_tables ───────────────────────────────────────────────


def test_detect_finds_table_with_header_separator_and_rows():
    md = "Intro text.\n| Name | Share |\n|---|---|\n| Alice | 50% |\n| Bob | 50% |\nOutro text."
    has_table, blocks, _ = detect_and_extract_tables(md)
    assert has_table is True
    assert len(blocks) == 1
    assert "| Alice | 50% |" in blocks[0].raw_text
    assert "| Bob | 50% |" in blocks[0].raw_text


def test_detect_rejects_header_only_table_with_no_data_rows():
    """Validator requires ≥2 non-separator table lines. Header + separator alone
    (no data row) yields only 1 non-separator line and is rejected."""
    md = "Some prose\n| Header |\n|---|\nMore prose"
    has_table, blocks, text = detect_and_extract_tables(md)
    assert has_table is False
    assert blocks == []
    assert text == md  # unchanged


def test_detect_returns_placeholder_for_extracted_region():
    md = "Before.\n| Name | Share |\n|---|---|\n| Alice | 50% |\n| Bob | 50% |\nAfter."
    _, _, text_without_tables = detect_and_extract_tables(md)
    assert "[TABLE_REMOVED]" in text_without_tables
    assert "| Alice | 50% |" not in text_without_tables
    assert "Before." in text_without_tables
    assert "After." in text_without_tables


def test_detect_no_tables_returns_unchanged_text():
    md = "Just some prose. No tables here at all."
    has_table, blocks, text = detect_and_extract_tables(md)
    assert has_table is False
    assert blocks == []
    assert text == md


def test_detect_preserves_preceding_context():
    md = "Schedule A: Royalty Rates\n| Name | Share |\n|---|---|\n| Alice | 50% |\n| Bob | 50% |"
    _, blocks, _ = detect_and_extract_tables(md)
    assert blocks[0].preceding_context == "Schedule A: Royalty Rates"


# ─── linearize_table ─────────────────────────────────────────────────────────


def test_linearize_converts_rows_to_sentences():
    table = "| Name | Share |\n|---|---|\n| Alice | 50% |"
    out = linearize_table(table)
    assert "Name: Alice" in out
    assert "Share: 50%" in out
    assert out.endswith(".")


def test_linearize_includes_preceding_context_prefix():
    table = "| Name | Share |\n|---|---|\n| Alice | 50% |"
    out = linearize_table(table, preceding_context="Royalty Schedule")
    assert out.startswith("Royalty Schedule\n")


def test_linearize_no_separator_falls_back_to_cleaned_text():
    # Single line without the |---| separator → fallback path.
    out = linearize_table("Some text with | symbols | but - no separator")
    # The fallback removes `|` and `-` and collapses whitespace.
    assert "|" not in out
    assert "Some text with" in out


# ─── categorize_table_content ────────────────────────────────────────────────


def test_categorize_table_routes_royalty_keywords():
    table = "| Type | Share |\n|---|---|\n| Royalty | 50% |"
    assert categorize_table_content(table) == "ROYALTY_CALCULATIONS"


def test_categorize_table_uses_preceding_context_too():
    table = "| Col | Val |\n|---|---|\n| x | 1 |"
    # Without context, falls to OTHER (no keyword in cells).
    assert categorize_table_content(table) == "OTHER"
    # "songwriter" in context is a PUBLISHING_RIGHTS keyword that doesn't
    # collide with any earlier category's keywords, so PUBLISHING_RIGHTS wins.
    assert categorize_table_content(table, preceding_context="songwriter credits") == "PUBLISHING_RIGHTS"


def test_categorize_table_unknown_returns_other():
    table = "| Foo | Bar |\n|---|---|\n| x | y |"
    assert categorize_table_content(table) == "OTHER"


# ─── split_table_if_oversized ────────────────────────────────────────────────


def _build_oversized_table(row_count: int, row_text: str = "| Alice | 50% |") -> TableBlock:
    """Build a TableBlock whose raw_text exceeds 4000 chars."""
    header = "| Name | Share |"
    sep = "|---|---|"
    rows = "\n".join([row_text] * row_count)
    raw = f"{header}\n{sep}\n{rows}"
    return TableBlock(raw_text=raw, preceding_context="", start_line=0, end_line=row_count + 2)


def test_split_small_table_returns_single_block():
    block = _build_oversized_table(row_count=2)  # well under 4000 chars
    assert len(split_table_if_oversized(block)) == 1


def test_split_oversized_preserves_header_in_each_chunk():
    block = _build_oversized_table(row_count=400)  # ~6400 chars of rows
    assert len(block.raw_text) > 4000  # sanity
    chunks = split_table_if_oversized(block, max_chars=4000)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "| Name | Share |" in chunk.raw_text  # header preserved
        assert "|---|---|" in chunk.raw_text  # separator preserved


def test_split_no_separator_returns_single_block_gracefully():
    """If the input has no separator, the splitter can't find a header to
    repeat — it returns the block unchanged rather than crashing."""
    raw = "| Name | Alice |\n" + ("| Other | Long row |\n" * 500)
    block = TableBlock(raw_text=raw, preceding_context="", start_line=0, end_line=501)
    assert len(block.raw_text) > 4000
    assert len(split_table_if_oversized(block)) == 1
