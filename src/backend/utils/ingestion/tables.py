"""Markdown table detection, extraction, linearization, and categorization.

Tables are detected as atomic blocks (header + separator + data rows). The raw
text is preserved for LLM extraction (good accuracy), and `linearize_table`
converts them into natural-language sentences for embedding (good semantic
similarity).
"""

import re
from dataclasses import dataclass

from utils.ingestion.sections import SECTION_CATEGORIES


@dataclass
class TableBlock:
    raw_text: str
    preceding_context: str
    start_line: int
    end_line: int


def detect_and_extract_tables(markdown_text: str) -> tuple[bool, list[TableBlock], str]:
    """
    Detect and extract all markdown tables from text.

    Returns tables as atomic blocks and the remaining text with tables replaced
    by [TABLE_REMOVED] placeholders so section splitting still works.

    Returns:
        (has_table, table_blocks, text_without_tables)
    """
    lines = markdown_text.splitlines()
    table_blocks: list[TableBlock] = []

    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

    def _is_separator_line(line: str) -> bool:
        return bool(re.search(r"\|[\s:]*-{2,}[\s:]*\|", line))

    i = 0
    while i < len(lines):
        if _is_table_line(lines[i]) or _is_separator_line(lines[i]):
            table_start = i
            # Walk backward to include the header row if we landed on a separator
            while table_start > 0 and _is_table_line(lines[table_start - 1]):
                table_start -= 1

            table_end = i + 1
            while table_end < len(lines) and (_is_table_line(lines[table_end]) or _is_separator_line(lines[table_end])):
                table_end += 1

            table_lines = lines[table_start:table_end]

            # Validate: a real table needs at least a header + separator + one data row
            has_separator = any(_is_separator_line(l) for l in table_lines)
            has_data_rows = sum(1 for l in table_lines if _is_table_line(l) and not _is_separator_line(l)) >= 2
            if has_separator and has_data_rows:
                # Gather preceding context (up to 3 non-empty lines before the table)
                context_lines = []
                j = table_start - 1
                while j >= 0 and len(context_lines) < 3:
                    stripped = lines[j].strip()
                    if stripped and not _is_table_line(lines[j]):
                        context_lines.insert(0, stripped)
                    elif not stripped and context_lines:
                        break
                    j -= 1

                table_blocks.append(
                    TableBlock(
                        raw_text="\n".join(table_lines),
                        preceding_context="\n".join(context_lines),
                        start_line=table_start,
                        end_line=table_end,
                    )
                )

            i = table_end
        else:
            i += 1

    # Build text_without_tables by replacing table regions with placeholder
    if not table_blocks:
        return False, [], markdown_text

    result_lines = list(lines)
    # Process in reverse so indices stay valid
    for tb in reversed(table_blocks):
        result_lines[tb.start_line : tb.end_line] = ["[TABLE_REMOVED]"]

    text_without_tables = "\n".join(result_lines)
    return True, table_blocks, text_without_tables


def linearize_table(table_text: str, preceding_context: str = "") -> str:
    """
    Convert a markdown table into natural-language sentences for embedding.

    The embedding model sees this linearized form (good semantic similarity),
    while the raw table is stored separately for LLM extraction (good accuracy).
    """
    lines = table_text.strip().splitlines()

    # Find header row (first non-separator table line)
    separator_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if re.search(r"\|[\s:]*-{2,}[\s:]*\|", stripped):
            separator_idx = idx
            break

    if separator_idx is None or separator_idx == 0:
        # No clear header/separator structure - return cleaned text
        cleaned = re.sub(r"[|\-]+", " ", table_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return f"{preceding_context}\n{cleaned}" if preceding_context else cleaned

    header_row = lines[separator_idx - 1]
    headers = [h.strip() for h in header_row.split("|") if h.strip()]
    data_lines = lines[separator_idx + 1 :]

    sentences = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        if not cells:
            continue

        pairs = []
        for col_idx, cell in enumerate(cells):
            col_name = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"
            pairs.append(f"{col_name}: {cell}")
        sentences.append(", ".join(pairs) + ".")

    linearized = " ".join(sentences)
    if preceding_context:
        return f"{preceding_context}\n{linearized}"
    return linearized


def categorize_table_content(table_text: str, preceding_context: str = "") -> str:
    """
    Categorize a table by running keyword matching against its cell values
    and preceding context.
    """
    combined = (preceding_context + " " + table_text).lower()

    for category, keywords in SECTION_CATEGORIES.items():
        if any(keyword in combined for keyword in keywords):
            return category

    return "OTHER"


def split_table_if_oversized(table_block: TableBlock, max_chars: int = 4000) -> list[TableBlock]:
    """
    If a table exceeds max_chars, split it into sub-tables by row groups,
    preserving the header row in each sub-chunk.
    """
    if len(table_block.raw_text) <= max_chars:
        return [table_block]

    lines = table_block.raw_text.splitlines()

    # Find header and separator
    header_line = None
    separator_line = None
    data_start = 0
    for idx, line in enumerate(lines):
        if re.search(r"\|[\s:]*-{2,}[\s:]*\|", line):
            separator_line = line
            if idx > 0:
                header_line = lines[idx - 1]
                data_start = idx + 1
            else:
                data_start = idx + 1
            break

    if header_line is None or separator_line is None:
        return [table_block]

    preamble = header_line + "\n" + separator_line
    preamble_len = len(preamble) + 1

    data_rows = lines[data_start:]
    sub_tables: list[TableBlock] = []
    current_rows: list[str] = []
    current_len = preamble_len

    for row in data_rows:
        row_len = len(row) + 1
        if current_rows and (current_len + row_len) > max_chars:
            sub_text = preamble + "\n" + "\n".join(current_rows)
            sub_tables.append(
                TableBlock(
                    raw_text=sub_text,
                    preceding_context=table_block.preceding_context,
                    start_line=table_block.start_line,
                    end_line=table_block.end_line,
                )
            )
            current_rows = []
            current_len = preamble_len

        current_rows.append(row)
        current_len += row_len

    if current_rows:
        sub_text = preamble + "\n" + "\n".join(current_rows)
        sub_tables.append(
            TableBlock(
                raw_text=sub_text,
                preceding_context=table_block.preceding_context,
                start_line=table_block.start_line,
                end_line=table_block.end_line,
            )
        )

    return sub_tables if sub_tables else [table_block]
