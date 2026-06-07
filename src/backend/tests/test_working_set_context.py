"""Byte-stable context block + per-turn selection note + prefer-selected citation."""

from zoe_chatbot.contract_chatbot import build_labeled_context, build_selection_note


def test_context_block_is_byte_stable_headers_by_id():
    processed = {"c2": "BODY TWO", "c1": "BODY ONE"}
    names = {"c1": "Old.pdf", "c2": "New.pdf"}
    ctx = build_labeled_context(processed, names)
    assert ctx.index("Old.pdf") < ctx.index("New.pdf")  # ordered by id
    assert "(currently selected)" not in ctx and "earlier in this conversation" not in ctx
    assert "=== CONTRACT: Old.pdf ===" in ctx and "=== CONTRACT: New.pdf ===" in ctx


def test_selection_note_cases():
    names = {"c1": "Old.pdf", "c2": "New.pdf"}
    carried = build_selection_note(processed_ids={"c1", "c2"}, selected_ids={"c2"}, contract_names=names)
    assert "New.pdf" in carried and "compare" in carried.lower() and "Old.pdf" not in carried
    both = build_selection_note(processed_ids={"c1", "c2"}, selected_ids={"c1", "c2"}, contract_names=names)
    assert "Old.pdf" in both and "New.pdf" in both and "carried" not in both.lower()
    assert build_selection_note(processed_ids={"c2"}, selected_ids={"c2"}, contract_names=names) == ""
    race = build_selection_note(processed_ids={"c2"}, selected_ids={"c1"}, contract_names=names)
    assert "Old.pdf" not in race
