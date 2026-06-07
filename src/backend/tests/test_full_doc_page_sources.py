"""Tests for marker-stripped answers and page-aware sources (approach A)."""

import json
from unittest.mock import MagicMock

from zoe_chatbot.contract_chatbot import ContractChatbot


def _events(gen):
    return [json.loads(ev[6:].strip()) for ev in gen if ev.startswith("data: ")]


def _bot():
    bot = ContractChatbot.__new__(ContractChatbot)  # skip __init__/config
    bot.llm_model = "test-model"
    bot.memory = MagicMock()
    bot._add_to_memory = MagicMock()
    bot._get_conversation_context = MagicMock(return_value=[])
    bot._extract_structured_data = MagicMock(return_value=None)
    bot._extract_suggestion_from_answer = MagicMock(side_effect=lambda a: (a, None))
    bot._sse_event = lambda t, d: "data: " + json.dumps({"type": t, **d}) + "\n\n"
    return bot


_CTX = "=== CONTRACT: Deal.pdf ===\n[[PAGE 1]]\nintro\n[[PAGE 4]]\nRoyalties are 50%.\n"


def _run(bot, names, query="What are the royalty splits?", selected_ids=None):
    # Default: every contract shown is treated as selected (matches production, where the Tier-3
    # caller always passes the selection). Pass selected_ids explicitly to model carried contracts.
    return _events(
        bot._generate_answer_stream_full_doc(
            query,
            _CTX,
            session_id="s1",
            contract_names=names,
            selected_ids=set(names) if selected_ids is None else selected_ids,
        )
    )


def test_answer_context_is_marker_stripped():
    bot = _bot()
    bot._stream_llm_completion = MagicMock(
        side_effect=[iter(["The split is 50%."]), iter(['[{"contract": "Deal.pdf", "page": 4}]'])]
    )
    _run(bot, {"cid-1": "Deal.pdf"})
    answer_messages = bot._stream_llm_completion.call_args_list[0].args[0]
    blob = " ".join(m["content"] for m in answer_messages)
    assert "[[PAGE" not in blob


def test_selected_contracts_get_chips_cited_or_not():
    # Both contracts selected; Deal is cited (page 4), Other is selected-but-uncited.
    # Every selected contract gets a chip — cited carries its page, selected-uncited is page-less.
    bot = _bot()
    bot._stream_llm_completion = MagicMock(
        side_effect=[iter(["Answer."]), iter(['[{"contract": "Deal.pdf", "page": 4}]'])]
    )
    events = _run(bot, {"cid-1": "Deal.pdf", "cid-2": "Other.pdf"})  # _run selects both by default
    types = [e["type"] for e in events]
    src_idx = max(i for i, e in enumerate(events) if e["type"] == "sources")
    assert types.index("done") < src_idx, "page sources must come AFTER done"
    by_file = {s["contract_file"]: s.get("page_number") for s in events[src_idx]["sources"]}
    assert by_file == {"Deal.pdf": 4, "Other.pdf": None}  # selected-uncited still gets a page-less chip
    # Each chip carries the authoritative contract id (so the PDF opens by id, never by filename).
    by_id = {s["contract_file"]: s.get("contract_id") for s in events[src_idx]["sources"]}
    assert by_id == {"Deal.pdf": "cid-1", "Other.pdf": "cid-2"}
    assert events[src_idx]["reference_sources"] == []  # no book-cite double-count


def test_no_extra_sources_event_when_extraction_empty():
    bot = _bot()
    bot._stream_llm_completion = MagicMock(side_effect=[iter(["Answer."]), iter(["not json"])])
    events = _run(bot, {"cid-1": "Deal.pdf"})
    assert [e for e in events if e["type"] == "sources" and e["sources"]] == []
    assert any(e["type"] == "done" for e in events)


def test_extraction_not_called_for_fallback_answer():
    bot = _bot()
    bot._stream_llm_completion = MagicMock(side_effect=[iter(["I couldn't find that information in the contract."])])
    events = _run(bot, {"cid-1": "Deal.pdf"})
    assert bot._stream_llm_completion.call_count == 1  # no extraction call
    assert any(e["type"] == "done" for e in events)


def test_extraction_exception_does_not_break_stream():
    bot = _bot()
    bot._stream_llm_completion = MagicMock(side_effect=[iter(["An answer."]), RuntimeError("LLM down")])
    events = _run(bot, {"cid-1": "Deal.pdf"})
    assert "".join(e["content"] for e in events if e["type"] == "token") == "An answer."
    assert any(e["type"] == "done" for e in events)


def test_fuzzy_resolves_model_label_to_known_filename():
    # Model drops the extension ("Deal" vs "Deal.pdf"); the fuzzy fallback should
    # resolve it to the known filename so the chip still carries the page.
    bot = _bot()
    bot._stream_llm_completion = MagicMock(side_effect=[iter(["Answer."]), iter(['[{"contract": "Deal", "page": 4}]'])])
    events = _run(bot, {"cid-1": "Deal.pdf"})
    src_idx = max(i for i, e in enumerate(events) if e["type"] == "sources")
    by_file = {s["contract_file"]: s.get("page_number") for s in events[src_idx]["sources"]}
    assert by_file == {"Deal.pdf": 4}


def test_carried_uncited_contract_gets_no_chip():
    # Two contracts in context: Deal.pdf (selected), Other.pdf (carried, not cited).
    # Extraction returns an extension-less label matching only the SELECTED contract.
    # After scoping: only cited-or-selected contracts appear in chips; Other.pdf (not
    # cited, not selected) must be absent.
    bot = _bot()
    answer_iter = iter(["Answer about Deal."])
    extraction_iter = iter(['[{"contract": "Deal", "page": 7}]'])
    bot._stream_llm_completion = MagicMock(side_effect=[answer_iter, extraction_iter])

    names = {"cid-1": "Deal.pdf", "cid-2": "Other.pdf"}
    ctx = "=== CONTRACT: Deal.pdf ===\n[[PAGE 7]]\nsome content.\n=== CONTRACT: Other.pdf ===\n[[PAGE 1]]\nother.\n"
    events = _events(
        bot._generate_answer_stream_full_doc(
            "What are the terms?",
            ctx,
            session_id="s2",
            contract_names=names,
            selected_ids={"cid-1"},
        )
    )

    source_events = [e for e in events if e["type"] == "sources" and e.get("sources")]
    # There must be exactly one page-sources event (after done)
    assert source_events, "expected at least one non-empty sources event"
    last_src = source_events[-1]
    by_file = {s["contract_file"]: s.get("page_number") for s in last_src["sources"]}
    # Selected+cited -> chip with page
    assert by_file.get("Deal.pdf") == 7, f"expected Deal.pdf page 7, got {by_file}"
    # Carried+uncited -> no chip at all
    assert "Other.pdf" not in by_file, f"Other.pdf should not appear in chips: {by_file}"


def test_fuzzy_citation_prefers_selected_over_carried():
    # The label "Master" is a substring of BOTH filenames (ambiguous). With cid-2 selected,
    # the fuzzy fallback must resolve to the SELECTED contract, not the carried one.
    # Without the selected-first ordering this resolves to the carried A.pdf and the test fails.
    bot = _bot()
    bot._stream_llm_completion = MagicMock(
        side_effect=[iter(["Answer."]), iter(['[{"contract": "Master", "page": 3}]'])]
    )
    names = {"cid-1": "Master Recording A.pdf", "cid-2": "Master Recording B.pdf"}
    ctx = (
        "=== CONTRACT: Master Recording A.pdf ===\n[[PAGE 1]]\na.\n"
        "=== CONTRACT: Master Recording B.pdf ===\n[[PAGE 3]]\nb.\n"
    )
    events = _events(
        bot._generate_answer_stream_full_doc(
            "terms?", ctx, session_id="s3", contract_names=names, selected_ids={"cid-2"}
        )
    )
    src = [e for e in events if e["type"] == "sources" and e.get("sources")][-1]["sources"]
    by_file = {s["contract_file"]: s.get("page_number") for s in src}
    assert by_file.get("Master Recording B.pdf") == 3  # resolved to the SELECTED contract
    assert "Master Recording A.pdf" not in by_file  # carried + uncited -> no chip
