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


def _run(bot, names, query="What are the royalty splits?"):
    return _events(bot._generate_answer_stream_full_doc(query, _CTX, session_id="s1", contract_names=names))


def test_answer_context_is_marker_stripped():
    bot = _bot()
    bot._stream_llm_completion = MagicMock(
        side_effect=[iter(["The split is 50%."]), iter(['[{"contract": "Deal.pdf", "page": 4}]'])]
    )
    _run(bot, {"cid-1": "Deal.pdf"})
    answer_messages = bot._stream_llm_completion.call_args_list[0].args[0]
    blob = " ".join(m["content"] for m in answer_messages)
    assert "[[PAGE" not in blob


def test_emits_all_chips_with_page_on_cited_after_done():
    bot = _bot()
    bot._stream_llm_completion = MagicMock(
        side_effect=[iter(["Answer."]), iter(['[{"contract": "Deal.pdf", "page": 4}]'])]
    )
    events = _run(bot, {"cid-1": "Deal.pdf", "cid-2": "Other.pdf"})
    types = [e["type"] for e in events]
    src_idx = max(i for i, e in enumerate(events) if e["type"] == "sources")
    assert types.index("done") < src_idx, "page sources must come AFTER done"
    src = events[src_idx]["sources"]
    by_file = {s["contract_file"]: s.get("page_number") for s in src}
    assert by_file == {"Deal.pdf": 4, "Other.pdf": None}  # all chips kept, page on cited only
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
