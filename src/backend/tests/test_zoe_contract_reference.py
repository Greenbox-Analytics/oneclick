"""Tests for Zoe contract-assist mode injecting labeled book context."""

import json
from unittest.mock import MagicMock, patch

from knowledge.reference_search import ReferencePassage
from zoe_chatbot.contract_chatbot import ContractChatbot


def _events(gen):
    return [json.loads(ev[6:].strip()) for ev in gen if ev.startswith("data: ")]


def _bot():
    bot = ContractChatbot.__new__(ContractChatbot)
    bot._add_to_memory = MagicMock()
    bot._get_conversation_context = MagicMock(return_value=[])
    bot._extract_structured_data = MagicMock(return_value=None)
    bot._extract_suggestion_from_answer = MagicMock(return_value=("ans", None))
    bot.memory = MagicMock()  # guard: set_pending_suggestion etc. never AttributeError
    bot.llm_model = "gpt-x"
    bot._sse_event = lambda t, d: "data: " + json.dumps({"type": t, **d}) + "\n\n"
    return bot


def _ask_stream_bot():
    """Build a minimal ContractChatbot stub suitable for driving ask_stream's contract path."""
    bot = ContractChatbot.__new__(ContractChatbot)
    # Memory stub — needed for get_last_contract_ids, set_last_contract_ids, get_pending_suggestion
    mem = MagicMock()
    mem.get_last_contract_ids.return_value = None
    mem.get_pending_suggestion.return_value = None
    bot.memory = mem
    bot.llm_model = "gpt-x"
    bot._add_to_memory = MagicMock()
    bot._sse_event = lambda t, d: "data: " + json.dumps({"type": t, **d}) + "\n\n"
    return bot


def test_reference_message_added_and_after_contract():
    captured = {}

    def _capture(messages, *a, **k):
        captured["messages"] = messages
        return iter(["Answer."])

    bot = _bot()
    with patch.object(bot, "_stream_llm_completion", side_effect=_capture):
        passages = [ReferencePassage("Net royalty basis.", "Part V", 210, 210, "Passman", 0.66)]
        events = _events(
            bot._generate_answer_stream_full_doc(
                "q",
                "=== CONTRACT: a.pdf ===\nbody",
                session_id="s1",
                reference_context="[Part V — pp. 210-210]\nNet royalty basis.",
                reference_passages=passages,
            )
        )
    src = next(e for e in events if e["type"] == "sources")
    assert src["sources"] == []
    assert src["reference_sources"][0]["book_title"] == "Passman"
    contents = [m["content"] for m in captured["messages"]]
    contract_idx = next(i for i, c in enumerate(contents) if "CONTRACT: a.pdf" in c)
    ref_idx = next(i for i, c in enumerate(contents) if "Net royalty basis." in c and "REFERENCE" in c)
    assert contract_idx < ref_idx  # contract governs; reference comes after


def test_reference_message_instructs_supplemental_structure():
    # When book context is present, the model is told to answer from the contract first
    # and only append a short supplemental section if the background is applicable.
    captured = {}

    def _capture(messages, *a, **k):
        captured["messages"] = messages
        return iter(["Answer."])

    bot = _bot()
    with patch.object(bot, "_stream_llm_completion", side_effect=_capture):
        passages = [ReferencePassage("Net royalty basis.", "Part V", 210, 210, "Passman", 0.66)]
        list(
            bot._generate_answer_stream_full_doc(
                "q",
                "=== CONTRACT: a.pdf ===\nbody",
                session_id="s1",
                reference_context="Net royalty basis.",
                reference_passages=passages,
            )
        )
    # Target the actual background message (the one carrying the passage text), not the
    # system-prompt rule that also mentions "BACKGROUND REFERENCE".
    ref_msg = next(m["content"] for m in captured["messages"] if "Net royalty basis." in m["content"])
    assert "Supplemental" in ref_msg  # the supplemental section is specified
    assert "STOP" in ref_msg  # instructed to add nothing when the contract fully answers


def test_no_reference_keeps_empty_sources():
    bot = _bot()
    with patch.object(bot, "_stream_llm_completion", return_value=iter(["Answer."])):
        events = _events(bot._generate_answer_stream_full_doc("q", "=== CONTRACT: a.pdf ===\nbody", session_id="s1"))
    src = next(e for e in events if e["type"] == "sources")
    assert src["sources"] == []
    assert src["reference_sources"] == []


def test_ask_stream_contract_path_survives_search_reference_failure():
    """Resilience: if search_reference raises inside ask_stream's contract path,
    the stream must still complete normally (reference_passages falls back to []).
    """
    bot = _ask_stream_bot()

    # Sentinel SSE events from the downstream streamer — proves the stream completed.
    sentinel_events = [
        "data: " + json.dumps({"type": "sources", "sources": [], "reference_sources": []}) + "\n\n",
        "data: " + json.dumps({"type": "token", "token": "Answer."}) + "\n\n",
        "data: " + json.dumps({"type": "complete", "answer": "Answer.", "sources": []}) + "\n\n",
    ]

    # Track what reference_passages value was forwarded to _generate_answer_stream_full_doc.
    captured = {}

    def _fake_generate(
        query,
        full_doc_context,
        session_id,
        model_override=None,
        reference_context="",
        reference_passages=None,
        contract_names=None,
        selection_note="",
        selected_ids=None,
    ):
        captured["reference_passages"] = reference_passages
        captured["reference_context"] = reference_context
        yield from iter(sentinel_events)

    with (
        patch("zoe_chatbot.contract_chatbot.search_reference", side_effect=RuntimeError("Pinecone down")),
        patch.object(bot, "_summarize_if_needed", return_value=None),
        patch.object(bot, "_llm_route_decision", return_value=MagicMock(route="contract", answer_mode="retrieve")),
        patch.object(bot, "_is_conversational_query", return_value=False),
        patch.object(bot, "_is_affirmative_response", return_value=False),
        patch.object(bot, "_generate_answer_stream_full_doc", side_effect=_fake_generate),
    ):
        events = list(
            bot.ask_stream(
                query="What are the royalty splits?",
                user_id="u1",
                project_id="proj-1",
                contract_ids=["c1"],
                session_id="s1",
                contract_markdowns={"c1": "Contract body text."},
                contract_names={"c1": "deal.pdf"},
            )
        )

    # Stream must have produced output (did not raise)
    assert len(events) > 0

    # The fallback must have been applied: empty list, not an exception propagation
    assert captured.get("reference_passages") == [], (
        f"Expected reference_passages=[] when search_reference raises, got: {captured.get('reference_passages')!r}"
    )
    assert captured.get("reference_context") == "", "Expected empty reference_context when search_reference raises"
