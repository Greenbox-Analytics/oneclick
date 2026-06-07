"""Tests for Zoe general (no-contract) mode backed by the reference book."""

import json
from unittest.mock import MagicMock, patch

from knowledge.reference_search import ReferencePassage
from zoe_chatbot.contract_chatbot import ContractChatbot


def _events(gen):
    return [json.loads(ev[6:].strip()) for ev in gen if ev.startswith("data: ")]


def _bot():
    bot = ContractChatbot.__new__(ContractChatbot)  # skip __init__/config
    bot._add_to_memory = MagicMock()
    bot._get_conversation_context = MagicMock(return_value=[])
    bot._sse_event = lambda t, d: "data: " + json.dumps({"type": t, **d}) + "\n\n"
    return bot


@patch.object(ContractChatbot, "_stream_llm_completion", return_value=iter(["Royalties ", "are net."]))
@patch("zoe_chatbot.contract_chatbot.search_reference")
def test_general_mode_with_hits_emits_reference_sources(mock_search, _llm):
    mock_search.return_value = [
        ReferencePassage(
            "Royalties are paid on net.", "Part V ▸ Record Deals", 210, 211, "The Music Business (Passman)", 0.71
        )
    ]
    bot = _bot()
    events = _events(bot._general_knowledge_stream("How are royalties computed?", session_id="s1"))
    src = next(e for e in events if e["type"] == "sources")
    assert src["sources"] == []  # contract sources untouched
    assert src["reference_sources"][0]["book_title"] == "The Music Business (Passman)"
    assert src["reference_sources"][0]["pages"] == "210-211"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Royalties are net."
    assert mock_search.call_args.kwargs.get("floor_count") == 5
    data_ev = next(e for e in events if e["type"] == "data")
    assert data_ev["answered_from"] == "reference_book"
    assert data_ev["confidence"] == "high"


@patch.object(ContractChatbot, "_stream_llm_completion", return_value=iter(["General answer."]))
@patch("zoe_chatbot.contract_chatbot.search_reference")
def test_general_mode_no_hits_low_confidence_no_sources(mock_search, _llm):
    mock_search.return_value = []
    bot = _bot()
    events = _events(bot._general_knowledge_stream("What's the weather?", session_id="s1"))
    src = next(e for e in events if e["type"] == "sources")
    assert src["reference_sources"] == []
    data_ev = next(e for e in events if e["type"] == "data")
    assert data_ev["confidence"] == "low"
    assert data_ev["answered_from"] == "general_knowledge"


@patch.object(ContractChatbot, "_stream_llm_completion", return_value=iter(["General answer."]))
@patch("zoe_chatbot.contract_chatbot.search_reference", side_effect=RuntimeError("Pinecone down"))
def test_general_mode_survives_retrieval_failure(mock_search, _llm):
    bot = _bot()
    events = _events(bot._general_knowledge_stream("How are royalties computed?", session_id="s1"))
    # No error event — it degrades to a general-knowledge answer.
    assert not any(e["type"] == "error" for e in events)
    src = next(e for e in events if e["type"] == "sources")
    assert src["reference_sources"] == []
    data_ev = next(e for e in events if e["type"] == "data")
    assert data_ev["answered_from"] == "general_knowledge"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "General answer."


@patch("zoe_chatbot.contract_chatbot.search_reference")
def test_general_mode_prompt_has_topical_guard(mock_search):
    mock_search.return_value = []
    captured = {}

    def _capture(messages, *a, **k):
        captured["messages"] = messages
        return iter(["ok"])

    bot = _bot()
    with patch.object(bot, "_stream_llm_completion", side_effect=_capture):
        list(bot._general_knowledge_stream("coding help?", session_id="s1"))
    system = captured["messages"][0]["content"].lower()
    assert "music" in system and ("decline" in system or "redirect" in system or "not related" in system)
