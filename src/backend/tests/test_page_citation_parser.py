"""Tests for parse_page_citations — tolerant extraction of {contract, page}."""

from zoe_chatbot.contract_chatbot import parse_page_citations


def test_parses_bare_json_array():
    assert parse_page_citations('[{"contract": "Deal.pdf", "page": 4}]') == [{"contract": "Deal.pdf", "page": 4}]


def test_parses_fenced_json_with_prose():
    raw = 'Here:\n```json\n[{"contract": "Deal.pdf", "page": "12"}]\n```\nthanks'
    assert parse_page_citations(raw) == [{"contract": "Deal.pdf", "page": 12}]


def test_multi_contract():
    raw = '[{"contract": "A.pdf", "page": 2}, {"contract": "B.pdf", "page": 9}]'
    assert parse_page_citations(raw) == [
        {"contract": "A.pdf", "page": 2},
        {"contract": "B.pdf", "page": 9},
    ]


def test_garbage_and_bad_shapes_return_empty():
    assert parse_page_citations("") == []
    assert parse_page_citations("no json here") == []
    assert parse_page_citations('{"contract": "A.pdf", "page": 1}') == []  # object, not array
    assert parse_page_citations('[{"contract": 5, "page": 1}]') == []  # contract not a string
    assert parse_page_citations('[{"contract": "A.pdf", "page": 0}]') == []  # page < 1
    assert parse_page_citations('[{"contract": "A.pdf"}]') == []  # missing page
