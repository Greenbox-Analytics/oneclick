"""Tests for registry.contract_splits.parse_royalty_splits.

Reuses the OpenAI-mock pattern from tests/test_utils_contract_parser.py so the
real LLM is never called. Locks the master/publishing pivoting + main-artist
detection behavior the Add Work wizard depends on.
"""

import json
from unittest.mock import MagicMock

import pytest


def _client_returning(payload: dict):
    """Mirrors tests/test_utils_contract_parser.py:_client_returning."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    return client


@pytest.fixture
def patched_parser(monkeypatch):
    """Patch the lazy-loaded OpenAI client + API-key guard so we can run offline."""
    import utils.contract_parsing.parser as parser_module

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _install(payload: dict):
        monkeypatch.setattr(parser_module, "get_openai_client", lambda: _client_returning(payload))

    return _install


# ─── pivoting ──────────────────────────────────────────────────────────────────


def test_pivots_master_and_publishing_into_per_party_buckets(patched_parser):
    patched_parser(
        {
            "parties": [
                {"name": "Nova Sky", "role": "Artist"},
                {"name": "Marcus Adebayo", "role": "Producer"},
            ],
            "works": [{"title": "Neon Tide", "work_type": "song"}],
            "royalty_shares": [
                {"party_name": "Nova Sky", "royalty_type": "Master", "percentage": 60.0},
                {"party_name": "Nova Sky", "royalty_type": "Publishing", "percentage": 70.0},
                {"party_name": "Marcus Adebayo", "royalty_type": "Producer", "percentage": 40.0},
                {"party_name": "Marcus Adebayo", "royalty_type": "Publishing", "percentage": 30.0},
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="contract markdown", main_artist_name="Nova Sky")

    assert result["main_artist_found"] is True
    by_name = {p["name"]: p for p in result["parties"]}
    assert by_name["Nova Sky"]["master_pct"] == 60.0
    assert by_name["Nova Sky"]["publishing_pct"] == 70.0
    assert by_name["Nova Sky"]["is_main_artist"] is True
    # Producer payout buckets into master per design
    assert by_name["Marcus Adebayo"]["master_pct"] == 40.0
    assert by_name["Marcus Adebayo"]["publishing_pct"] == 30.0
    assert by_name["Marcus Adebayo"]["is_main_artist"] is False


def test_streaming_share_pivots_into_master_bucket(patched_parser):
    patched_parser(
        {
            "parties": [{"name": "Nova Sky", "role": "Artist"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {"party_name": "Nova Sky", "royalty_type": "Streaming", "percentage": 50.0},
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    nova = next(p for p in result["parties"] if p["name"] == "Nova Sky")
    assert nova["master_pct"] == 50.0
    assert nova["publishing_pct"] == 0.0


# ─── main-artist detection ────────────────────────────────────────────────────


def test_main_artist_found_true_when_party_matches(patched_parser):
    patched_parser(
        {
            "parties": [{"name": "Nova Sky", "role": "Artist"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [{"party_name": "Nova Sky", "royalty_type": "Master", "percentage": 100.0}],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    assert result["main_artist_found"] is True


def test_main_artist_not_found_drops_collaborator_only(patched_parser):
    """When the main artist isn't named, only collaborators come back."""
    patched_parser(
        {
            "parties": [{"name": "Marcus Adebayo", "role": "Producer"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [{"party_name": "Marcus Adebayo", "royalty_type": "Producer", "percentage": 40.0}],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    assert result["main_artist_found"] is False
    assert len(result["parties"]) == 1
    assert result["parties"][0]["name"] == "Marcus Adebayo"
    assert result["parties"][0]["is_main_artist"] is False


def test_main_artist_matched_via_substring(patched_parser):
    """Fuzzy compare: 'Nova Sky (p/k/a Nova)' should still match 'Nova Sky'."""
    patched_parser(
        {
            "parties": [{"name": "Nova Sky p/k/a Nova", "role": "Artist"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [{"party_name": "Nova Sky p/k/a Nova", "royalty_type": "Master", "percentage": 60.0}],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    assert result["main_artist_found"] is True


# ─── empty contract result ─────────────────────────────────────────────────────


def test_zero_parties_returns_empty(patched_parser):
    """Caller (router) toasts an error and falls back to manual mode in this case."""
    patched_parser(
        {
            "parties": [],
            "works": [],
            "royalty_shares": [],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    assert result["parties"] == []
    assert result["main_artist_found"] is False


# ─── input validation ─────────────────────────────────────────────────────────


def test_requires_text_or_pdf_path():
    from registry.contract_splits import parse_royalty_splits

    with pytest.raises(ValueError, match="Either text or pdf_path"):
        parse_royalty_splits(main_artist_name="Nova Sky")
