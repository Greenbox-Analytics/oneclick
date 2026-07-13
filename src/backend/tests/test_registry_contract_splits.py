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


# ─── soundexchange bucketing ──────────────────────────────────────────────────


def test_soundexchange_share_pivots_into_own_bucket(patched_parser):
    """SoundExchange shares stay out of the master total."""
    patched_parser(
        {
            "parties": [{"name": "Nova Sky", "role": "Artist"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {"party_name": "Nova Sky", "royalty_type": "Master", "percentage": 60.0},
                {"party_name": "Nova Sky", "royalty_type": "SoundExchange", "percentage": 45.0},
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    nova = next(p for p in result["parties"] if p["name"] == "Nova Sky")
    assert nova["master_pct"] == 60.0
    assert nova["soundexchange_pct"] == 45.0
    assert nova["publishing_pct"] == 0.0


def test_soundexchange_only_party_survives_zero_split_filter(patched_parser):
    patched_parser(
        {
            "parties": [
                {"name": "Nova Sky", "role": "Artist"},
                {"name": "Marcus Adebayo", "role": "Producer"},
            ],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {"party_name": "Nova Sky", "royalty_type": "Master", "percentage": 100.0},
                {
                    "party_name": "Marcus Adebayo",
                    "royalty_type": "SoundExchange",
                    "percentage": 10.0,
                },
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    by_name = {p["name"]: p for p in result["parties"]}
    assert "Marcus Adebayo" in by_name
    assert by_name["Marcus Adebayo"]["soundexchange_pct"] == 10.0
    assert by_name["Marcus Adebayo"]["master_pct"] == 0.0


def test_non_interactive_digital_performance_buckets_to_soundexchange(patched_parser):
    """Regression: the soundexchange substring check must run before the
    streaming-terms fallback, which also contains this phrase."""
    patched_parser(
        {
            "parties": [{"name": "Nova Sky", "role": "Artist"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {
                    "party_name": "Nova Sky",
                    "royalty_type": "non-interactive digital performance royalties",
                    "percentage": 25.0,
                },
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    nova = next(p for p in result["parties"] if p["name"] == "Nova Sky")
    assert nova["soundexchange_pct"] == 25.0
    assert nova["master_pct"] == 0.0


def test_bare_digital_performance_royalties_stay_in_master(patched_parser):
    """Guards the conservative mapping: only explicit SoundExchange /
    non-interactive terms leave the master bucket."""
    patched_parser(
        {
            "parties": [{"name": "Nova Sky", "role": "Artist"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {
                    "party_name": "Nova Sky",
                    "royalty_type": "digital performance royalties",
                    "percentage": 30.0,
                },
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")
    nova = next(p for p in result["parties"] if p["name"] == "Nova Sky")
    assert nova["master_pct"] == 30.0
    assert nova["soundexchange_pct"] == 0.0


# ─── zero-split party filtering ───────────────────────────────────────────────


def test_party_without_royalty_share_is_dropped(patched_parser):
    """Parties merely mentioned in the contract (no split) are filtered out."""
    patched_parser(
        {
            "parties": [
                {"name": "Nova Sky", "role": "Artist"},
                {"name": "Big Label LLC", "role": "Label"},
                {"name": "Rita Moses", "role": "Manager"},
            ],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {"party_name": "Nova Sky", "royalty_type": "Master", "percentage": 60.0},
                # Unbucketable royalty type — counts as no split.
                {"party_name": "Rita Moses", "royalty_type": "flat fee", "percentage": 10.0},
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")

    names = {p["name"] for p in result["parties"]}
    assert names == {"Nova Sky"}


def test_main_artist_kept_even_with_zero_split(patched_parser):
    """The main artist prefills the "You" row, so they survive the filter."""
    patched_parser(
        {
            "parties": [
                {"name": "Nova Sky", "role": "Artist"},
                {"name": "Marcus Adebayo", "role": "Producer"},
            ],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {"party_name": "Marcus Adebayo", "royalty_type": "Producer", "percentage": 40.0},
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Nova Sky")

    assert result["main_artist_found"] is True
    by_name = {p["name"]: p for p in result["parties"]}
    assert set(by_name) == {"Nova Sky", "Marcus Adebayo"}
    assert by_name["Nova Sky"]["is_main_artist"] is True
    assert by_name["Nova Sky"]["master_pct"] == 0.0
    assert by_name["Nova Sky"]["publishing_pct"] == 0.0


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
    """Fuzzy compare: 'Nova Sky p/k/a Nova' still matches 'Nova Sky' — the
    marker splitter canonicalizes the name and the alias carries 'Nova'."""
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
    main = next(p for p in result["parties"] if p["is_main_artist"])
    assert main["name"] == "Nova Sky"
    assert main["aliases"] == ["Nova"]
    assert main["master_pct"] == 60.0


def test_main_artist_matched_via_exact_alias(patched_parser):
    """Legal name in `name`, stage name in `aliases` — artist matched by alias."""
    patched_parser(
        {
            "parties": [{"name": "Jane Q. Doe", "role": "Artist", "aliases": ["Jasmine Kiara"]}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [{"party_name": "Jane Q. Doe", "royalty_type": "Master", "percentage": 55.0}],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="Jasmine Kiara")
    assert result["main_artist_found"] is True
    main = next(p for p in result["parties"] if p["is_main_artist"])
    assert main["name"] == "Jane Q. Doe"
    assert main["aliases"] == ["Jasmine Kiara"]
    assert main["master_pct"] == 55.0


def test_royalty_share_attached_via_alias_key(patched_parser):
    """A share emitted under the stage name attaches to the legal-name party."""
    patched_parser(
        {
            "parties": [{"name": "Jane Q. Doe", "role": "Artist", "aliases": ["Jasmine Kiara"]}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [{"party_name": "Jasmine Kiara", "royalty_type": "Publishing", "percentage": 40.0}],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="")
    assert len(result["parties"]) == 1
    party = result["parties"][0]
    assert party["name"] == "Jane Q. Doe"
    assert party["publishing_pct"] == 40.0


def test_output_parties_always_carry_aliases_list(patched_parser):
    """`aliases` is present (possibly empty) on every party, even when the LLM
    omits the field or a party is registered on the fly from a share."""
    patched_parser(
        {
            "parties": [{"name": "Marcus Adebayo", "role": "Producer"}],
            "works": [{"title": "X", "work_type": "song"}],
            "royalty_shares": [
                {"party_name": "Marcus Adebayo", "royalty_type": "Producer", "percentage": 40.0},
                {"party_name": "Unlisted Publisher", "royalty_type": "Publishing", "percentage": 10.0},
            ],
        }
    )
    from registry.contract_splits import parse_royalty_splits

    result = parse_royalty_splits(text="md", main_artist_name="")
    assert len(result["parties"]) == 2
    for party in result["parties"]:
        assert party["aliases"] == []


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
