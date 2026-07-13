"""Tests for registry.contract_splits.parse_royalty_splits.

Reuses the OpenAI-mock pattern from tests/test_utils_contract_parser.py so the
real LLM is never called. Locks the master/publishing pivoting + main-artist
detection behavior the Add Work wizard depends on.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from registry.contract_splits import parse_royalty_splits
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work


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


# ─── pre-parsed ContractData (shared cache) ───────────────────────────────────


def test_parse_royalty_splits_pivots_preparsed_without_llm():
    cd = ContractData(
        parties=[Party(name="Jane Doe", role="artist")],
        works=[Work(title="Midnight")],
        royalty_shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="Streaming", percentage=60.0)],
        contract_summary="",
        default_basis=None,
    )
    with patch("registry.contract_splits.MusicContractParser") as MockParser:
        out = parse_royalty_splits(contract_data=cd, main_artist_name="")
    MockParser.assert_not_called()
    assert out["parties"][0]["name"] == "Jane Doe"
    assert out["parties"][0]["master_pct"] == 60.0
    assert out["parties"][0]["publishing_pct"] == 0.0


def test_parse_contract_splits_endpoint_routes_through_cache(monkeypatch):
    """The Add-Work 'suggest splits' endpoint must go through the shared parse cache
    (get_or_parse), not an uncached parse_royalty_splits(pdf_path=...)."""
    import asyncio

    import main
    from registry import router as registry_router
    from utils.contract_parsing import cache as cache_mod
    from utils.contract_parsing.models import ContractData, Party, RoyaltyShare

    cd = ContractData(
        parties=[Party(name="Marcus", role="producer")],
        works=[],
        royalty_shares=[RoyaltyShare(party_name="Marcus", royalty_type="Master", percentage=40.0)],
        contract_summary="",
        default_basis=None,
    )

    fake_db = MagicMock()
    (
        fake_db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value
    ) = MagicMock(data={"file_path": "c/a.pdf", "file_name": "a.pdf", "contract_markdown": "stored canonical text"})

    monkeypatch.setattr(registry_router, "gated_feature", lambda *a, **k: None)
    monkeypatch.setattr(registry_router, "_get_supabase", lambda: fake_db)
    monkeypatch.setattr(registry_router, "analytics_capture", lambda *a, **k: None)
    monkeypatch.setattr(main, "verify_user_owns_contract", lambda *a, **k: True)

    gop = MagicMock(return_value=cd)
    monkeypatch.setattr(cache_mod, "get_or_parse", gop)

    result = asyncio.run(
        registry_router.parse_contract_splits(main_artist_name="", contract_file_id="f1", file=None, user_id="u1")
    )

    # The endpoint parsed via the shared cache...
    gop.assert_called_once()
    # ...feeding it a loader that prefers the stored canonical markdown (no PDF download).
    loader = gop.call_args.args[1]
    assert loader() == "stored canonical text"
    # ...and pivoted the cached ContractData into per-party splits.
    assert result["parties"][0]["name"] == "Marcus"
    assert result["parties"][0]["master_pct"] == 40.0
