"""Tests for utils.contract_parsing.parser.MusicContractParser.

Before the refactor, the parser had ZERO direct test coverage — it was the
heart of OneClick's contract extraction but exercised only end-to-end. Now
that it lives in `utils/` for future tools to call directly, lock its
contract here.

OpenAI is mocked using the same `_client_returning` pattern as
tests/test_nuance_adjuster.py:14-19.
"""

import json
from unittest.mock import MagicMock

import pytest

from utils.contract_parsing.models import ContractData
from utils.contract_parsing.parser import MusicContractParser

# ─── helpers ────────────────────────────────────────────────────────────────


def _client_returning(payload: dict):
    """Build a mock OpenAI client whose chat.completions.create returns
    `payload` JSON-encoded — mirrors test_nuance_adjuster._client_returning."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    return client


def _parser_with_client(monkeypatch, openai_client):
    """Construct a MusicContractParser whose shared OpenAI client is mocked."""
    import utils.contract_parsing.parser as parser_module

    monkeypatch.setattr(parser_module, "get_openai_client", lambda: openai_client)
    return MusicContractParser(api_key="sk-test")


# ─── __init__ guards ────────────────────────────────────────────────────────


def test_init_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing or invalid OpenAI API key"):
        MusicContractParser(api_key=None)


def test_init_rejects_invalid_api_key_shape(monkeypatch):
    """Locks the historical fail-fast 'sk-' guard that OneClick relies on."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing or invalid OpenAI API key"):
        MusicContractParser(api_key="bad-key")


def test_init_accepts_valid_key_and_wires_shared_client(monkeypatch):
    import utils.contract_parsing.parser as parser_module

    sentinel = MagicMock(name="sentinel_openai_client")
    monkeypatch.setattr(parser_module, "get_openai_client", lambda: sentinel)

    parser = MusicContractParser(api_key="sk-test")
    assert parser.openai_client is sentinel
    assert parser.api_key == "sk-test"


# ─── parse_contract input validation ────────────────────────────────────────


def test_parse_contract_requires_full_text(monkeypatch):
    parser = _parser_with_client(monkeypatch, MagicMock())
    with pytest.raises(ValueError, match="full_text is required"):
        parser.parse_contract(full_text="")


def test_parse_contract_requires_non_none_full_text(monkeypatch):
    parser = _parser_with_client(monkeypatch, MagicMock())
    with pytest.raises(ValueError, match="full_text is required"):
        parser.parse_contract(full_text=None)


# ─── parse_contract — happy path ────────────────────────────────────────────


def test_parse_contract_returns_structured_data(monkeypatch):
    payload = {
        "parties": [{"name": "Alice Cooper", "role": "Artist"}],
        "works": [{"title": "Blue Sky", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Alice Cooper", "royalty_type": "Streaming", "percentage": 50.0}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="some contract markdown")

    assert isinstance(result, ContractData)
    assert len(result.parties) == 1
    assert result.parties[0].name == "Alice Cooper"
    assert len(result.works) == 1
    assert result.works[0].title == "Blue Sky"
    assert len(result.royalty_shares) == 1
    assert result.royalty_shares[0].percentage == 50.0


def test_parse_contract_passes_soundexchange_type_through_lowercased(monkeypatch):
    """A 'SoundExchange' share survives extraction as royalty_type
    'soundexchange' so the registry pivot can bucket it separately."""
    payload = {
        "parties": [{"name": "Alice Cooper", "role": "Artist"}],
        "works": [{"title": "Blue Sky", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Alice Cooper", "royalty_type": "SoundExchange", "percentage": 45.0}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].royalty_type == "soundexchange"
    assert result.royalty_shares[0].percentage == 45.0


# ─── post-processing behavior ───────────────────────────────────────────────


def test_parse_contract_simplifies_roles(monkeypatch):
    """The parser lowercases roles in _extract_all_unified, then applies
    simplify_role in parse_contract. 'Songwriter' → 'writer'."""
    payload = {
        "parties": [{"name": "Alice", "role": "Songwriter"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.parties[0].role == "writer"


def test_parse_contract_reconciles_share_party_name_to_canonical(monkeypatch):
    """If the LLM uses different casing in a share's party_name vs the parties
    list, the parser rewrites the share's party_name to the canonical form."""
    payload = {
        "parties": [{"name": "Alice Cooper", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [{"party_name": "alice cooper", "royalty_type": "Streaming", "percentage": 50.0}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].party_name == "Alice Cooper"


def test_parse_contract_deduplicates_parties_by_normalized_name(monkeypatch):
    """Two LLM entries that normalize to the same name → one Party."""
    payload = {
        "parties": [
            {"name": "Alice", "role": "Producer"},
            {"name": "ALICE", "role": "Producer"},
        ],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert len(result.parties) == 1


def test_parse_contract_deduplicates_works_by_normalized_title(monkeypatch):
    """'Blue Sky' and 'Blue Sky (Remix)' both normalize to 'blue sky' → one Work."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [
            {"title": "Blue Sky", "work_type": "song"},
            {"title": "Blue Sky (Remix)", "work_type": "song"},
        ],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert len(result.works) == 1


# ─── alias extraction ────────────────────────────────────────────────────────


def test_parse_contract_captures_party_aliases(monkeypatch):
    payload = {
        "parties": [{"name": "Jane Q. Doe", "role": "Artist", "aliases": ["Nova Sky"]}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.parties[0].name == "Jane Q. Doe"
    assert result.parties[0].aliases == ["Nova Sky"]


def test_parse_contract_aliases_default_to_empty_list(monkeypatch):
    """LLM output without an aliases field still yields Party.aliases == []."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.parties[0].aliases == []


def test_parse_contract_filters_garbage_aliases(monkeypatch):
    """Non-strings, blanks, duplicates-of-name, and repeat aliases are dropped."""
    payload = {
        "parties": [
            {
                "name": "Jane Doe",
                "role": "Artist",
                "aliases": ["Nova Sky", "", None, 42, "jane doe", "NOVA SKY"],
            }
        ],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.parties[0].aliases == ["Nova Sky"]


def test_parse_contract_splits_alias_markers_left_in_name(monkeypatch):
    """Fallback for LLM non-compliance: 'Legal p/k/a Stage' in the name field
    is split into name + alias deterministically."""
    payload = {
        "parties": [{"name": "Jane Doe p/k/a Nova Sky", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.parties[0].name == "Jane Doe"
    assert result.parties[0].aliases == ["Nova Sky"]


def test_parse_contract_duplicate_parties_union_aliases(monkeypatch):
    """The same party listed twice keeps one entry with the union of aliases."""
    payload = {
        "parties": [
            {"name": "Jane Doe", "role": "Artist", "aliases": ["Nova Sky"]},
            {"name": "JANE DOE", "role": "Artist", "aliases": ["Nova"]},
        ],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert len(result.parties) == 1
    assert result.parties[0].aliases == ["Nova Sky", "Nova"]


def test_parse_contract_reconciles_share_party_name_via_alias(monkeypatch):
    """A share emitted under a stage name is reconciled to the legal-name party."""
    payload = {
        "parties": [{"name": "Jane Doe", "role": "Artist", "aliases": ["Nova Sky"]}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Nova Sky", "royalty_type": "Streaming", "percentage": 50.0}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].party_name == "Jane Doe"


def test_split_alias_markers_variants():
    """The marker splitter handles the common alias notations."""
    from utils.text.normalize import split_alias_markers

    assert split_alias_markers("Jane Doe p/k/a Nova Sky") == ("Jane Doe", ["Nova Sky"])
    assert split_alias_markers("Jane Doe a.k.a. Nova Sky") == ("Jane Doe", ["Nova Sky"])
    assert split_alias_markers("Jane Doe, professionally known as Nova Sky") == ("Jane Doe", ["Nova Sky"])
    assert split_alias_markers("Acme Ltd d/b/a Neon Records") == ("Acme Ltd", ["Neon Records"])
    assert split_alias_markers("Jane Doe") == ("Jane Doe", [])
    assert split_alias_markers("") == ("", [])


# ─── parse_contract — defensive handling of malformed LLM output ────────────


def test_parse_contract_skips_share_without_percentage(monkeypatch):
    """A share with percentage=None must be dropped, not crash."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Alice", "royalty_type": "Streaming", "percentage": None}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares == []


def test_parse_contract_handles_non_numeric_percentage_gracefully(monkeypatch):
    """A share whose percentage isn't float-convertible is silently dropped."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Alice", "royalty_type": "Streaming", "percentage": "not-a-number"}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")  # no exception
    assert result.royalty_shares == []


def test_parse_contract_skips_parties_missing_name_or_role(monkeypatch):
    """Parties without both name AND role are dropped, no crash."""
    payload = {
        "parties": [
            {"name": "Alice", "role": "Artist"},
            {"name": "", "role": "Producer"},  # missing name
            {"name": "Bob", "role": ""},  # missing role
        ],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert len(result.parties) == 1
    assert result.parties[0].name == "Alice"


def test_parse_contract_propagates_terms_field(monkeypatch):
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [
            {
                "party_name": "Alice",
                "royalty_type": "Streaming",
                "percentage": 50.0,
                "terms": "recoupable from advance",
            }
        ],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].terms == "recoupable from advance"


def test_parse_contract_normalizes_null_terms_to_none(monkeypatch):
    """The string 'null' (a common LLM hallucination of a null) becomes None."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [
            {
                "party_name": "Alice",
                "royalty_type": "Streaming",
                "percentage": 50.0,
                "terms": "null",
            }
        ],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].terms is None


# ─── income basis extraction (net vs gross) ─────────────────────────────────


def test_parse_contract_extracts_per_share_net_basis(monkeypatch):
    """A share marked net in the contract is captured as basis='net'."""
    payload = {
        "parties": [{"name": "Alice", "role": "Producer"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Alice", "royalty_type": "Producer", "percentage": 20.0, "basis": "net"}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].basis == "net"


def test_parse_contract_basis_silent_is_none(monkeypatch):
    """When the contract is silent, share.basis and default_basis are None."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Alice", "royalty_type": "Streaming", "percentage": 50.0}],
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].basis is None
    assert result.default_basis is None


def test_parse_contract_captures_contract_default_basis(monkeypatch):
    """A contract-wide net term lands in default_basis."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [{"party_name": "Alice", "royalty_type": "Streaming", "percentage": 50.0}],
        "default_basis": "net",
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.default_basis == "net"


def test_parse_contract_rejects_garbage_basis(monkeypatch):
    """An out-of-vocabulary basis value is normalized to None."""
    payload = {
        "parties": [{"name": "Alice", "role": "Artist"}],
        "works": [{"title": "X", "work_type": "song"}],
        "royalty_shares": [
            {"party_name": "Alice", "royalty_type": "Streaming", "percentage": 50.0, "basis": "wholesale"}
        ],
        "default_basis": "maybe",
    }
    parser = _parser_with_client(monkeypatch, _client_returning(payload))

    result = parser.parse_contract(full_text="markdown")

    assert result.royalty_shares[0].basis is None
    assert result.default_basis is None


def test_effective_basis_resolution_chain(monkeypatch):
    """effective_basis: share basis wins, then contract default, then gross."""
    from utils.contract_parsing.models import ContractData, RoyaltyShare, effective_basis

    net_share = RoyaltyShare("A", "streaming", 50.0, basis="net")
    bare_share = RoyaltyShare("B", "streaming", 50.0)

    contract_default_net = ContractData([], [], [], default_basis="net")
    contract_silent = ContractData([], [], [])

    assert effective_basis(net_share, contract_silent) == "net"
    assert effective_basis(bare_share, contract_default_net) == "net"
    assert effective_basis(bare_share, contract_silent) == "gross"
