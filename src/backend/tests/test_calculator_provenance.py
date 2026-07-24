"""Provenance + per-work scoping through merge_contracts and payment calc."""

from oneclick.royalty_calculator import RoyaltyCalculator
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work


def _contract(cid, party, pct, work_title):
    return ContractData(
        parties=[Party(name=party, role="artist")],
        works=[Work(title=work_title, work_type="song")],
        royalty_shares=[
            RoyaltyShare(
                party_name=party,
                royalty_type="streaming",
                percentage=pct,
                terms=None,
                source_contract_ids=[cid],
            )
        ],
        contract_summary=None,
    )


def _bare_calculator() -> RoyaltyCalculator:
    """merge_contracts / _calculate_payments_from_data touch no instance state,
    so skip __init__ (which requires a real OpenAI key)."""
    return RoyaltyCalculator.__new__(RoyaltyCalculator)


def test_corroborating_shares_union_sources():
    calc = _bare_calculator()
    merged = calc.merge_contracts([_contract("K", "Romes", 30.0, "Home"), _contract("L", "Romes", 30.0, "Home")])
    romes = [s for s in merged.royalty_shares if s.party_name == "Romes"]
    assert len(romes) == 1
    assert set(romes[0].source_contract_ids) == {"K", "L"}


def test_conflicting_shares_both_kept_with_own_sources():
    calc = _bare_calculator()
    merged = calc.merge_contracts([_contract("K", "Romes", 30.0, "Home"), _contract("L", "Romes", 35.0, "Home")])
    romes = [s for s in merged.royalty_shares if s.party_name == "Romes"]
    assert len(romes) == 2
    assert {tuple(s.source_contract_ids) for s in romes} == {("K",), ("L",)}


def test_share_does_not_apply_to_other_contracts_work():
    calc = _bare_calculator()
    merged = calc.merge_contracts([_contract("K", "Kenji", 20.0, "Home"), _contract("Y", "Someone", 10.0, "Away")])
    work_sources = {"home": {"K"}, "away": {"Y"}}
    payments = calc._calculate_payments_from_data(
        merged, {"Home": 10.0, "Away": 10.0}, expenses=None, work_sources=work_sources
    )
    kenji = [p for p in payments if p.party_name == "Kenji"]
    assert {p.song_title for p in kenji} == {"Home"}  # never "Away"
    assert kenji[0].source_contract_ids == ["K"]
