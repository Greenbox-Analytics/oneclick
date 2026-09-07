"""Drift tests pin the internal shapes the partner DTOs map from. If one of
these fails, the internal dataclass changed: update partner_api/models.py's
mapping DELIBERATELY (v1 output must not change shape) and re-pin here."""

from dataclasses import asdict, fields

import pytest

from oneclick.royalty_calculator import RoyaltyPayment
from partner_api.models import (
    PartnerContractTerms,
    calc_result,
    payment_to_dto,
    to_contract_data,
    to_partner_splits,
)
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work


def _names(dc):
    return {f.name for f in fields(dc)}


def test_internal_shapes_are_pinned():
    assert _names(Party) == {"name", "role", "aliases"}
    assert _names(Work) == {"title", "work_type"}
    assert _names(RoyaltyShare) == {
        "party_name",
        "royalty_type",
        "percentage",
        "terms",
        "basis",
        "source_contract_ids",
    }
    assert _names(ContractData) == {
        "parties",
        "works",
        "royalty_shares",
        "contract_summary",
        "default_basis",
    }
    assert _names(RoyaltyPayment) == {
        "song_title",
        "party_name",
        "role",
        "royalty_type",
        "percentage",
        "total_royalty",
        "amount_to_pay",
        "terms",
        "basis",
        "gross_amount",
        "expenses_applied",
        "net_amount",
        "source_contract_ids",
    }


def test_to_contract_data_maps_all_partner_fields():
    terms = PartnerContractTerms(
        parties=[{"name": "Artist A", "role": "artist"}],
        works=[{"title": "Song One"}],
        royalty_shares=[
            {
                "party_name": "Artist A",
                "royalty_type": "Streaming",
                "percentage": 50.0,
                "basis": "net",
            }
        ],
        default_basis="gross",
    )
    cd = to_contract_data(terms)
    assert cd.parties[0].name == "Artist A"
    assert cd.works[0].title == "Song One"
    assert cd.royalty_shares[0].percentage == 50.0
    assert cd.royalty_shares[0].basis == "net"
    assert cd.royalty_shares[0].source_contract_ids == []
    assert cd.default_basis == "gross"


def test_payment_to_dto_sections_the_payment():
    p = RoyaltyPayment(
        song_title="Song One",
        party_name="Artist A",
        role="artist",
        royalty_type="Streaming",
        percentage=50.0,
        total_royalty=100.0,
        amount_to_pay=50.0,
        basis="net",
        gross_amount=100.0,
        expenses_applied=20.0,
        net_amount=80.0,
    )
    assert payment_to_dto(asdict(p)).model_dump() == {
        "song": "Song One",
        "payee": {"name": "Artist A", "role": "artist"},
        "share": {"type": "Streaming", "percentage": 50.0, "basis": "net"},
        "amounts": {"gross": 100.0, "expenses": 20.0, "net": 80.0, "payable": 50.0},
    }


def test_calc_result_summary_sums_payable_and_flags_net_basis():
    a = {
        "song_title": "A",
        "party_name": "P",
        "role": "artist",
        "royalty_type": "master",
        "percentage": 50.0,
        "amount_to_pay": 400.004,
        "basis": "net",
        "gross_amount": 1000.0,
        "expenses_applied": 200.0,
        "net_amount": 800.0,
        "terms": None,
        "total_royalty": 0.0,
        "source_contract_ids": [],
    }
    b = {**a, "song_title": "B", "amount_to_pay": 200.0, "basis": "gross", "expenses_applied": 0.0}
    out = calc_result([a, b])
    assert out["summary"] == {"payments": 2, "total_payable": 600.0, "expense_review_required": True}
    assert [p["song"] for p in out["payments"]] == ["A", "B"]
    assert out["payments"][0]["amounts"]["payable"] == 400.0
    assert "terms" not in out["payments"][0]
    assert calc_result([])["summary"] == {"payments": 0, "total_payable": 0.0, "expense_review_required": False}


def test_calc_result_total_is_the_sum_of_the_rounded_lines():
    # Three lines of 33.335 show as 33.34 each; the total a partner adds up from
    # the shown lines is 100.02 — never the 100.0 a raw sum would give.
    line = {
        "song_title": "A",
        "party_name": "P",
        "role": "artist",
        "royalty_type": "master",
        "percentage": 33.335,
        "amount_to_pay": 33.335,
        "basis": "gross",
        "gross_amount": 100.0,
        "expenses_applied": 0.0,
        "net_amount": 100.0,
        "terms": None,
        "total_royalty": 0.0,
        "source_contract_ids": [],
    }
    out = calc_result([line, {**line, "song_title": "B"}, {**line, "song_title": "C"}])
    assert [p["amounts"]["payable"] for p in out["payments"]] == [33.34, 33.34, 33.34]
    assert out["summary"]["total_payable"] == 100.02


def test_to_partner_splits_names_the_main_artist_and_drops_flags():
    pivot = {
        "parties": [
            {
                "name": "Jane Doe",
                "role": "producer",
                "aliases": ["JD"],
                "master_pct": 50.0,
                "publishing_pct": 0.0,
                "soundexchange_pct": 0.0,
                "is_main_artist": False,
            },
            {
                "name": "Sam Ray",
                "role": "artist",
                "aliases": [],
                "master_pct": 50.0,
                "publishing_pct": 100.0,
                "soundexchange_pct": 10.0,
                "is_main_artist": True,
            },
        ],
        "main_artist_found": True,
    }
    assert to_partner_splits(pivot) == {
        "main_artist": "Sam Ray",
        "parties": [
            {
                "name": "Jane Doe",
                "role": "producer",
                "master_pct": 50.0,
                "publishing_pct": 0.0,
                "soundexchange_pct": 0.0,
            },
            {
                "name": "Sam Ray",
                "role": "artist",
                "master_pct": 50.0,
                "publishing_pct": 100.0,
                "soundexchange_pct": 10.0,
            },
        ],
    }
    assert to_partner_splits({"parties": [], "main_artist_found": False}) == {"main_artist": None, "parties": []}


def test_to_partner_splits_raises_on_missing_percentage():
    pivot = {
        "parties": [{"name": "Jane Doe", "role": "producer", "is_main_artist": False}],
        "main_artist_found": False,
    }
    with pytest.raises(KeyError):
        to_partner_splits(pivot)


def test_to_partner_splits_main_artist_none_when_no_party_flagged():
    pivot = {
        "parties": [
            {
                "name": "Jane Doe",
                "role": "producer",
                "master_pct": 100.0,
                "publishing_pct": 0.0,
                "soundexchange_pct": 0.0,
                "is_main_artist": False,
            }
        ],
        "main_artist_found": True,
    }
    assert to_partner_splits(pivot)["main_artist"] is None


def test_partner_key_create_shapes():
    import pytest
    from pydantic import ValidationError

    from partner_api.models import PartnerKeyCreate

    key = PartnerKeyCreate(label="Production backend")
    assert key.expires_at is None
    assert not hasattr(key, "user_ref")  # one key type — no hierarchy fields
    with pytest.raises(ValidationError):
        PartnerKeyCreate(label="")
    with pytest.raises(ValidationError):  # inherited from ExpiringKeyCreate
        PartnerKeyCreate(label="x", expires_at="2020-01-01")


def test_expiring_key_create_rejects_past_and_normalizes_naive_to_utc():
    import pytest
    from pydantic import ValidationError

    from partner_api.models import ExpiringKeyCreate

    with pytest.raises(ValidationError):
        ExpiringKeyCreate(expires_at="2020-01-01")
    with pytest.raises(ValidationError):
        ExpiringKeyCreate(expires_at="2020-01-01T00:00:00Z")
    ok = ExpiringKeyCreate(expires_at="2099-01-01")  # bare date -> naive midnight -> UTC
    assert ok.expires_at.tzinfo is not None
    assert ok.expires_at.utcoffset().total_seconds() == 0
    assert ExpiringKeyCreate().expires_at is None
