"""Drift tests pin the internal shapes the partner DTOs map from. If one of
these fails, the internal dataclass changed: update partner_api/models.py's
mapping DELIBERATELY (v1 output must not change shape) and re-pin here."""

from dataclasses import asdict, fields

from oneclick.royalty_calculator import RoyaltyPayment
from partner_api.models import (
    PartnerContractTerms,
    payment_to_dto,
    to_contract_data,
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


def test_payment_to_dto_from_royalty_payment():
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
    dto = payment_to_dto(asdict(p))
    assert dto.amount_to_pay == 50.0
    assert dto.basis == "net"
    assert dto.net_amount == 80.0
    assert dto.terms is None


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
