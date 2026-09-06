"""Frozen v1 DTOs for the partner API — the public contract.

Internal shapes (ContractData / RoyaltyPayment) map to and from these
EXPLICITLY; test_partner_models.py pins the internal field sets so internal
drift fails a test instead of silently breaking partners. Breaking changes
go to /partner/v2, never into these models.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work

# ---- key-mint request base --------------------------------------------------


class ExpiringKeyCreate(BaseModel):
    """Base for the key-mint body: one place decides what a valid expiry is."""

    # Typed datetime, not str — a malformed string must 422 at the edge, not
    # 500 at the DB.
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def _future_and_aware(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        # A bare date ("2026-09-30") parses to NAIVE midnight. Treat naive as
        # UTC so the stored value is unambiguous, then refuse the past — a key
        # born expired is always a caller mistake, never an intent.
        aware = v if v.tzinfo else v.replace(tzinfo=UTC)
        if aware <= datetime.now(UTC):
            raise ValueError("expires_at must be in the future")
        return aware


class PartnerKeyCreate(ExpiringKeyCreate):
    """Body of every mint — the Msanii-admin endpoint and the org console.
    One model, two callers, so they can't drift."""

    label: str = Field(min_length=1, max_length=120)


# ---- request DTOs -----------------------------------------------------------


class PartnerParty(BaseModel):
    name: str
    role: str
    aliases: list[str] = []


class PartnerWork(BaseModel):
    title: str
    work_type: str = "song"


class PartnerRoyaltyShare(BaseModel):
    party_name: str
    royalty_type: str
    # Money path with no LLM in front of it — bound the input.
    percentage: float = Field(ge=0, le=100)
    terms: str | None = None
    basis: str | None = None  # "net" | "gross" | None (contract default applies)


class PartnerContractTerms(BaseModel):
    parties: list[PartnerParty]
    works: list[PartnerWork]
    royalty_shares: list[PartnerRoyaltyShare]
    contract_summary: str | None = None
    default_basis: str | None = None  # "net" | "gross"


class PartnerExpense(BaseModel):
    amount: float = Field(ge=0)
    description: str | None = None
    # Track titles this expense is tagged to; empty = project-wide.
    # allocate_expenses reads `work_titles` from each expense dict (royalty_calculator.py:127).
    work_titles: list[str] = []


class PartnerContributor(BaseModel):
    """One line of a split sheet — the product's ContributorInput, frozen."""

    name: str = Field(min_length=1, max_length=200)
    role: str = Field(min_length=1, max_length=100)
    # Publishing side — composition. A self-published writer keeps the whole
    # publishing_share; a published writer collects writer_share while their
    # publisher collects publisher_share.
    publishing_share: float | None = Field(default=None, ge=0, le=100)
    writer_share: float | None = Field(default=None, ge=0, le=100)
    publisher_share: float | None = Field(default=None, ge=0, le=100)
    ipi_number: str | None = None
    is_published: bool = False
    publisher_name: str | None = None
    publisher_ipi: str | None = None
    # Master side — sound recording.
    master_percentage: float | None = Field(default=None, ge=0, le=100)
    label: str | None = None


class PartnerSplitSheetRequest(BaseModel):
    work_title: str = Field(min_length=1, max_length=200)
    work_type: str = Field(default="single", max_length=40)
    split_type: Literal["publishing", "master", "both"] = "both"
    # Printed on the sheet verbatim, so the partner picks the wording.
    date: str = Field(min_length=1, max_length=40)
    format: Literal["pdf", "docx"] = "pdf"
    contributors: list[PartnerContributor] = Field(min_length=1, max_length=50)


# ---- response DTOs ----------------------------------------------------------


class PartnerPayment(BaseModel):
    song_title: str
    party_name: str
    role: str
    royalty_type: str
    percentage: float
    amount_to_pay: float
    basis: str
    gross_amount: float
    expenses_applied: float
    net_amount: float
    terms: str | None = None


class PartnerCalcResult(BaseModel):
    payments: list[PartnerPayment]
    total_payments: int
    expense_review_required: bool


# ---- mapping ----------------------------------------------------------------


def to_contract_data(terms: PartnerContractTerms) -> ContractData:
    """Partner DTO -> internal ContractData. source_contract_ids stays empty:
    partner-supplied terms have no stored-contract provenance."""
    return ContractData(
        parties=[Party(name=p.name, role=p.role, aliases=list(p.aliases)) for p in terms.parties],
        works=[Work(title=w.title, work_type=w.work_type) for w in terms.works],
        royalty_shares=[
            RoyaltyShare(
                party_name=s.party_name,
                royalty_type=s.royalty_type,
                percentage=s.percentage,
                terms=s.terms,
                basis=s.basis,
            )
            for s in terms.royalty_shares
        ],
        contract_summary=terms.contract_summary,
        default_basis=terms.default_basis,
    )


def payment_to_dto(p: dict) -> PartnerPayment:
    """RoyaltyPayment (as dict, via dataclasses.asdict) -> frozen v1 DTO.
    Field names match 1:1 (the drift test pins them); pydantic ignores the
    two internal extras (total_royalty, source_contract_ids)."""
    return PartnerPayment.model_validate(p)


def from_contract_data(cd: ContractData) -> PartnerContractTerms:
    """Internal ContractData -> the frozen v1 terms DTO. The SAME shape
    /oneclick/v1/royalties accepts as contract_terms, so a parse from
    /registry/v1/splits feeds a calculation with no translation.
    source_contract_ids is internal provenance and is dropped."""
    return PartnerContractTerms(
        parties=[PartnerParty(name=p.name, role=p.role, aliases=list(p.aliases)) for p in cd.parties],
        works=[PartnerWork(title=w.title, work_type=w.work_type) for w in cd.works],
        royalty_shares=[
            PartnerRoyaltyShare(
                party_name=s.party_name,
                royalty_type=s.royalty_type,
                percentage=s.percentage,
                terms=s.terms,
                basis=s.basis,
            )
            for s in cd.royalty_shares
        ],
        contract_summary=cd.contract_summary,
        default_basis=cd.default_basis,
    )
