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
    # Optional grouping. Validated against the org at mint time, not here: a
    # well-formed id for another org's folder is still unknown.
    folder_id: str | None = None


class KeyFolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def _stripped(cls, v: str) -> str:
        # min_length sees the raw string, so "   " passes it. Strip and
        # re-check: the service stores the stripped name, and a whitespace-only
        # one must 422 at the edge rather than raise deeper in.
        v = v.strip()
        if not 1 <= len(v) <= 80:
            raise ValueError("name must be 1-80 characters")
        return v


class KeyFolderAssign(BaseModel):
    """None = unfile the key."""

    folder_id: str | None = None


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


class PartnerPayee(BaseModel):
    name: str
    role: str


class PartnerShare(BaseModel):
    type: str
    percentage: float
    basis: str


class PartnerAmounts(BaseModel):
    gross: float
    expenses: float
    net: float
    payable: float


class PartnerPayment(BaseModel):
    """One line of a calculation — the song, who is paid, on what share, and
    the money — sectioned so a partner reads it without a field legend."""

    song: str
    payee: PartnerPayee
    share: PartnerShare
    amounts: PartnerAmounts


class PartnerCalcSummary(BaseModel):
    payments: int
    total_payable: float
    expense_review_required: bool


class PartnerCalcResult(BaseModel):
    summary: PartnerCalcSummary
    payments: list[PartnerPayment]


class PartnerSplitParty(BaseModel):
    name: str
    role: str
    master_pct: float
    publishing_pct: float
    soundexchange_pct: float


class PartnerSplits(BaseModel):
    main_artist: str | None
    parties: list[PartnerSplitParty]


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
    """RoyaltyPayment (as dict, via dataclasses.asdict) -> the sectioned v1
    payment. The internal free-text `terms`, `total_royalty` and
    `source_contract_ids` are dropped: the clause is returned by
    /registry/v1/splits under contract_terms.royalty_shares[].terms."""
    return PartnerPayment(
        song=p["song_title"],
        payee=PartnerPayee(name=p["party_name"], role=p["role"]),
        share=PartnerShare(type=p["royalty_type"], percentage=p["percentage"], basis=p["basis"]),
        amounts=PartnerAmounts(
            gross=round(p["gross_amount"], 2),
            expenses=round(p["expenses_applied"], 2),
            net=round(p["net_amount"], 2),
            payable=round(p["amount_to_pay"], 2),
        ),
    )


def calc_result(payments: list[dict]) -> dict:
    """The royalties result event body, minus `type` and `billing` (the router
    adds those): a summary block, then the sectioned payments."""
    dtos = [payment_to_dto(p) for p in payments]
    summary = PartnerCalcSummary(
        payments=len(dtos),
        # Sum the already-rounded payables so a partner adding up the lines
        # they were shown gets exactly total_payable.
        total_payable=round(sum(d.amounts.payable for d in dtos), 2),
        expense_review_required=any(d.share.basis == "net" for d in dtos),
    )
    return PartnerCalcResult(summary=summary, payments=dtos).model_dump()


def to_partner_splits(pivot: dict) -> dict:
    """Registry pivot (contract_splits.parse_royalty_splits) -> the API's
    `splits` section. `main_artist` is the party the pivot flagged, by the name
    the contract uses, or None when the name sent was not found (or none was
    sent). `aliases` and the per-party flag are dropped. Percentages are
    indexed directly (not `.get(..., 0.0)`): a pivot missing one is a bug in
    the parse, and this is a money path — it must raise, not silently publish
    a 0.0 split."""
    parties = pivot.get("parties") or []
    return PartnerSplits(
        main_artist=next((p["name"] for p in parties if p.get("is_main_artist")), None),
        parties=[
            PartnerSplitParty(
                name=p["name"],
                role=p.get("role") or "",
                master_pct=p["master_pct"],
                publishing_pct=p["publishing_pct"],
                soundexchange_pct=p["soundexchange_pct"],
            )
            for p in parties
        ],
    ).model_dump()


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
