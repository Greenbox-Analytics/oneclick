"""Dataclasses representing structured contract data extracted by the parser."""

from dataclasses import dataclass


@dataclass
class Party:
    name: str
    role: str


@dataclass
class Work:
    title: str
    work_type: str = "song"


@dataclass
class RoyaltyShare:
    party_name: str
    royalty_type: str
    percentage: float
    terms: str | None = None


@dataclass
class ContractData:
    parties: list[Party]
    works: list[Work]
    royalty_shares: list[RoyaltyShare]
    contract_summary: str | None = None
