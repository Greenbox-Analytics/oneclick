"""Dataclasses representing structured contract data extracted by the parser."""

from dataclasses import dataclass, field


@dataclass
class Party:
    name: str
    role: str
    # Alternate names the contract uses for this party (p/k/a, a/k/a, d/b/a,
    # stage names). The primary/legal name stays in `name`.
    aliases: list[str] = field(default_factory=list)


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
    # Income basis this party is paid on: "net" (after track expenses), "gross"
    # (full earnings), or None when the contract is silent for this party.
    basis: str | None = None
    # IDs of the contract(s) that assert this share. Populated by
    # calculate_payments_from_contract_ids and unioned across corroborating
    # shares in merge_contracts; empty in single-contract mode.
    source_contract_ids: list[str] = field(default_factory=list)


@dataclass
class ContractData:
    parties: list[Party]
    works: list[Work]
    royalty_shares: list[RoyaltyShare]
    contract_summary: str | None = None
    # Contract-wide income basis applied when a share doesn't state its own.
    default_basis: str | None = None


def effective_basis(share: RoyaltyShare, contract: ContractData) -> str:
    """Resolve the income basis for a share: its own basis, else the contract
    default, else "gross". Always returns "net" or "gross". Uses getattr so it
    tolerates lightweight share/contract stand-ins without the basis fields."""
    basis = getattr(share, "basis", None) or getattr(contract, "default_basis", None)
    return basis if basis in ("net", "gross") else "gross"
