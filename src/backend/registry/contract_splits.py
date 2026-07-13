"""Pivot a parsed music contract into per-party master/publishing splits.

Used by the Add Work wizard. Wraps `MusicContractParser` so the UI gets the
shape it needs:
    {parties: [{name, role, master_pct, publishing_pct, is_main_artist}],
     main_artist_found: bool}
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.contract_parsing.models import ContractData
from utils.contract_parsing.parser import STREAMING_EQUIVALENT_TERMS, MusicContractParser
from utils.ingestion.pdf_markdown import pdf_to_markdown
from utils.text.normalize import normalize_name

# Royalty type buckets. Anything not matching either is dropped.
_MASTER_TYPES = {"master", "streaming", "producer", "mixer", "remixer"}
_PUBLISHING_TYPES = {"publishing", "composition", "songwriter", "writer", "mechanical", "sync"}
_STREAMING_LOWER = [t.lower() for t in STREAMING_EQUIVALENT_TERMS]


def _bucket_for(royalty_type: str) -> str | None:
    """Return 'master' / 'publishing' / None for a royalty_type string."""
    rt = (royalty_type or "").lower().strip()
    if not rt:
        return None
    if rt in _MASTER_TYPES:
        return "master"
    if rt in _PUBLISHING_TYPES:
        return "publishing"
    # Fallback: substring match against the long master-side allowlist.
    for term in _STREAMING_LOWER:
        if term in rt:
            return "master"
    if "publish" in rt or "compos" in rt or "songwrit" in rt or "mechanical" in rt:
        return "publishing"
    return None


def _is_main_artist(party_name: str, main_artist_name: str) -> bool:
    """Fuzzy name compare — exact or substring, both directions."""
    if not main_artist_name:
        return False
    a = normalize_name(party_name)
    b = normalize_name(main_artist_name)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def parse_royalty_splits(
    *,
    text: str | None = None,
    pdf_path: str | None = None,
    main_artist_name: str = "",
    contract_data: ContractData | None = None,
) -> dict:
    """Parse a contract (markdown text OR a PDF path) and return per-party splits.

    Exactly one of `text` / `pdf_path` / `contract_data` must be provided. When
    `contract_data` is supplied (already parsed, e.g. from the shared cache), the LLM
    extraction is skipped and only the pivot runs.

    Returns:
        {
          "parties": [
            {"name": str, "role": str, "master_pct": float,
             "publishing_pct": float, "is_main_artist": bool},
            ...
          ],
          "main_artist_found": bool
        }
    """
    if contract_data is None:
        if not text and not pdf_path:
            raise ValueError("Either text or pdf_path must be provided.")

        if pdf_path:
            text = pdf_to_markdown(pdf_path)

        parser = MusicContractParser()
        contract_data = parser.parse_contract(text)

    # Index parties by normalized name so royalty shares can attach to them
    # even when the LLM emits a slightly different spelling on the share row.
    by_norm: dict[str, dict] = {}
    for p in contract_data.parties:
        key = normalize_name(p.name)
        if key and key not in by_norm:
            by_norm[key] = {
                "name": p.name,
                "role": p.role,
                "master_pct": 0.0,
                "publishing_pct": 0.0,
            }

    # Pivot royalty shares into the two buckets per party.
    for share in contract_data.royalty_shares:
        bucket = _bucket_for(share.royalty_type)
        if not bucket:
            continue
        key = normalize_name(share.party_name)
        if not key:
            continue
        # Attach to the matched party — fall back to substring match if no exact key hit.
        target = by_norm.get(key)
        if target is None:
            for k, party in by_norm.items():
                if k in key or key in k:
                    target = party
                    break
        if target is None:
            # Royalty mentions an unlisted party — register one on the fly.
            target = by_norm[key] = {
                "name": share.party_name,
                "role": "",
                "master_pct": 0.0,
                "publishing_pct": 0.0,
            }
        target[f"{bucket}_pct"] += float(share.percentage or 0)

    # Mark the main artist + drop them if not actually found in the contract.
    parties_out: list[dict] = []
    main_artist_found = False
    for party in by_norm.values():
        is_main = _is_main_artist(party["name"], main_artist_name)
        if is_main:
            main_artist_found = True
        party = {**party, "is_main_artist": is_main}
        parties_out.append(party)

    if not main_artist_found:
        # Drop any party flagged as main (none should be, but guard anyway).
        parties_out = [p for p in parties_out if not p["is_main_artist"]]

    return {"parties": parties_out, "main_artist_found": main_artist_found}
