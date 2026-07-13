"""Pivot a parsed music contract into per-party master/publishing splits.

Used by the Add Work wizard. Wraps `MusicContractParser` so the UI gets the
shape it needs:
    {parties: [{name, role, aliases, master_pct, publishing_pct,
                soundexchange_pct, is_main_artist}],
     main_artist_found: bool}

SoundExchange (US non-interactive digital performance) shares are kept in
their own bucket — they are paid directly by SoundExchange and must never be
folded into the master total.

Only parties holding a master, publishing, or SoundExchange share are
returned; parties merely mentioned in the contract are filtered out (except
the main artist).
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

# Royalty type buckets. Anything not matching any bucket is dropped.
_MASTER_TYPES = {"master", "streaming", "producer", "mixer", "remixer"}
_PUBLISHING_TYPES = {"publishing", "composition", "songwriter", "writer", "mechanical", "sync"}
_SOUNDEXCHANGE_TYPES = {"soundexchange", "sound exchange"}
_SOUNDEXCHANGE_SUBSTRINGS = ("soundexchange", "sound exchange", "non-interactive digital performance")
_STREAMING_LOWER = [t.lower() for t in STREAMING_EQUIVALENT_TERMS]


def _bucket_for(royalty_type: str) -> str | None:
    """Return 'master' / 'publishing' / 'soundexchange' / None for a royalty_type string."""
    rt = (royalty_type or "").lower().strip()
    if not rt:
        return None
    if rt in _SOUNDEXCHANGE_TYPES:
        return "soundexchange"
    if rt in _MASTER_TYPES:
        return "master"
    if rt in _PUBLISHING_TYPES:
        return "publishing"
    # SoundExchange substring check must run before the master allowlist —
    # STREAMING_EQUIVALENT_TERMS contains "non-interactive digital performance
    # royalties", which would otherwise swallow these into master.
    for term in _SOUNDEXCHANGE_SUBSTRINGS:
        if term in rt:
            return "soundexchange"
    # Fallback: substring match against the long master-side allowlist.
    for term in _STREAMING_LOWER:
        if term in rt:
            return "master"
    if "publish" in rt or "compos" in rt or "songwrit" in rt or "mechanical" in rt:
        return "publishing"
    return None


def matches_party_name(name: str, aliases: list[str] | None, target: str) -> bool:
    """Match a target name against a party's name and aliases.

    Exact normalized match first (name or any alias), then the fuzzy
    substring fallback in both directions. Shared by the splits pivot and
    `derive_service` collaborator matching.
    """
    if not target:
        return False
    b = normalize_name(target)
    if not b:
        return False
    candidates = [normalize_name(name)] + [normalize_name(a) for a in (aliases or [])]
    candidates = [c for c in candidates if c]
    if b in candidates:
        return True
    return any(c in b or b in c for c in candidates)


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

    Parties without any master or publishing share are omitted; the main
    artist is always kept (even at 0/0) so the UI can prefill the "You" row.

    Returns:
        {
          "parties": [
            {"name": str, "role": str, "aliases": list[str], "master_pct": float,
             "publishing_pct": float, "soundexchange_pct": float,
             "is_main_artist": bool},
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

    # Index parties by normalized name AND every normalized alias so royalty
    # shares can attach to them even when the LLM emits a slightly different
    # spelling — or the stage name — on the share row. `by_norm` may hold
    # several keys per party dict, so unique parties are tracked separately.
    by_norm: dict[str, dict] = {}
    unique_parties: list[dict] = []
    for p in contract_data.parties:
        key = normalize_name(p.name)
        if not key or key in by_norm:
            continue
        aliases = list(getattr(p, "aliases", []) or [])
        entry = {
            "name": p.name,
            "role": p.role,
            "aliases": aliases,
            "master_pct": 0.0,
            "publishing_pct": 0.0,
            "soundexchange_pct": 0.0,
        }
        by_norm[key] = entry
        unique_parties.append(entry)
        for alias in aliases:
            alias_key = normalize_name(alias)
            if alias_key and alias_key not in by_norm:
                by_norm[alias_key] = entry

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
                "aliases": [],
                "master_pct": 0.0,
                "publishing_pct": 0.0,
                "soundexchange_pct": 0.0,
            }
            unique_parties.append(target)
        target[f"{bucket}_pct"] += float(share.percentage or 0)

    # Mark the main artist + drop them if not actually found in the contract.
    # Parties with no royalty split at all are dropped (merely mentioned in the
    # contract — e.g. a label named as licensor); the main artist is kept even
    # at 0/0 because their row prefills the wizard's "You" row.
    parties_out: list[dict] = []
    main_artist_found = False
    for party in unique_parties:
        is_main = matches_party_name(party["name"], party["aliases"], main_artist_name)
        if is_main:
            main_artist_found = True
        elif party["master_pct"] <= 0 and party["publishing_pct"] <= 0 and party["soundexchange_pct"] <= 0:
            continue
        parties_out.append({**party, "is_main_artist": is_main})

    if not main_artist_found:
        # Drop any party flagged as main (none should be, but guard anyway).
        parties_out = [p for p in parties_out if not p["is_main_artist"]]

    return {"parties": parties_out, "main_artist_found": main_artist_found}
