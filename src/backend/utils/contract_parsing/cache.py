"""Shared cache of parsed ContractData, keyed by the canonical parse text.

Both Registry (Add Work) and OneClick parse the same contracts. The extraction is a
pure function of the (canonical, marker-stripped) contract markdown — see
MusicContractParser.parse_contract, whose args except full_text are unused — so a
cached result is reusable across tools and across runs.

The cache key is the SHA-256 of the canonical text actually fed to the parser (page
markers stripped), NOT the source file bytes. Keying on the exact parser input makes
each entry single-valued by construction: if two callers ever derive different markdown
from the same PDF (e.g. after a pymupdf4llm upgrade), they compute different keys and can
never serve each other a mismatched parse. The parser_version folds in the extraction
prompt version and the OpenAI model, so a prompt or model change invalidates cleanly.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import asdict

from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work
from utils.contract_parsing.parser import LLM_MODEL_LARGE, MusicContractParser
from utils.ingestion.pdf_markdown import strip_page_markers

logger = logging.getLogger(__name__)

# Bump when the extraction PROMPT changes. The OpenAI model is appended automatically,
# so changing OPENAI_LLM_MODEL_LARGE also invalidates cached entries.
PARSER_PROMPT_VERSION = "v1-2026-07-12"


def parser_version() -> str:
    """Cache-invalidation key component: prompt version + active extraction model."""
    return f"{PARSER_PROMPT_VERSION}:{LLM_MODEL_LARGE}"


def serialize_contract_data(cd: ContractData) -> dict:
    """ContractData -> plain JSON-safe dict (nested dataclasses become dicts)."""
    return asdict(cd)


# Top-level ContractData fields the cache knows how to reconstruct. A cached payload with
# any other key means the schema changed without this function being updated.
_CONTRACT_DATA_FIELDS = {"parties", "works", "royalty_shares", "contract_summary", "default_basis"}


def deserialize_contract_data(d: dict) -> ContractData:
    """Plain dict (as stored) -> ContractData. Rejects unknown top-level fields so a schema
    change that forgets to update this function fails loudly (the caller's best-effort read
    handler turns it into a live re-parse) instead of silently dropping the new field. Bump
    PARSER_PROMPT_VERSION on any schema change."""
    unknown = set(d) - _CONTRACT_DATA_FIELDS
    if unknown:
        raise ValueError(
            f"Unexpected ContractData fields in cached payload: {sorted(unknown)}. "
            "Update deserialize_contract_data and bump PARSER_PROMPT_VERSION on schema changes."
        )
    return ContractData(
        parties=[Party(**p) for p in d.get("parties", [])],
        works=[Work(**w) for w in d.get("works", [])],
        royalty_shares=[RoyaltyShare(**s) for s in d.get("royalty_shares", [])],
        contract_summary=d.get("contract_summary"),
        default_basis=d.get("default_basis"),
    )


def get_or_parse(
    db,
    load_text: Callable[[], str],
    *,
    parser: MusicContractParser | None = None,
    bypass: bool = False,
) -> ContractData:
    """Return parsed ContractData for a contract, using contract_parse_cache.

    The cache key is the SHA-256 of the CANONICAL parse text — the contract markdown with
    [[PAGE n]] navigation markers stripped — NOT the source file bytes. Keying on the exact
    text the parser sees makes the entry single-valued by construction. `load_text` is
    therefore always invoked (its result is needed to compute the key), so callers should
    make it cheap on the hot path (e.g. return already-fetched markdown rather than
    re-downloading the source PDF).

    Args:
        db: Supabase (service-role) client, or None to skip caching entirely.
        load_text: Zero-arg callable returning the contract markdown.
        parser: Optional MusicContractParser (constructed if omitted).
        bypass: If True, ignore any cached entry but still write the fresh result back.
            Reserved for a future explicit re-parse flag; NOT wired to force_recalculate.

    Cache I/O is best-effort: a read or write failure never propagates — the caller always
    gets a valid ContractData (falling back to a live parse).
    """
    version = parser_version()

    # Canonicalize first (strip [[PAGE n]] markers, idempotent on already-stripped text),
    # then key on the hash of that exact text so the entry is single-valued regardless of
    # which tool produced the markdown.
    full_text = strip_page_markers(load_text())
    if not full_text:
        raise ValueError("full_text is required. The contract markdown must be provided.")

    cache_key = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    if db is not None and not bypass:
        try:
            hit = (
                db.table("contract_parse_cache")
                .select("parsed")
                .eq("content_hash", cache_key)
                .eq("parser_version", version)
                .maybe_single()
                .execute()
            )
            if hit and hit.data:
                return deserialize_contract_data(hit.data["parsed"])
        except Exception:
            logger.exception("parse cache read failed; falling back to live parse")

    parser = parser or MusicContractParser()
    contract_data = parser.parse_contract(full_text=full_text)

    if db is not None:
        try:
            db.table("contract_parse_cache").upsert(
                {
                    "content_hash": cache_key,
                    "parser_version": version,
                    "parsed": serialize_contract_data(contract_data),
                },
                on_conflict="content_hash,parser_version",
            ).execute()
        except Exception:
            logger.exception("parse cache write failed; continuing without caching")

    return contract_data
