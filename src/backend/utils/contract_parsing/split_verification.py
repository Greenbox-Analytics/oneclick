"""Advisory split verification (v1) — checks extracted royalty shares against the contract.

Second-pass confidence check for OneClick: the LLM is asked to BLINDLY report what the
contract says each party's royalty percentage/basis is (it never sees the extracted
values), and the verdict is computed here in Python. Advisory only — callers surface
the verdict in the UI; nothing in this module changes any payout.

Guardrails (in the safe direction — never a false green):
- a finding only counts when its clause quote is verbatim-present in the contract
  (same normalization as basis_detection) AND the reported percentage's digits appear
  in that quote;
- anything uncertain degrades to "unverified" per share or "unavailable" overall.

What this catches: values the model reads differently on a fresh blind pass, and
confirmations unsupported by the quoted clause. It does NOT catch: an error both
passes make identically; a party extraction OMITTED entirely (only extracted shares
are checked — a dropped collaborator is invisible here); and basis mismatches are
only flagged when the clause states net/gross explicitly, which many contracts
don't. It is a disagreement detector, not a proof of correctness — which is why the
UI copy says "extracted splits match the contract", never "the contract is fully
covered". Verdicts come from a reasoning model and may differ between recalculations
of the same contract; the verdict is stable only once confirmed/cached.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field

from utils.contract_parsing.basis_detection import LLM_MODEL_LARGE, _normalize
from utils.contract_parsing.models import ContractData, RoyaltyShare, effective_basis

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 45.0
_PCT_TOLERANCE = 0.01


@dataclass
class SplitFinding:
    party_name: str
    royalty_type: str
    extracted_percentage: float
    extracted_basis: str | None
    verdict: str  # "verified" | "mismatch" | "unverified"
    contract_percentage: float | None = None
    contract_basis: str | None = None
    contract_quote: str = ""
    note: str = ""


@dataclass
class ReviewResult:
    overall: str  # "verified" | "needs_review" | "unavailable"
    checked: int = 0
    flagged: int = 0
    findings: list[SplitFinding] = field(default_factory=list)


_PROMPT = """You are auditing a music contract. For EACH party / royalty-type pair listed below, find the \
clause in the contract that defines that party's royalty percentage for that royalty type, and report what \
the contract says. Do NOT judge, infer, or assume — only report what is explicitly written.

PAIRS TO LOOK UP:
{pairs}

Rules:
- contract_percentage: the numeric percentage the contract states for that party and royalty type, or null \
if no explicit percentage is stated. Royalty-type labels are broad categories (master royalties encompass \
streaming); do not treat wording differences between category labels as different royalty types.
- If the contract contains a conditional major-label override (e.g. an "Upstream Agreement"), report the \
baseline/independent royalty, NOT the override.
- Ignore royalties collected directly from SoundExchange, PROs, or the MLC when they are paid directly to a party.
- contract_basis: "net" if the clause pays on net receipts/income after deductions, "gross" if on gross \
receipts/all income, null if the clause does not say.
- contract_quote: quote the governing clause VERBATIM from the contract. Never paraphrase; never quote text \
that is not in the contract.
- In each finding, copy party_name and royalty_type EXACTLY as written in the PAIRS list above — do not \
expand, shorten, or normalize them.
- note: one short plain-English sentence describing what the contract says for this pair.

Return JSON ONLY:
{"findings": [{"party_name": string, "royalty_type": string, "contract_percentage": number or null, \
"contract_basis": "net" or "gross" or null, "contract_quote": string, "note": string}]}

CONTRACT:
{contract}
"""


def _pct_in_quote(percentage: float, quote: str) -> bool:
    """True when the percentage's digits appear in the quote as a standalone number
    followed by a percent marker (%, "percent", "per cent").

    50.0 -> "50" (trailing .0 stripped via %g) and must not match inside a longer
    number: "150%" does not confirm 50, and neither does "50.5%" (lookahead rejects
    a following digit or decimal-fraction). Requiring the percent marker means bare
    numbers like "within 50 days" or "Section 50" never confirm a rate.
    """
    digits = f"{percentage:g}"
    return (
        re.search(rf"(?<![\d.]){re.escape(digits)}(?!\d|\.\d)\s*(?:%|percent\b|per cent\b)", quote, re.IGNORECASE)
        is not None
    )


def _key(party_name: str, royalty_type: str) -> tuple[str, str]:
    return ((party_name or "").strip().lower(), (royalty_type or "").strip().lower())


def _match_reported(
    share: RoyaltyShare, reported_by_key: dict[tuple[str, str], dict], share_keys: set[tuple[str, str]]
) -> dict:
    """Find the model's report for a share. Exact normalized key first; then — because
    models sometimes expand/shorten names despite being told not to — the same
    containment fallback the parser's reconciliation uses, accepted only when
    unambiguous (mirrors utils/contract_parsing/parser.py name reconciliation).
    Two ambiguity guards: reports whose key exactly matches a DIFFERENT share are
    never fallback candidates (they belong to that share), and a candidate is
    accepted only when NO other share's key also containment-matches it — one fuzzy
    report must never verify two different shares."""
    key = _key(share.party_name, share.royalty_type)
    if key in reported_by_key:
        return reported_by_key[key]
    p_norm, t_norm = key
    candidates = [
        (cand_p, item)
        for (cand_p, cand_t), item in reported_by_key.items()
        if (cand_p, cand_t) not in share_keys and cand_t == t_norm and cand_p and (p_norm in cand_p or cand_p in p_norm)
    ]
    if len(candidates) != 1:
        return {}
    cand_p, item = candidates[0]
    for other_p, other_t in share_keys:
        if other_t == t_norm and other_p != p_norm and (other_p in cand_p or cand_p in other_p):
            return {}
    return item


def _judge(share: RoyaltyShare, contract: ContractData, reported: dict, contract_norm: str) -> SplitFinding:
    """Compute the verdict for one share from the model's blind report. Pure Python —
    the model never sees the extracted values and never decides the verdict."""
    extracted_effective = effective_basis(share, contract)
    finding = SplitFinding(
        party_name=share.party_name,
        royalty_type=share.royalty_type,
        extracted_percentage=share.percentage,
        extracted_basis=extracted_effective,
        verdict="unverified",
        note=str(reported.get("note") or ""),
    )
    if not reported:
        finding.note = "The verification pass returned no report for this split."
        return finding

    quote_raw = reported.get("contract_quote")
    quote = quote_raw.strip() if isinstance(quote_raw, str) else ""
    pct = reported.get("contract_percentage")
    basis_raw = reported.get("contract_basis")
    basis = basis_raw.strip().lower() if isinstance(basis_raw, str) else None
    finding.contract_quote = quote
    finding.contract_basis = basis if basis in ("net", "gross") else None
    if isinstance(pct, (int, float)) and not isinstance(pct, bool) and math.isfinite(pct):
        finding.contract_percentage = float(pct)

    # Guardrail 1: quote must be verbatim-present in the contract (>=12 chars, normalized).
    quote_norm = _normalize(quote)
    if not quote or len(quote_norm) < 12 or quote_norm not in contract_norm:
        finding.note = finding.note or "No verbatim supporting clause could be confirmed in the contract."
        return finding
    # Guardrail 2: the reported number must appear in the quoted clause.
    if finding.contract_percentage is None or not _pct_in_quote(finding.contract_percentage, quote):
        finding.note = finding.note or "The supporting clause does not state this percentage as a number."
        return finding

    pct_ok = abs(finding.contract_percentage - share.percentage) <= _PCT_TOLERANCE
    basis_mismatch = finding.contract_basis is not None and finding.contract_basis != extracted_effective
    finding.verdict = "mismatch" if (not pct_ok or basis_mismatch) else "verified"
    return finding


def verify_royalty_shares(
    contract_markdown: str,
    contract_data: ContractData,
    *,
    openai_client,
    search_fn=None,  # optional KB enrichment for notes ONLY — must never affect verdicts
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ReviewResult:
    """Blind-verify each extracted royalty share against the contract text.

    Returns a ReviewResult; overall is "unavailable" when the pass could not run
    (empty markdown, no shares, LLM error/timeout). Never raises.
    """
    shares = list(getattr(contract_data, "royalty_shares", None) or [])
    if not contract_markdown or not contract_markdown.strip() or not shares:
        return ReviewResult(overall="unavailable")

    # Blind projection: only (party, royalty_type) reach the model — never the
    # extracted percentage/basis (the values under test).
    pairs = "\n".join(f"- party: {s.party_name} | royalty type: {s.royalty_type}" for s in shares)
    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL_LARGE,
            messages=[
                {
                    "role": "user",
                    "content": _PROMPT.replace("{pairs}", pairs).replace("{contract}", contract_markdown),
                }
            ],
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:  # noqa: BLE001 — advisory pass must never break the calc
        logger.warning(f"[SplitVerify] verification unavailable: {e}")
        return ReviewResult(overall="unavailable")

    if not isinstance(data, dict):
        logger.warning("[SplitVerify] malformed verification output (not an object); unavailable")
        return ReviewResult(overall="unavailable")
    raw_findings = data.get("findings") or []
    if not isinstance(raw_findings, list):
        logger.warning("[SplitVerify] malformed verification output (findings not a list); unavailable")
        return ReviewResult(overall="unavailable")

    reported_by_key: dict[tuple[str, str], dict] = {}
    for item in raw_findings:
        if isinstance(item, dict):
            reported_by_key.setdefault(_key(item.get("party_name", ""), item.get("royalty_type", "")), item)

    contract_norm = _normalize(contract_markdown)
    share_keys = {_key(s.party_name, s.royalty_type) for s in shares}
    findings = [
        _judge(share, contract_data, _match_reported(share, reported_by_key, share_keys), contract_norm)
        for share in shares
    ]

    # Optional KB context for flagged notes — reviewer aid only, never the verdict.
    if search_fn is not None and any(f.verdict == "mismatch" for f in findings):
        try:
            passages = search_fn("royalty split percentage ownership shares", floor_count=0)
            if passages:
                ref = getattr(passages[0], "section_path", "")
                for f in findings:
                    if f.verdict == "mismatch" and ref:
                        f.note = (f.note + f" (See also: {ref}.)").strip()
        except Exception as e:  # noqa: BLE001 — book is optional enrichment
            logger.warning(f"[SplitVerify] reference lookup failed (verdicts unaffected): {e}")

    flagged = sum(1 for f in findings if f.verdict != "verified")
    return ReviewResult(
        overall="verified" if flagged == 0 else "needs_review",
        checked=len(findings),
        flagged=flagged,
        findings=findings,
    )
