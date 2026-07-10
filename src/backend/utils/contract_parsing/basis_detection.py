"""Contract-nuance basis DETECTION (log-only v1) — originally OneClick-specific.

The OneClick calc applies a flat percentage to each song's net-payable total. Some
contracts describe a different base (net receipts, a stated deduction, wholesale vs
retail). This module DETECTS such an explicit clause and logs it for human review.
It does NOT change any payout — the statement base may already be net of the
deduction, and the calc only emits streaming lines, so auto-adjustment is unsafe in v1.

Guardrails: a finding requires a clause that is verbatim-present in the contract
(quote/dash normalized); the book only enriches the human note; anything uncertain → None.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

from analytics import capture as analytics_capture

logger = logging.getLogger(__name__)
# Logger name is load-bearing: tests use `caplog.at_level(logger="oneclick.audit")`.
# Keep the string literal even though this module now lives outside oneclick/.
audit_logger = logging.getLogger("oneclick.audit")

LLM_MODEL_LARGE = os.getenv("OPENAI_LLM_MODEL_LARGE", "gpt-5.2")

_WS = re.compile(r"\s+")
# PDF→markdown artifacts: curly quotes → straight, en/em dash → hyphen, soft hyphen → removed.
_PDF_ARTIFACTS = {0x2018: 0x27, 0x2019: 0x27, 0x201C: 0x22, 0x201D: 0x22, 0x2013: 0x2D, 0x2014: 0x2D, 0x00AD: None}


@dataclass
class BasisFinding:
    basis: str  # human description, e.g. "Net Receipts after 25% packaging deduction"
    implied_factor: float  # what the clause WOULD imply (0<f<=1); informational only, never applied
    clause_quote: str  # verbatim text from the contract
    affected_types: list[str] | str  # royalty types the LLM believes are affected (informational)
    kb_reference: str = ""  # book passage(s) for the reviewer
    kb_context: str = field(default="", repr=False)


# NOTE: split_verification.py imports _normalize for its verbatim-quote guardrail — behavior changes affect both modules' false-green defenses.
def _normalize(text: str) -> str:
    # Fold PDF artifacts, then reconcile spacing around hyphens ("a — b" / "a-b" -> "a-b")
    # so an em-dash with/without surrounding spaces doesn't defeat the verbatim check.
    text = re.sub(r"\s*-\s*", "-", text.translate(_PDF_ARTIFACTS))
    return _WS.sub(" ", text).strip().lower()


_PROMPT = """You audit a music contract for ONE narrow question: does an EXPLICIT clause change the BASE \
on which royalties are paid, vs paying the stated percentage on the full statement total?

In scope (ONLY these): net-receipts vs gross, a stated percentage deduction (packaging, free goods, reserve), \
wholesale vs retail base. OUT of scope: recoupment, advances, escalating/conditional rates — ignore those.

The royalty types present for this contract are: {types}. If you can tell which of these the basis clause \
affects, list them using these exact strings; otherwise use "all".

Rules:
- Only report a clause EXPLICITLY in the contract. Quote it verbatim. NEVER infer or assume a deduction.
- implied_factor = the fraction of the base that remains (25% packaging deduction -> 0.75; net of 10% fee -> 0.90).
- If no explicit in-scope clause exists, return applies=false.

Return JSON ONLY: {"applies": bool, "implied_factor": number, "basis": string, "clause_quote": string, \
"affected_types": "all" or [list of the royalty-type strings above]}

CONTRACT:
{contract}
"""


def audit_contract_basis(
    contract_markdown: str,
    royalty_types_present: list[str],
    *,
    openai_client,
    search_fn,
) -> BasisFinding | None:
    """Detect an explicit basis clause and return a verified BasisFinding, or None."""
    if not contract_markdown or not contract_markdown.strip():
        return None

    try:
        prompt = _PROMPT.replace("{types}", ", ".join(royalty_types_present) or "(none)").replace(
            "{contract}", contract_markdown
        )
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL_LARGE,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:  # noqa: BLE001 — never raise into the calc path
        logger.warning(f"[NuanceAudit] detection failed (no finding): {e}")
        return None

    if not data.get("applies"):
        return None

    factor = data.get("implied_factor")
    quote = (data.get("clause_quote") or "").strip()
    if not isinstance(factor, (int, float)) or not (0 < factor <= 1):
        logger.warning(f"[NuanceAudit] rejected out-of-range implied_factor={factor!r}")
        return None
    if not quote or len(quote) < 12 or _normalize(quote) not in _normalize(contract_markdown):
        logger.warning("[NuanceAudit] rejected finding — clause_quote not verbatim in contract")
        return None

    basis = str(data.get("basis", ""))
    # Book lookup keyed off the DETECTED basis — reviewer context only, never sets the factor.
    kb_reference, kb_context = "", ""
    try:
        passages = search_fn(f"royalty basis {basis}", floor_count=0)
        kb_context = "\n".join(getattr(p, "text", "") for p in passages[:2])
        kb_reference = "; ".join(
            f"{getattr(p, 'section_path', '')} p.{getattr(p, 'page_start', '')}" for p in passages[:2]
        )
    except Exception as e:  # noqa: BLE001 — book is optional enrichment
        logger.warning(f"[NuanceAudit] reference lookup failed (finding still logged): {e}")

    return BasisFinding(
        basis=basis,
        implied_factor=float(factor),
        clause_quote=quote,
        affected_types=data.get("affected_types", "all"),
        kb_reference=kb_reference,
        kb_context=kb_context,
    )


def log_basis_finding(
    payments: list, finding: BasisFinding | None, *, contract_id: str, user_id: str | None = None
) -> list:
    """Record a detected basis nuance for human review. Returns payments UNCHANGED (v1 is log-only).

    Full detail (verbatim clause + book passage text) goes to the secure backend
    `oneclick.audit` log. An aggregate `oneclick_basis_detected` PostHog event makes the
    feature observable — it carries NO contract text (no clause), only counts/factor.
    """
    if finding is None:
        return payments

    affected = finding.affected_types
    if affected == "all":
        lines = payments
    else:
        # Coerce a bare string ("Streaming") to a set so the review count isn't silently zeroed.
        type_set = set(affected) if isinstance(affected, list) else {affected}
        lines = [p for p in payments if getattr(p, "royalty_type", None) in type_set]
    review_sum = sum(getattr(p, "amount_to_pay", 0.0) for p in lines)

    audit_logger.info(
        "basis-nuance-detected (NOT applied) contract=%s basis=%r implied_factor=%s affected=%s "
        "clause=%r kb=%r kb_text=%r review_lines=%d review_amount=%.2f",
        contract_id,
        finding.basis,
        finding.implied_factor,
        finding.affected_types,
        finding.clause_quote,
        finding.kb_reference,
        finding.kb_context[:200],  # the useful half of the lookup, not just the pointer
        len(lines),
        review_sum,
    )

    # Observability — aggregate only. NEVER the clause text / contract content.
    if user_id:
        analytics_capture(
            user_id,
            "oneclick_basis_detected",
            {
                "tool": "oneclick",
                "implied_factor": finding.implied_factor,
                "affected_count": len(lines),
                "has_kb": bool(finding.kb_reference),
            },
        )
    return payments
