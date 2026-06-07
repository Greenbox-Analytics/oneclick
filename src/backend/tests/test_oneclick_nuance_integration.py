"""Integration + namespace tests for OneClick's log-only nuance audit.

Covers the gaps left by the unit tests in test_nuance_adjuster.py:
  B1 — the wired path through RoyaltyCalculator.calculate_payments: the nuance audit
       runs (and is logged) without changing the payout, and is skipped when there's
       no full_text.
  B2 — audit_contract_basis actually consults the reference namespace (search_fn) with
       the basis-keyed query and threads the result into the finding.
  B3 — a golden-output regression for a normal contract: the core math is pinned so a
       future change can't silently shift a payout.
  C  — an opt-in LIVE smoke against the real Pinecone namespace (skipped by default).
"""

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import oneclick.royalty_calculator as rc
from oneclick.nuance_adjuster import BasisFinding, audit_contract_basis
from oneclick.royalty_calculator import RoyaltyCalculator

# A contract whose clause is verbatim-present (the audit rejects non-verbatim quotes).
CONTRACT = (
    "ARTIST ROYALTY. The Company shall pay Artist fifty percent (50%) of Net Receipts. "
    '"Net Receipts" means gross receipts less a packaging deduction of twenty-five percent (25%).'
)


def _client_returning(payload: dict):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    return client


def _contract_data():
    return SimpleNamespace(
        works=[SimpleNamespace(title="Midnight Drive")],
        parties=[SimpleNamespace(name="Alice", role="Artist")],
        royalty_shares=[SimpleNamespace(royalty_type="Streaming", percentage=50.0, party_name="Alice", terms=None)],
    )


def _calc_with_mocked_seams(payout):
    """A RoyaltyCalculator whose contract/statement parsing and per-line math are stubbed,
    so calculate_payments() exercises only the nuance-audit wiring deterministically."""
    calc = RoyaltyCalculator.__new__(RoyaltyCalculator)  # skip __init__
    calc.contract_parser = MagicMock()
    calc.contract_parser.parse_contract.return_value = _contract_data()
    calc.read_royalty_statement = MagicMock(return_value={"Midnight Drive": 100.0})
    calc._calculate_payments_from_data = MagicMock(return_value=payout)
    return calc


# ── B1: the wired calculate_payments path ───────────────────────────────────────────


def test_calc_logs_nuance_without_changing_payout(caplog):
    payout = [SimpleNamespace(royalty_type="Streaming", amount_to_pay=50.0, song_title="Midnight Drive")]
    calc = _calc_with_mocked_seams(payout)
    finding = BasisFinding(
        basis="Net Receipts after 25% packaging deduction",
        implied_factor=0.75,
        clause_quote="less a packaging deduction of twenty-five percent (25%)",
        affected_types="all",
        kb_reference="Part V p.210",
        kb_context="Royalties are paid on net receipts.",
    )

    with (
        patch("oneclick.royalty_calculator.audit_contract_basis", return_value=finding) as mock_audit,
        caplog.at_level(logging.INFO, logger="oneclick.audit"),
    ):
        result = calc.calculate_payments(
            contract_path="c.pdf",
            statement_path="s.xlsx",
            user_id="u1",
            contract_id="c1",
            full_text=CONTRACT,
        )

    # Payout is returned UNCHANGED (log-only): same object, same amounts.
    assert result is payout
    assert [p.amount_to_pay for p in result] == [50.0]
    # The audit was wired to the REAL reference-namespace search (not a no-op).
    assert mock_audit.call_args.kwargs["search_fn"] is rc.search_reference
    # The finding was recorded for human review.
    assert any("basis-nuance-detected (NOT applied)" in r.getMessage() for r in caplog.records)


def test_calc_skips_audit_when_no_full_text():
    payout = [SimpleNamespace(royalty_type="Streaming", amount_to_pay=50.0, song_title="Midnight Drive")]
    calc = _calc_with_mocked_seams(payout)

    with patch("oneclick.royalty_calculator.audit_contract_basis") as mock_audit:
        result = calc.calculate_payments(
            contract_path="c.pdf",
            statement_path="s.xlsx",
            user_id="u1",
            contract_id="c1",
            full_text=None,
        )

    mock_audit.assert_not_called()
    assert result is payout


# ── B2: audit consults the reference namespace ───────────────────────────────────────


def test_audit_queries_namespace_with_basis_and_threads_kb():
    client = _client_returning(
        {
            "applies": True,
            "implied_factor": 0.75,
            "basis": "Net Receipts after 25% packaging deduction",
            "clause_quote": "less a packaging deduction of twenty-five percent (25%)",
            "affected_types": "all",
        }
    )
    passage = SimpleNamespace(
        text="Royalties are typically paid on net receipts after deductions.",
        section_path="Part V ▸ Record Deals",
        page_start=210,
    )
    search_fn = MagicMock(return_value=[passage])

    finding = audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=search_fn)

    assert finding is not None
    # The namespace is queried with the basis-keyed query (strict floor).
    search_fn.assert_called_once_with("royalty basis Net Receipts after 25% packaging deduction", floor_count=0)
    # The returned passage is threaded into the reviewer-facing fields.
    assert "net receipts after deductions" in finding.kb_context
    assert "Part V" in finding.kb_reference and "210" in finding.kb_reference


def test_audit_survives_namespace_failure_still_returns_finding():
    client = _client_returning(
        {
            "applies": True,
            "implied_factor": 0.75,
            "basis": "Net Receipts",
            "clause_quote": "less a packaging deduction of twenty-five percent (25%)",
            "affected_types": "all",
        }
    )

    def _boom(*a, **k):
        raise RuntimeError("Pinecone down")

    finding = audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_boom)
    assert finding is not None  # detection still logged; book is optional enrichment
    assert finding.kb_context == "" and finding.kb_reference == ""


# ── B3: golden-output regression for a normal contract ───────────────────────────────


def test_golden_normal_contract_payout():
    """Pin the core math: 50% of each matched song's total. Guards against silent drift."""
    calc = RoyaltyCalculator.__new__(RoyaltyCalculator)
    contract_data = SimpleNamespace(
        works=[SimpleNamespace(title="Midnight Drive"), SimpleNamespace(title="Ocean Eyes")],
        parties=[SimpleNamespace(name="Alice", role="Artist")],
        royalty_shares=[SimpleNamespace(royalty_type="Streaming", percentage=50.0, party_name="Alice", terms=None)],
    )
    song_totals = {"Midnight Drive": 100.0, "Ocean Eyes": 40.0}

    payments = calc._calculate_payments_from_data(contract_data, song_totals)

    by_song = {p.song_title: p.amount_to_pay for p in payments}
    assert by_song == {"Midnight Drive": 50.0, "Ocean Eyes": 20.0}
    assert all(p.percentage == 50.0 and p.party_name == "Alice" for p in payments)


# ── C: opt-in LIVE namespace smoke (skipped unless explicitly enabled) ────────────────


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_REFERENCE_TESTS") != "1",
    reason=(
        "Live namespace smoke. Set RUN_LIVE_REFERENCE_TESTS=1 to run "
        "(needs PINECONE_API_KEY, OPENAI_API_KEY, and the reference book uploaded)."
    ),
)
def test_live_namespace_returns_relevant_passages():
    from knowledge.reference_search import search_reference

    passages = search_reference("royalty basis net receipts", floor_count=0)
    assert passages, "reference namespace returned no passages for a royalty-basis query"
    top = passages[0]
    assert top.score >= 0.40
    assert top.text and top.section_path is not None
