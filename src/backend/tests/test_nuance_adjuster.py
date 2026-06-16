"""Tests for the OneClick contract-nuance basis detector (log-only v1)."""

import json
from unittest.mock import MagicMock

from utils.contract_parsing.basis_detection import BasisFinding, audit_contract_basis

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


def _no_kb(*a, **k):
    return []


def test_explicit_basis_yields_verified_finding():
    client = _client_returning(
        {
            "applies": True,
            "implied_factor": 0.75,
            "basis": "Net Receipts after 25% packaging deduction",
            "clause_quote": "less a packaging deduction of twenty-five percent (25%)",
            "affected_types": "all",
        }
    )
    f = audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_no_kb)
    assert isinstance(f, BasisFinding)
    assert f.implied_factor == 0.75
    assert f.clause_quote in CONTRACT


def test_no_basis_clause_returns_none():
    client = _client_returning({"applies": False, "implied_factor": 1.0, "clause_quote": "", "affected_types": "all"})
    assert audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_no_kb) is None


def test_hallucinated_quote_is_rejected():
    client = _client_returning(
        {
            "applies": True,
            "implied_factor": 0.5,
            "basis": "fabricated",
            "clause_quote": "a 50% deduction for imaginary fees not in this contract",
            "affected_types": "all",
        }
    )
    assert audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_no_kb) is None


def test_pdf_artifact_quote_still_matches():
    # Contract: curly quotes + em-dash with NO surrounding spaces. LLM: straight quotes + spaced hyphen.
    # Only the quote/dash + hyphen-spacing normalization makes these match.
    contract = "Royalty is paid on “Net Sales”—defined as gross less a 10% reserve."
    client = _client_returning(
        {
            "applies": True,
            "implied_factor": 0.9,
            "basis": "net sales less 10% reserve",
            "clause_quote": 'on "Net Sales" - defined as gross less a 10% reserve',
            "affected_types": "all",
        }
    )
    f = audit_contract_basis(contract, ["Master"], openai_client=client, search_fn=_no_kb)
    assert isinstance(f, BasisFinding) and f.implied_factor == 0.9


def test_out_of_range_factor_rejected():
    client = _client_returning(
        {"applies": True, "implied_factor": 1.5, "clause_quote": "Net Receipts", "affected_types": "all"}
    )
    assert audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_no_kb) is None


def test_royalty_types_injected_into_prompt():
    client = _client_returning({"applies": False, "clause_quote": "", "affected_types": "all"})
    audit_contract_basis(CONTRACT, ["Master", "Producer"], openai_client=client, search_fn=_no_kb)
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Master" in sent and "Producer" in sent


def test_llm_exception_returns_none():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("api down")
    assert audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_no_kb) is None


def test_empty_markdown_returns_none():
    client = _client_returning({"applies": True, "implied_factor": 0.75, "clause_quote": "x", "affected_types": "all"})
    assert audit_contract_basis("", ["Streaming"], openai_client=client, search_fn=_no_kb) is None


def test_short_quote_is_rejected():
    # A degenerate short quote that appears in any contract must NOT produce a finding.
    client = _client_returning(
        {"applies": True, "implied_factor": 0.5, "basis": "b", "clause_quote": "0%", "affected_types": "all"}
    )
    assert audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_no_kb) is None


def test_non_numeric_factor_rejected():
    client = _client_returning(
        {"applies": True, "implied_factor": None, "clause_quote": "less a packaging deduction", "affected_types": "all"}
    )
    assert audit_contract_basis(CONTRACT, ["Streaming"], openai_client=client, search_fn=_no_kb) is None


import logging
from dataclasses import dataclass

from utils.contract_parsing.basis_detection import log_basis_finding


@dataclass
class _Pay:
    song_title: str
    party_name: str
    royalty_type: str
    amount_to_pay: float


def _payments():
    return [_Pay("Song A", "Artist", "Streaming", 500.0), _Pay("Song A", "Producer", "Producer", 30.0)]


def test_log_does_not_change_payments():
    pays = _payments()
    f = BasisFinding(basis="net", implied_factor=0.75, clause_quote="less 25%", affected_types="all")
    out = log_basis_finding(pays, f, contract_id="c1")
    assert [p.amount_to_pay for p in out] == [500.0, 30.0]  # unchanged


def test_log_writes_audit_record(caplog):
    f = BasisFinding(
        basis="net receipts",
        implied_factor=0.75,
        clause_quote="less 25%",
        affected_types="all",
        kb_reference="Publishing p.247",
    )
    with caplog.at_level(logging.INFO, logger="oneclick.audit"):
        log_basis_finding(_payments(), f, contract_id="c1")
    text = caplog.text
    assert "c1" in text and "less 25%" in text and "0.75" in text and "Publishing p.247" in text


def test_log_none_is_silent_noop(caplog):
    with caplog.at_level(logging.INFO, logger="oneclick.audit"):
        out = log_basis_finding(_payments(), None, contract_id="c1")
    assert caplog.text == "" and [p.amount_to_pay for p in out] == [500.0, 30.0]


def test_affected_types_bare_string_counts_lines(caplog):
    # A bare "Streaming" (not a list) must still count the streaming line, not zero it.
    f = BasisFinding(basis="net", implied_factor=0.75, clause_quote="q", affected_types="Streaming")
    with caplog.at_level(logging.INFO, logger="oneclick.audit"):
        log_basis_finding(_payments(), f, contract_id="c1")
    assert "review_lines=1" in caplog.text


def test_log_includes_kb_text(caplog):
    f = BasisFinding(
        basis="net",
        implied_factor=0.75,
        clause_quote="q",
        affected_types="all",
        kb_context="Packaging deductions are standard at 25%.",
    )
    with caplog.at_level(logging.INFO, logger="oneclick.audit"):
        log_basis_finding(_payments(), f, contract_id="c1")
    assert "Packaging deductions are standard" in caplog.text


def test_emits_aggregate_analytics_no_clause_text():
    from unittest.mock import patch

    f = BasisFinding(basis="net", implied_factor=0.75, clause_quote="SECRET CLAUSE", affected_types="all")
    with patch("utils.contract_parsing.basis_detection.analytics_capture") as cap:
        log_basis_finding(_payments(), f, contract_id="c1", user_id="u1")
    cap.assert_called_once()
    assert cap.call_args.args[0] == "u1" and cap.call_args.args[1] == "oneclick_basis_detected"
    assert "SECRET CLAUSE" not in str(cap.call_args)  # clause text never leaves for analytics
