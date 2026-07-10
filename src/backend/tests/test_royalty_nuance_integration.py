"""Integration: single-contract calc detects + logs a basis nuance WITHOUT changing payouts,
and the helper forwards contract_id so the audit log is accountable."""

from unittest.mock import MagicMock, patch

from oneclick.royalty_calculator import CalcOutput, RoyaltyCalculator, RoyaltyPayment
from utils.contract_parsing.basis_detection import BasisFinding


def _calc_with(payments):
    calc = RoyaltyCalculator.__new__(RoyaltyCalculator)  # skip __init__
    calc.contract_parser = MagicMock()
    calc.contract_parser.parse_contract.return_value = MagicMock(
        works=[MagicMock(title="Song A")], royalty_shares=[MagicMock(royalty_type="Streaming")]
    )
    calc.read_royalty_statement = MagicMock(return_value={"Song A": 1000.0})
    calc._calculate_payments_from_data = MagicMock(return_value=payments)
    return calc


@patch("oneclick.royalty_calculator.audit_contract_basis")
@patch("oneclick.royalty_calculator.verify_royalty_shares", return_value=None)
def test_single_contract_logs_but_does_not_change_payouts(mock_verify, mock_audit):
    mock_audit.return_value = BasisFinding(basis="net", implied_factor=0.75, clause_quote="q", affected_types="all")
    pays = [RoyaltyPayment("Song A", "Artist", "Artist", "Streaming", 50.0, 1000.0, 500.0)]
    calc = _calc_with(pays)
    out = calc.calculate_payments(contract_path="x", statement_path="y", full_text="... net ...", contract_id="c1")
    assert out.payments[0].amount_to_pay == 500.0  # UNCHANGED — log-only
    assert mock_audit.called  # detection ran


@patch("oneclick.royalty_calculator.audit_contract_basis", return_value=None)
@patch("oneclick.royalty_calculator.verify_royalty_shares", return_value=None)
def test_single_contract_no_finding_is_unchanged(mock_verify, _mock_audit):
    pays = [RoyaltyPayment("Song A", "Artist", "Artist", "Streaming", 50.0, 1000.0, 500.0)]
    calc = _calc_with(pays)
    out = calc.calculate_payments(contract_path="x", statement_path="y", full_text="no clause", contract_id="c1")
    assert out.payments[0].amount_to_pay == 500.0


def test_helper_forwards_contract_id():
    # The helper must pass contract_id into calculate_payments so audit logs aren't "unknown".
    with patch("oneclick.royalty_calculator.RoyaltyCalculator") as MockCalc:
        instance = MockCalc.return_value
        instance.calculate_payments.return_value = CalcOutput(payments=[], review=None)
        from zoe_chatbot.helpers import calculate_royalty_payments

        calculate_royalty_payments(
            contract_path="x",
            statement_path="y",
            user_id="u",
            contract_id="c1",
            contract_markdowns={"c1": "full text"},
        )
        assert instance.calculate_payments.call_args.kwargs.get("contract_id") == "c1"
        assert instance.calculate_payments.call_args.kwargs.get("user_id") == "u"


def test_multi_contract_logs_deferred(caplog):
    import logging

    calc = RoyaltyCalculator.__new__(RoyaltyCalculator)
    calc.contract_parser = MagicMock()
    calc.contract_parser.parse_contract.return_value = MagicMock(parties=[], works=[], royalty_shares=[])
    calc.merge_contracts = MagicMock(return_value=MagicMock())
    calc.read_royalty_statement = MagicMock(return_value={"Song A": 1000.0})
    calc._calculate_payments_from_data = MagicMock(return_value=[])
    with caplog.at_level(logging.INFO):
        calc.calculate_payments_from_contract_ids(
            contract_ids=["c1", "c2"],
            user_id="u",
            statement_path="y",
            contract_markdowns={"c1": "a", "c2": "b"},
        )
    assert "deferred" in caplog.text.lower()


@patch("oneclick.royalty_calculator.verify_royalty_shares")
def test_multi_contract_partial_verification_is_not_green(mock_verify):
    from utils.contract_parsing.split_verification import ReviewResult, SplitFinding

    share = MagicMock(party_name="Alice", royalty_type="Streaming", percentage=50.0, basis=None)
    parsed = MagicMock(parties=[], works=[], royalty_shares=[share], default_basis=None)

    ok = ReviewResult(
        overall="verified",
        checked=1,
        flagged=0,
        findings=[
            SplitFinding(
                party_name="Alice",
                royalty_type="Streaming",
                extracted_percentage=50.0,
                extracted_basis="gross",
                verdict="verified",
            )
        ],
    )
    # Keyed off the markdown each verify receives: contract "a" runs, contract "b" can't.
    mock_verify.side_effect = lambda md, data, **kw: ok if md == "a" else ReviewResult(overall="unavailable")

    calc = RoyaltyCalculator.__new__(RoyaltyCalculator)
    calc.contract_parser = MagicMock()
    calc.contract_parser.parse_contract.return_value = parsed
    calc.read_royalty_statement = MagicMock(return_value={"Song A": 1000.0})
    calc.merge_contracts = MagicMock(return_value=MagicMock())
    calc._calculate_payments_from_data = MagicMock(return_value=[])

    out = calc.calculate_payments_from_contract_ids(
        contract_ids=["c1", "c2"],
        user_id="u",
        statement_path="y",
        contract_markdowns={"c1": "a", "c2": "b"},
    )
    # Partial coverage must never render as fully verified: the unchecked contract's
    # share is counted as an unverified finding.
    assert out.review.overall == "needs_review"
    assert out.review.checked == 2 and out.review.flagged == 1
    assert any(f.verdict == "unverified" for f in out.review.findings)


@patch("oneclick.royalty_calculator.verify_royalty_shares")
def test_parse_failed_contract_blocks_green_banner(mock_verify):
    from utils.contract_parsing.split_verification import ReviewResult, SplitFinding

    share = MagicMock(party_name="Alice", royalty_type="Streaming", percentage=50.0, basis=None)
    parsed = MagicMock(parties=[], works=[], royalty_shares=[share], default_basis=None)
    mock_verify.return_value = ReviewResult(
        overall="verified",
        checked=1,
        flagged=0,
        findings=[
            SplitFinding(
                party_name="Alice",
                royalty_type="Streaming",
                extracted_percentage=50.0,
                extracted_basis="gross",
                verdict="verified",
            )
        ],
    )

    calc = RoyaltyCalculator.__new__(RoyaltyCalculator)
    calc.contract_parser = MagicMock()
    calc.contract_parser.parse_contract.side_effect = [parsed, ValueError("unreadable")]
    calc.read_royalty_statement = MagicMock(return_value={"Song A": 1000.0})
    calc.merge_contracts = MagicMock(return_value=MagicMock())
    calc._calculate_payments_from_data = MagicMock(return_value=[])

    out = calc.calculate_payments_from_contract_ids(
        contract_ids=["c1", "c2"],
        user_id="u",
        statement_path="y",
        contract_markdowns={"c1": "a", "c2": "b"},
    )
    # One contract verified clean, one couldn't be parsed -> never a green banner.
    assert out.review.overall == "needs_review"
    assert out.review.flagged >= 1
    assert any("could not be read" in f.note for f in out.review.findings)


def test_helper_serializes_review_to_dict():
    import json as _json

    from utils.contract_parsing.split_verification import ReviewResult, SplitFinding

    with patch("oneclick.royalty_calculator.RoyaltyCalculator") as MockCalc:
        instance = MockCalc.return_value
        instance.calculate_payments.return_value = CalcOutput(
            payments=[],
            review=ReviewResult(
                overall="verified",
                checked=1,
                flagged=0,
                findings=[
                    SplitFinding(
                        party_name="A",
                        royalty_type="Streaming",
                        extracted_percentage=50.0,
                        extracted_basis="gross",
                        verdict="verified",
                    )
                ],
            ),
        )
        from zoe_chatbot.helpers import calculate_royalty_payments

        payments, review = calculate_royalty_payments(
            contract_path="x", statement_path="y", user_id="u", contract_id="c1", contract_markdowns={"c1": "t"}
        )
    assert payments == []
    assert review["overall"] == "verified"
    assert review["findings"][0]["party_name"] == "A"
    _json.dumps(review)  # dataclass -> dict -> JSON round-trip holds
