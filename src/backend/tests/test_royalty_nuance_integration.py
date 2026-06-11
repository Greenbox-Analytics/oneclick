"""Integration: single-contract calc detects + logs a basis nuance WITHOUT changing payouts,
and the helper forwards contract_id so the audit log is accountable."""

from unittest.mock import MagicMock, patch

from oneclick.royalty_calculator import RoyaltyCalculator, RoyaltyPayment
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
def test_single_contract_logs_but_does_not_change_payouts(mock_audit):
    mock_audit.return_value = BasisFinding(basis="net", implied_factor=0.75, clause_quote="q", affected_types="all")
    pays = [RoyaltyPayment("Song A", "Artist", "Artist", "Streaming", 50.0, 1000.0, 500.0)]
    calc = _calc_with(pays)
    out = calc.calculate_payments(contract_path="x", statement_path="y", full_text="... net ...", contract_id="c1")
    assert out[0].amount_to_pay == 500.0  # UNCHANGED — log-only
    assert mock_audit.called  # detection ran


@patch("oneclick.royalty_calculator.audit_contract_basis", return_value=None)
def test_single_contract_no_finding_is_unchanged(_mock_audit):
    pays = [RoyaltyPayment("Song A", "Artist", "Artist", "Streaming", 50.0, 1000.0, 500.0)]
    calc = _calc_with(pays)
    out = calc.calculate_payments(contract_path="x", statement_path="y", full_text="no clause", contract_id="c1")
    assert out[0].amount_to_pay == 500.0


def test_helper_forwards_contract_id():
    # The helper must pass contract_id into calculate_payments so audit logs aren't "unknown".
    with patch("oneclick.royalty_calculator.RoyaltyCalculator") as MockCalc:
        instance = MockCalc.return_value
        instance.calculate_payments.return_value = []
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
