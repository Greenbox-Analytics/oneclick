from unittest.mock import MagicMock, patch

from oneclick.royalty_calculator import RoyaltyCalculator
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work


def _cd():
    return ContractData(
        parties=[Party(name="Jane Doe", role="artist")],
        works=[Work(title="Midnight")],
        royalty_shares=[RoyaltyShare(party_name="Jane Doe", royalty_type="Streaming", percentage=100.0)],
        contract_summary="",
        default_basis="gross",
    )


def test_single_uses_preparsed_and_skips_parser():
    calc = RoyaltyCalculator(api_key="sk-test-key")
    calc.contract_parser = MagicMock()
    calc.read_royalty_statement = MagicMock(return_value={"Midnight": 100.0})

    out = calc.calculate_payments(
        contract_path=None,
        statement_path="ignored.xlsx",
        full_text=None,
        contract_id="cid-1",
        user_id="u1",
        contract_data=_cd(),
    )

    calc.contract_parser.parse_contract.assert_not_called()
    assert any(p.party_name == "Jane Doe" for p in out.payments)


def test_single_without_preparsed_still_parses():
    calc = RoyaltyCalculator(api_key="sk-test-key")
    calc.contract_parser = MagicMock()
    calc.contract_parser.parse_contract.return_value = _cd()
    calc.contract_parser.openai_client = MagicMock()
    calc.read_royalty_statement = MagicMock(return_value={"Midnight": 100.0})

    with (
        patch("oneclick.royalty_calculator.audit_contract_basis", return_value=None),
        patch("oneclick.royalty_calculator.verify_royalty_shares", return_value=None),
        patch("oneclick.royalty_calculator.log_basis_finding", side_effect=lambda payments, *a, **k: payments),
    ):
        calc.calculate_payments(
            contract_path=None,
            statement_path="ignored.xlsx",
            full_text="some markdown",
            contract_id="cid-1",
            user_id="u1",
        )

    calc.contract_parser.parse_contract.assert_called_once_with(full_text="some markdown")


def test_multi_uses_preparsed_map_and_skips_parser():
    calc = RoyaltyCalculator(api_key="sk-test-key")
    calc.contract_parser = MagicMock()
    calc.contract_parser.openai_client = MagicMock()
    calc.read_royalty_statement = MagicMock(return_value={"Midnight": 100.0})

    with patch("oneclick.royalty_calculator.verify_royalty_shares", return_value=None):
        calc.calculate_payments_from_contract_ids(
            contract_ids=["cid-1", "cid-2"],
            user_id="u1",
            statement_path="ignored.xlsx",
            contract_markdowns={"cid-1": "md1", "cid-2": "md2"},
            contract_data_by_id={"cid-1": _cd(), "cid-2": _cd()},
        )

    calc.contract_parser.parse_contract.assert_not_called()


def test_multi_partial_preparsed_only_parses_missing():
    calc = RoyaltyCalculator(api_key="sk-test-key")
    calc.contract_parser = MagicMock()
    calc.contract_parser.parse_contract.return_value = _cd()
    calc.contract_parser.openai_client = MagicMock()
    calc.read_royalty_statement = MagicMock(return_value={"Midnight": 100.0})

    with patch("oneclick.royalty_calculator.verify_royalty_shares", return_value=None):
        calc.calculate_payments_from_contract_ids(
            contract_ids=["cid-1", "cid-2"],
            user_id="u1",
            statement_path="ignored.xlsx",
            contract_markdowns={"cid-1": "m1", "cid-2": "m2"},
            contract_data_by_id={"cid-1": _cd()},  # only cid-1 pre-parsed
        )

    calc.contract_parser.parse_contract.assert_called_once_with(full_text="m2")


def test_helper_forwards_preparsed_single(monkeypatch):
    import zoe_chatbot.helpers as helpers

    fake_calc = MagicMock()
    fake_calc.calculate_payments.return_value = MagicMock(payments=[], review=None)
    monkeypatch.setattr(helpers, "RoyaltyCalculator", lambda *a, **k: fake_calc, raising=False)

    cd = _cd()
    helpers.calculate_royalty_payments(
        contract_path=None,
        statement_path="s.xlsx",
        user_id="u1",
        contract_id="cid-1",
        contract_markdowns={"cid-1": "md"},
        contract_data_by_id={"cid-1": cd},
    )

    kwargs = fake_calc.calculate_payments.call_args.kwargs
    assert kwargs["contract_data"] is cd


def test_helper_forwards_preparsed_multi(monkeypatch):
    import zoe_chatbot.helpers as helpers

    fake_calc = MagicMock()
    fake_calc.calculate_payments_from_contract_ids.return_value = MagicMock(payments=[], review=None)
    monkeypatch.setattr(helpers, "RoyaltyCalculator", lambda *a, **k: fake_calc, raising=False)

    m = {"cid-1": _cd(), "cid-2": _cd()}
    helpers.calculate_royalty_payments(
        contract_path=None,
        statement_path="s.xlsx",
        user_id="u1",
        contract_id=None,
        contract_ids=["cid-1", "cid-2"],
        contract_markdowns={"cid-1": "m1", "cid-2": "m2"},
        contract_data_by_id=m,
    )

    kwargs = fake_calc.calculate_payments_from_contract_ids.call_args.kwargs
    assert kwargs["contract_data_by_id"] is m
