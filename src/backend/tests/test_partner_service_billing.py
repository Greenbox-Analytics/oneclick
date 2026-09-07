"""partner_api.service billing helpers (spec 2026-09-06 §3.1)."""

from unittest.mock import MagicMock

from partner_api import service as psvc
from tests.conftest import MockQueryBuilder


def _sb(rows):
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=rows)
    sb = MagicMock()
    sb.table.side_effect = lambda name: {"credit_ledger": b}[name]
    return sb, b


def test_already_charged_is_one_ledger_read():
    sb, b = _sb([{"id": "row"}])
    b.select = MagicMock(return_value=b)
    b.eq = MagicMock(return_value=b)
    b.limit = MagicMock(return_value=b)
    assert psvc.already_charged(sb, "rid") is True
    assert b.select.call_args.args == ("id",)
    assert ("request_id", "rid") in [c.args for c in b.eq.call_args_list]
    assert b.limit.call_args.args == (1,)
    assert b.execute.call_count == 1  # ONE ledger read

    sb2, _ = _sb([])
    assert psvc.already_charged(sb2, "rid") is False


def test_already_charged_failure_reports_not_charged():
    # Over-report the price rather than fail a delivered run.
    sb = MagicMock()
    sb.table.side_effect = RuntimeError("db down")
    assert psvc.already_charged(sb, "rid") is False


def test_billing_block_and_unbilled():
    assert psvc.billing_block(30, "r1") == {"credits": 30, "request_id": "r1"}
    assert psvc.billing_block(30, "r1", replayed=True) == {"credits": 0, "request_id": "r1", "replayed": True}
    assert psvc.unbilled() == {"credits": 0}
