"""Tests for royalty_ledger_history recorder."""

from unittest.mock import MagicMock

import pytest

from oneclick.royalties import history


def test_record_inserts_row():
    db = MagicMock()
    history.record(db, "u1", "deleted", {"id": "l1", "amount_owed": 2.0}, "contract_deleted")
    db.table.assert_called_with("royalty_ledger_history")
    inserted = db.table.return_value.insert.call_args[0][0]
    assert inserted["user_id"] == "u1"
    assert inserted["action"] == "deleted"
    assert inserted["old_row"] == {"id": "l1", "amount_owed": 2.0}
    assert inserted["cause"] == "contract_deleted"


def test_record_propagates_db_errors():
    db = MagicMock()
    db.table.return_value.insert.return_value.execute.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        history.record(db, "u1", "updated", {"id": "l1"}, "calc-1")
