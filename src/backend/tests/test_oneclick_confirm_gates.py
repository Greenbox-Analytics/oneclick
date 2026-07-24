"""Tests for POST /oneclick/confirm's gate resequencing (ledger reconciliation Task 7).

Acceptance criteria:
1. A gate raised before any write returns a structured 409 and leaves no
   royalty_calculations cache row behind.
2. The happy path calls sync_calc_royalties twice — check_only=True first,
   then the full sync — and reports ledger_synced in the response.
3. A non-gate sync failure on the full sync surfaces via ledger_synced=False /
   ledger_error, instead of being silently swallowed.
4. A gate that fires only on the full sync (state changed between the two
   calls) still returns a structured 409 — the "belt" path.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oneclick.royalties.ledger_sync import SyncGateError
from tests.conftest import MockQueryBuilder

PROJECT_ID = "proj-0000-0000-0000-0000-000000000099"
CONTRACT_ID = "cont-0000-0000-0000-0000-000000000099"
STATEMENT_FILE_ID = "file-0000-0000-0000-0000-000000000099"
CALCULATION_ID = "calc-0000-0000-0000-0000-000000000099"

RESULTS_PAYLOAD = {
    "status": "success",
    "total_payments": 1,
    "payments": [
        {
            "song_title": "Blue Sky",
            "party_name": "Jane Doe",
            "role": "Producer",
            "royalty_type": "master",
            "percentage": 50.0,
            "total_royalty": 1000.0,
            "amount_to_pay": 500.0,
            "terms": "Net 30 days",
        }
    ],
    "message": "1 royalty payment",
}


@pytest.fixture(autouse=True)
def _bypass_oneclick_ownership(monkeypatch):
    """Bypass the OneClick ownership guard — access control is covered
    elsewhere (test_oneclick_ownership.py); these tests focus on gate
    sequencing and response shape."""
    monkeypatch.setattr("main._assert_can_access_oneclick_inputs", AsyncMock(return_value=None))


@pytest.fixture(autouse=True)
def _bypass_row_persistence(monkeypatch):
    """Stub the best-effort statement-rows helpers so the endpoint reaches
    the gate/sync logic without needing a real statement file to parse."""
    monkeypatch.setattr("main._persist_statement_rows", MagicMock(return_value=None))
    monkeypatch.setattr("main._parse_statement_rows", MagicMock(return_value=[]))
    monkeypatch.setattr("main._statement_currency", MagicMock(return_value="USD"))


def _table_queue_side_effect(queues: dict):
    """Return a `mock_supabase.table.side_effect` that pops the next canned
    `data` list for *name* off its queue (FIFO), defaulting to `[]` once a
    table's queue is exhausted or if it has none. Keeps mocks robust against
    extra/incidental table() calls instead of relying on call position."""

    def _side_effect(name):
        b = MockQueryBuilder()
        q = queues.get(name)
        data = q.pop(0) if q else []
        b.execute.return_value = MagicMock(data=data, count=len(data))
        return b

    return _side_effect


def _post_confirm(client, **overrides):
    payload = {
        "contract_ids": [CONTRACT_ID],
        "royalty_statement_id": STATEMENT_FILE_ID,
        "project_id": PROJECT_ID,
        "results": RESULTS_PAYLOAD,
        **overrides,
    }
    return client.post("/oneclick/confirm", json=payload)


class TestConfirmGateSequencing:
    def test_gate_raise_returns_409_and_writes_no_cache(self, client, mock_supabase):
        """A gate firing at check-time returns a structured 409 and never
        reaches the royalty_calculations cache write (or even its select)."""
        called_tables = []

        def _side_effect(name):
            called_tables.append(name)
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)
            return b

        mock_supabase.table.side_effect = _side_effect

        with patch(
            "main.sync_calc_royalties",
            side_effect=SyncGateError("revision", {"candidates": []}),
        ) as mock_sync:
            response = _post_confirm(client)

        assert response.status_code == 409
        assert response.json()["detail"]["gate"] == "revision"
        assert response.json()["detail"]["payload"] == {"candidates": []}
        mock_sync.assert_called_once()
        assert mock_sync.call_args.kwargs.get("check_only") is True

        # No write path was reached at all — the gate check happens before
        # the endpoint ever touches royalty_calculations.
        assert "royalty_calculations" not in called_tables
        assert "royalty_calculation_contracts" not in called_tables

    def test_happy_path_reports_ledger_synced(self, client, mock_supabase):
        """Both sync_calc_royalties calls succeed: check_only=True first,
        then the full sync. The response reports ledger_synced=True."""
        queues = {
            "royalty_calculations": [[], [{"id": CALCULATION_ID}]],
            "royalty_calculation_contracts": [[]],
        }
        mock_supabase.table.side_effect = _table_queue_side_effect(queues)

        with patch("main.sync_calc_royalties", side_effect=[None, "stmt-1"]) as mock_sync:
            response = _post_confirm(client)

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == CALCULATION_ID
        assert body["ledger_synced"] is True
        assert body["ledger_error"] is None

        assert mock_sync.call_count == 2
        first_call, second_call = mock_sync.call_args_list
        assert first_call.kwargs.get("check_only") is True
        assert second_call.kwargs.get("check_only", False) is False

    def test_sync_failure_surfaces_not_swallowed(self, client, mock_supabase):
        """The gate check passes, but the full sync raises a non-gate error.
        It must surface via ledger_synced/ledger_error, not be swallowed into
        a bare success response."""
        queues = {
            "royalty_calculations": [[], [{"id": CALCULATION_ID}]],
            "royalty_calculation_contracts": [[]],
        }
        mock_supabase.table.side_effect = _table_queue_side_effect(queues)

        with patch("main.sync_calc_royalties", side_effect=[None, RuntimeError("db down")]) as mock_sync:
            response = _post_confirm(client)

        assert response.status_code == 200
        body = response.json()
        assert body["ledger_synced"] is False
        assert "db down" in body["ledger_error"]
        assert mock_sync.call_count == 2

    def test_belt_gate_after_cache_write_is_409(self, client, mock_supabase):
        """The gate check passes, but state changes before the full sync runs
        and a gate fires there instead ('belt' path). Still a structured 409,
        even though the cache row was already written."""
        queues = {
            "royalty_calculations": [[], [{"id": CALCULATION_ID}]],
            "royalty_calculation_contracts": [[]],
        }
        mock_supabase.table.side_effect = _table_queue_side_effect(queues)

        belt_error = SyncGateError("conflict", {"scope": "cross_run", "conflicts": []})
        with patch("main.sync_calc_royalties", side_effect=[None, belt_error]) as mock_sync:
            response = _post_confirm(client)

        assert response.status_code == 409
        assert response.json()["detail"]["gate"] == "conflict"
        assert mock_sync.call_count == 2
