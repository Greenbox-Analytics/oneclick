"""Tests for the SSE cache-hit path honoring sync gates.

Acceptance criteria:
1. A SyncGateError raised during the cache-hit sync surfaces as a
   `needs_confirmation` SSE event AFTER the `complete` event — the cached
   result is always shown; the gate is a separate, additive signal.
2. The cache-hit sync call passes `contract_ids` equal to the junction set
   that matched the cache (order-insensitive).
3. Any other exception during the cache-hit sync is swallowed — the stream
   still yields `complete` and no `needs_confirmation` event, and does not
   raise.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oneclick.royalties.ledger_sync import SyncGateError
from tests.conftest import MockQueryBuilder, _default_table_side_effect

PROJECT_ID = "proj-0000-0000-0000-0000-000000000001"
CONTRACT_ID = "cont-0000-0000-0000-0000-000000000001"
CONTRACT_ID_2 = "cont-0000-0000-0000-0000-000000000002"
STATEMENT_FILE_ID = "file-0000-0000-0000-0000-000000000001"
CALCULATION_ID = "calc-0000-0000-0000-0000-000000000001"

SAMPLE_PAYMENTS = [
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
]

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


@pytest.fixture(autouse=True)
def _bypass_oneclick_ownership(monkeypatch):
    """Bypass the OneClick ownership guard — this suite tests the cache-hit
    sync/gate wiring, not access control."""
    monkeypatch.setattr(
        "main._assert_can_access_oneclick_inputs",
        AsyncMock(return_value=None),
    )


def _builder(data: list):
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=data, count=len(data))
    return b


def _sub_table(name: str, data_for_others: list):
    if name in _SUBSCRIPTION_TABLES:
        return _default_table_side_effect(name)
    return _builder(data_for_others)


def _sse_events(raw: bytes) -> list[dict]:
    events = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _wire_cache_hit(mock_supabase, contract_ids):
    """Wire mock_supabase so a cache check finds a single calculation whose
    junction rows match *contract_ids* exactly (a cache hit)."""
    cached_result = {
        "status": "success",
        "total_payments": 1,
        "payments": SAMPLE_PAYMENTS,
        "message": "1 payment",
    }
    cached_calc = {"id": CALCULATION_ID, "results": cached_result}
    junction_rows = [{"calculation_id": CALCULATION_ID, "contract_id": cid} for cid in contract_ids]

    call_idx = [0]
    # 1. royalty_calculations select -> [cached_calc]
    # 2. royalty_calculation_contracts select -> junction_rows
    # 3. royalty_statement_rows select (existing-rows check) -> [] (count=0,
    #    triggers the statement-rows self-heal parse, which is harmless here
    #    since sync_calc_royalties itself is patched in every test below)
    sequences = [[cached_calc], junction_rows, []]

    def _side_effect(name):
        if name in _SUBSCRIPTION_TABLES:
            return _default_table_side_effect(name)
        data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
        call_idx[0] += 1
        return _builder(data)

    mock_supabase.table.side_effect = _side_effect


class TestCacheHitSyncGates:
    def _stream(self, client, mock_supabase, contract_ids=None):
        contract_ids = contract_ids or [CONTRACT_ID]
        _wire_cache_hit(mock_supabase, contract_ids)

        with patch("main.calculate_royalty_payments") as mock_calc:
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": contract_ids,
                },
            )
            mock_calc.assert_not_called()

        return response, _sse_events(response.content)

    def test_cache_hit_gate_emits_needs_confirmation(self, client, mock_supabase):
        """A SyncGateError raised on the cache-hit sync path yields both the
        cached `complete` event AND a `needs_confirmation` event afterward."""
        with patch(
            "main.sync_calc_royalties",
            side_effect=SyncGateError("conflict", {"scope": "cross_run", "conflicts": []}),
        ):
            response, events = self._stream(client, mock_supabase)

        assert response.status_code == 200
        types = [e.get("type") for e in events]
        assert "complete" in types
        assert "needs_confirmation" in types
        assert types.index("complete") < types.index("needs_confirmation")

        gate_event = next(e for e in events if e.get("type") == "needs_confirmation")
        assert gate_event["gate"] == "conflict"
        assert gate_event["payload"] == {"scope": "cross_run", "conflicts": []}

    def test_cache_hit_clear_passes_junction_contract_ids(self, client, mock_supabase):
        """sync_calc_royalties is called with contract_ids == the junction set
        the cache matched against (order-insensitive)."""
        contract_ids = [CONTRACT_ID, CONTRACT_ID_2]
        with patch("main.sync_calc_royalties", MagicMock()) as mock_sync:
            response, events = self._stream(client, mock_supabase, contract_ids=contract_ids)

        assert response.status_code == 200
        types = [e.get("type") for e in events]
        assert "complete" in types
        assert "needs_confirmation" not in types

        mock_sync.assert_called_once()
        _, kwargs = mock_sync.call_args
        assert set(kwargs["contract_ids"]) == set(contract_ids)

    def test_cache_hit_sync_error_stream_survives(self, client, mock_supabase):
        """A non-gate exception during the cache-hit sync is swallowed: the
        stream still yields `complete`, no `needs_confirmation`, no raise."""
        with patch("main.sync_calc_royalties", side_effect=RuntimeError("boom")):
            response, events = self._stream(client, mock_supabase)

        assert response.status_code == 200
        types = [e.get("type") for e in events]
        assert "complete" in types
        assert "needs_confirmation" not in types
